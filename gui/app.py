"""
MovieShort AI — Gradio GUI
"""
import glob
import os
import re
import threading
import time as time_module
from typing import List, Tuple

import gradio as gr

import config as app_config
from core.pipeline import process_multiple
from core.batch import process_movie
from core.subtitle import generate_word_group_srt
from utils.ffmpeg_utils import render_full_preview, _probe_duration
from utils.log_capture import LogCapture
from utils import user_config
from utils.font_manager import POPULAR_FONTS, ensure_font, FONTS_DIR
from analyzers.text_analyzer import check_api_key
from utils import fmt_duration


# ---------------------------------------------------------------------------
# Glassmorphism theme (task v17, UI cohesion overhaul).
#
# THE ONE RULE: backdrop-filter ONLY on leaf surfaces — button.primary,
# #header-card, and the dropdown popup ul.options itself. Gradio renders
# dropdown popups as position:fixed inside the component tree; if ANY ancestor
# (.block/.form/.gr-group/.accordion/.tabitem/.wrap) gets backdrop-filter,
# filter, transform or will-change it becomes a containing block → the popup
# is clipped by ancestor overflow and positioned relative to that ancestor
# instead of the viewport. Panels therefore use rgba() fills ("fake glass").
#
# COHESION RULE: individual .block components get NO box of their own (theme
# paints them transparent) — PARENT containers (.form/.gr-group/.accordion/
# .tabitem) are the unified glass panels. One indigo accent family for every
# interactive state: radios, checkboxes, sliders, focus rings, buttons, tabs.
# ---------------------------------------------------------------------------
_GLASS_CSS = """
/* ── Ambient background: clean solid dark (todo 44: gradient blobs removed). ── */
body { background: #191714 !important; }
.gradio-container { position: relative; z-index: 1; background: transparent !important; }

/* ── Theme variable overrides on top of gr.themes.Soft ── */
.gradio-container {
  --block-radius: 16px;
  --radius-lg: 16px;
  --radius-md: 10px;
  --block-title-text-color: #C2C0B6;
  --block-info-text-color: #9C988B;
  --body-text-color: #F5F4EE;
  --body-text-color-subdued: #9C988B;
  --border-color-accent: rgba(204,120,92,.6);
  --input-border-color-focus: #CC785C;
  --checkbox-background-color-selected: #C15F3C;
  --checkbox-border-color-selected: #CC785C;
  --button-primary-text-color: #ffffff;
  --button-secondary-background-fill: rgba(255,255,255,.06);
  --button-secondary-background-fill-hover: rgba(255,255,255,.11);
  --button-secondary-border-color: rgba(255,255,255,.14);
  --button-secondary-text-color: #F5F4EE;
}

/* ── Unified panels: PARENT containers only. Every visible container matches
   #header-card exactly, but via linear-gradient instead of backdrop-filter
   (all of these can be dropdown ancestors → frost/transform/filter here would
   create a containing block and clip position:fixed popups). Child .block
   components stay transparent and blend into the panel. ── */
.gradio-container .form,
.gradio-container .gr-group,
.gradio-container .accordion,
.gradio-container .tabitem {
    background: rgba(26,24,20,.92) !important;
    border: 1px solid rgba(255,255,255,.09) !important;
    border-radius: 20px !important;
    padding: 22px 28px;
    box-shadow: 0 8px 32px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.08);
}
/* Nested boxes kill the unified look: any .block inside a panel is transparent */
.gradio-container .form .block,
.gradio-container .gr-group .block,
.gradio-container .accordion .block,
.gradio-container .tabitem .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* ── Todo 6: ONLY the outermost panel paints. Diagnosis (playwright ancestor
   walk from a checkbox): each component's own .form wrapper, the Group's
   .gr-group AND the .tabitem all matched the panel rule → three stacked
   alpha-composited gradients ("big block with three mini-blocks"), plus the
   Group's inner .styler painted a solid slate strip (the gray underlay).
   Flatten every panel-kind element nested inside a painted panel. ── */
.gradio-container .tabitem .form,
.gradio-container .tabitem .gr-group,
.gradio-container .tabitem .accordion,
.gradio-container .gr-group .form,
.gradio-container .gr-group .gr-group,
.gradio-container .gr-group .accordion,
.gradio-container .accordion .form,
.gradio-container .accordion .gr-group,
.gradio-container .accordion .accordion,
.gradio-container .accordion .tabitem,
.gradio-container .form .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    /* Todo 23: paint was flattened here in todo 6 but the 22/28px padding
       survived, so components nested one level deeper than their siblings
       (console in a .form, slider rows in Group>Form) rendered 28/56px
       narrower than the primary button sitting directly on the panel.
       Invisible containers must not inset content either. */
    padding: 0 !important;
}
/* Gray underlay: Group's .styler wrapper carries a solid theme fill */
.gradio-container .styler {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* ── Todo 38: Gradio merges some Row columns into ONE inner .form (flex,
   gap 0), so field groups render with no gutter while sibling button rows
   keep the Row's 16px — stacked rows on the provider tabs got misaligned
   seams (fields @866 vs buttons @858/874). Restore the standard gutter;
   column-gap is a no-op for single-child and column-direction forms. ── */
.gradio-container .stretch > .form { column-gap: 16px; }

/* ── Todo 8: Accordion disclosure arrow. Rendered DOM (Gradio 4.44): the
   header is button.label-wrap (NOT a <summary>/svg) holding the label span
   plus span.icon with a "▼" glyph; the button is justify-content:
   space-between, which flung the glyph to the far edge (851px from the
   label). Pull content left with a tight gap; recolor/resize the glyph.
   The button itself stays full-width → whole-row clickability preserved. ── */
.gradio-container button.label-wrap { justify-content: flex-start; column-gap: 8px; }
.gradio-container button.label-wrap .icon { color: #D4A27F !important; font-size: 18px; }
.gradio-container button.label-wrap:hover .icon,
.gradio-container button.label-wrap:focus-visible .icon { color: #EBDBBC !important; }
/* ── Todo 31: accordion header styled as a button. Same indigo family as
   tab-selected/secondary hover; disclosure behavior untouched (the button
   stays the click target). Arrow metrics from todo 8 preserved via the
   column-gap rule above — re-measured ≤12px after this change. ── */
.gradio-container button.label-wrap {
    background: rgba(193,95,60,.22) !important;
    border: 1px solid rgba(204,120,92,.5) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    transition: background .2s ease, border-color .2s ease;
}
.gradio-container button.label-wrap:hover {
    background: rgba(193,95,60,.3) !important;
    border-color: rgba(204,120,92,.7) !important;
}

/* ── Header card: leaf surface (no dropdown inside) → real frost allowed ── */
#header-card {
    background: rgba(26,24,20,.92);
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 20px;
    padding: 22px 28px;
    margin-bottom: 10px;
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    box-shadow: 0 8px 32px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.08);
}
#header-card h1 { margin: 0; font-size: 2em; letter-spacing: -.02em; color: #F5F4EE; }
#header-card p { margin: 6px 0 0; color: #9C988B; font-size: .95em; }

/* ── Dropdown popup: glass it up — styling the POPUP itself is always safe.
   Readability: ≥16px font (cascades to li.item), near-opaque fill, explicit
   li text color + generous padding + high-contrast hover. ── */
.gradio-container ul.options {
    font-size: 16px;
    /* Todo 32: .96 (was .92) — open popup must not let panel gradients
       bleed through and weaken choice readability. */
    background: rgba(20,18,16,.97) !important;
    backdrop-filter: blur(18px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(160%) !important;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 12px;
    box-shadow: 0 12px 40px rgba(0,0,0,.55);
    /* Todo 10: predictable scroll. Capped height makes long lists internally
       scrollable; overscroll-behavior:contain stops the wheel from chaining
       out of the popup into the page (the popup itself is a scroll container
       even when its content fits, so chaining is suppressed either way). */
    max-height: 320px !important;
    overflow-y: auto;
    overscroll-behavior: contain;
}
.gradio-container ul.options li.item {
    padding: 10px 14px;
    color: #F5F4EE;
    /* Todo 30: «обводка выбранного текста» — браузерный text-selection
       хайлайт на кликнутом пункте (todo 15 сделал неселектируемым UI-хром,
       но не li списков) + потенциальный UA focus-ring на активном пункте.
       Снимаем и то, и другое; hover-подсветка фоном остаётся. */
    -webkit-user-select: none !important;
    user-select: none !important;
    outline: none !important;
    outline-width: 0 !important;
    box-shadow: none !important;
}
.gradio-container ul.options li.item:hover,
.gradio-container ul.options li.item.active { background: rgba(193,95,60,.35) !important; }
/* ── Todo 39 v2: the SELECTED item stays neutral even while hovered or
   keyboard-active (.active) — no accent wash, no outline/glow. These
   selectors out-specify the hover/active rule above, so the !important
   tie resolves in favour of the neutral fill. Non-selected items keep
   the hover affordance. ── */
.gradio-container ul.options li.item.selected,
.gradio-container ul.options li.item.selected:hover,
.gradio-container ul.options li.item.selected.active {
    background: rgba(255,255,255,.06) !important;
    color: #F5F4EE !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ── Inputs: dark fill (todo 12: transparent fill made fields merge into the
   panel gradient) + indigo focus ring. input:not([type]): Gradio renders
   Textbox/Dropdown <input> WITHOUT a type attribute (IDL default "text") —
   audited live: the only typeless inputs are text-kind. ── */
.gradio-container input[type="text"],
.gradio-container input:not([type]),
.gradio-container input[type="number"],
.gradio-container input[type="password"],
.gradio-container textarea {
    background: rgba(22,20,18,.60) !important;
    color: #F5F4EE !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 10px !important;
}
/* Closed Dropdown field: rendered DOM paints NOTHING (input transparent AND
   BlockWrapper .wrap border-width 0 / bg transparent). Gradio 4.44 structure:
   .block > .container > .wrap > .wrap-inner > .secondary-wrap > input.
   Paint the frame element itself; :has() keeps the generic ".wrap" class from
   matching unrelated layout wrappers. Inner input stays transparent to avoid
   double alpha-compositing over the frame fill. */
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) {
    background: rgba(22,20,18,.60) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 10px !important;
}
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) > .wrap-inner > .secondary-wrap > input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    text-shadow: none !important;
    color: #F5F4EE !important;
}
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) .secondary-wrap,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) .wrap-inner {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* Value span variant if Gradio renders value as span (token) inside secondary-wrap */
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) .secondary-wrap span {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #F5F4EE !important;
}
/* ── Todo 12: checkboxes. Rendered DOM: bare native input (appearance:none,
   16x16) + text span inside <label>; Gradio paints it via theme vars and the
   unchecked box vanished against the panel. Explicit box: dark fill + light
   border unchecked, indigo checked (accent-color above stays a fallback).
   !important required: Gradio's svelte-scoped var-driven rules otherwise win
   the tie on cascade order for the unchecked state. ── */
.gradio-container input[type="checkbox"] {
    width: 16px !important;
    height: 16px !important;
    min-width: 16px !important;
    border: 1px solid rgba(255,255,255,.28) !important;
    border-radius: 4px !important;
    background-color: rgba(22,20,18,.60) !important;
}
.gradio-container input[type="checkbox"]:checked {
    background-color: #C15F3C !important;
    border-color: #CC785C !important;
}
/* ── Todo 32/46/8-R7: radios — WCAG AA on #191714. Live diagnose:
    inactive was 1px rgba(255,255,255,.28) + transparent bg + span #F5F4EE
    for all states (R5 pinned active color to every label). Fixed per spec:
    inactive circle 2px #9C988B bg rgba(28,26,22,.6) label #C2C0B6 14px;
    active circle 2px #C15F3C bg #C15F3C + white dot label #F5F4EE 14px bold;
    hover inactive: border #D4A27F + label #F5F4EE. Preserves R5 active token. ── */
.gradio-container input[type="radio"] {
    width: 16px !important;
    height: 16px !important;
    min-width: 16px !important;
    border: 2px solid #9C988B !important;
    border-radius: 50% !important;
    background-color: rgba(28,26,22,.6) !important;
}
.gradio-container input[type="radio"]:checked {
    background-color: #C15F3C !important;
    border-color: #C15F3C !important;
    background-image: radial-gradient(closest-side, #fff 40%, transparent 41%) !important;
}
.gradio-container label:has(> input[type="radio"]:not(:checked)) > span {
    color: #C2C0B6 !important;
    font-size: 14px !important;
    font-weight: 400 !important;
}
.gradio-container label:has(> input[type="radio"]:checked) > span {
    color: #F5F4EE !important;
    font-size: 14px !important;
    font-weight: 700 !important;
}
/* ── Radio PILL container (not just dot): inactive pill must be visible, not just circle. Gradio renders each option as <label><input><span>. The label is the pill. ── */
.gradio-container label:has(> input[type="radio"]:not(:checked)) {
    background: rgba(32,30,26,.75) !important;
    border: 1px solid rgba(156,152,139,.32) !important;
    border-radius: 999px !important;
    padding: 7px 14px !important;
}
.gradio-container label:has(> input[type="radio"]:checked) {
    background: #C15F3C !important;
    border: 1px solid #C15F3C !important;
    border-radius: 999px !important;
    padding: 7px 14px !important;
}
/* Hover inactive: border #D4A27F + label #F5F4EE — both input-hover and label-hover paths, label:hover covers the whole option row (input is only 16px). */
.gradio-container input[type="radio"]:not(:checked):hover {
    border-color: #D4A27F !important;
}
.gradio-container label:has(> input[type="radio"]:not(:checked)):hover {
    background: rgba(38,36,32,.85) !important;
    border-color: #D4A27F !important;
}
.gradio-container label:has(> input[type="radio"]:not(:checked)):hover > span {
    color: #F5F4EE !important;
}
.gradio-container label:has(> input[type="radio"]:not(:checked)):hover input[type="radio"] {
    border-color: #D4A27F !important;
}
/* Extra: plain label:hover (no :has guard) as fallback for older selector engines — inactive only, active already #F5F4EE so no harm but guard via :not(:checked) above is primary. */
.gradio-container label:hover:has(> input[type="radio"]:not(:checked)) > span {
    color: #F5F4EE !important;
}
/* ── Todo 15: click-selection artifacts removed, focus is keyboard-first.
   The old blanket .wrap:focus-within lit a 3px glow around every clicked
   field AND around whole layout sections (any .wrap ancestor matches).
   CSS spec caveat discovered live: :focus-visible ALWAYS matches text-entry
   fields (input/textarea) even after mouse clicks, so no pure-CSS selector
   can give text fields a keyboard-only ring. Split therefore:
   - buttons/tabs/checkboxes/radios: real keyboard-only outline rings;
   - text fields: subtle border tint on focus (no glow, no outline) —
     standard caret-field affordance, identical for keyboard and mouse.
   # ponytail: if the border tint on clicked text fields is ever rejected,
     upgrade path is a 5-line JS keydown-heuristic toggling a .kb-focus class. ── */
.gradio-container .wrap:focus-within {
    border-color: rgba(204,120,92,.7) !important;
}
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input):focus-within {
    border-color: rgba(204,120,92,.7) !important;
}
.gradio-container button:focus-visible,
.gradio-container input[type="checkbox"]:focus-visible,
.gradio-container input[type="radio"]:focus-visible {
    outline: 2px solid #CC785C !important;
    outline-offset: 2px;
}
/* Click artifacts: no tap highlight; text selection only where it makes
   sense. UI chrome (buttons incl. tab buttons and accordion headers,
   component labels) is not selectable; console/log/output text lives in
   textarea/input/HTML blocks and stays selectable. */
.gradio-container { -webkit-tap-highlight-color: transparent; }
.gradio-container button,
.gradio-container label,
.gradio-container span[data-testid="block-info"] {
    -webkit-user-select: none;
    user-select: none;
}
/* ── Todo 48: field label chips must be FLAT TEXT. Diagnosis: Gradio's own
   span.svelte-1gfkn6j paints every block-info with the theme's
   --block-title-background-fill → an opaque orange pill (rgb(234,88,12),
   padding 4px 6px, radius 6px) sitting above each closed dropdown/input and
   reading as a selection highlight. Flatten: no bg/border/radius/padding.
   Accordion headers are button.label-wrap — untouched by this rule. ── */
.gradio-container span[data-testid="block-info"] {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}
/* ── Todo 48: closed-dropdown arrow unobtrusive — muted khaki glyph, no fill
   of its own (was bright #F5F4EE, same as value text → read as selected). ── */
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) .icon-wrap {
    background: transparent !important;
    color: #9C988B !important;
}
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) .icon-wrap svg,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) .icon-wrap svg path {
    color: #9C988B !important;
    fill: #9C988B !important;
    stroke: #9C988B !important;
}
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input):hover .icon-wrap,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input):hover .icon-wrap svg,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input):hover .icon-wrap svg path,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input):focus-within .icon-wrap,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input):focus-within .icon-wrap svg,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input):focus-within .icon-wrap svg path {
    color: #9C988B !important;
    fill: #9C988B !important;
    stroke: #9C988B !important;
}
.gradio-container ::selection { background: rgba(193,95,60,.35); }
/* ── Todo 51 v4: closed Dropdown ::selection must be invisible. The global
    ::selection above paints selected text with crail accent — on a closed
    Dropdown that means the entire value ("Bebas Neue") flashes selected when
    the input is auto-focused on click. Override to transparent for the closed
    Dropdown's input/span only; open-popup search highlighting is unaffected
    because it uses ul.options li, not this input. ── */
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) input::selection,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) input::-moz-selection,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) span::selection,
.gradio-container .wrap:has(> .wrap-inner > .secondary-wrap > input) span::-moz-selection {
    background: transparent !important;
    color: #F5F4EE !important;
}
.gradio-container ::placeholder { color: #7A766B !important; }
.gradio-container input[type="range"],
.gradio-container input[type="checkbox"],
.gradio-container input[type="radio"] { accent-color: #CC785C; }
/* Todo 15 residual (F6): editable fields keep a keyboard-visible ring,
   and log/output areas stay mouse-selectable despite gradio's own
   user-select:none on disabled textareas. The label/.form selector
   variants out-specify svelte's `textarea.svelte-* { outline:none
   !important }`, which otherwise ties (0,2,1) and wins by order. */
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container label input:focus-visible,
.gradio-container label textarea:focus-visible,
.gradio-container .form input:focus-visible,
.gradio-container .form textarea:focus-visible {
    outline: 2px solid #CC785C !important;
    outline-offset: 1px !important;
}
.gradio-container textarea[disabled],
.gradio-container label textarea[disabled],
.gradio-container #console-log textarea {
    user-select: text !important;
    -webkit-user-select: text !important;
}

/* ── Buttons: primary = flat (gradients removed) + real frost (leaf surface); ghost secondary ── */
.gradio-container button.primary {
    background: #C15F3C !important;
    border: none !important;
    color: #fff !important;
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    transition: transform .15s ease, box-shadow .15s ease;
}
.gradio-container button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(193,95,60,.45) !important;
}
.gradio-container button.secondary {
    background: rgba(255,255,255,.06) !important;
    border: 1px solid rgba(255,255,255,.14) !important;
    color: #F5F4EE !important;
    /* Pin line-height: inherited value varies by tab context (1.4 vs 1.5)
       and made this button 2px taller than its siblings. */
    line-height: 1.4 !important;
    transition: background .2s ease, box-shadow .2s ease, transform .15s ease;
}
.gradio-container button.secondary:hover {
    background: rgba(255,255,255,.11) !important;
    box-shadow: 0 4px 16px rgba(193,95,60,.25) !important;
    transform: translateY(-1px);
}
/* ── Todo 24: queue button aligns with the movie-title FIELD, not its label.
   Row children: DIV.form(.block.padded = label + 42px input) + this button.
   Diagnosis: label no longer wraps (row 97.6px = label 27.6 + input 42 +
   block padding 2×10), so the fix is alignment, not de-wrapping: pin the
   button to the row bottom, cancel the field block's 10px bottom padding
   (button bottom == input bottom == row bottom), pin height to the input's
   measured 42px. Label gets nowrap+ellipsis so a long i18n string can never
   re-inflate the row (the old 144px bug). Right edge already flush (flex
   scale=1 → button.right == panel content edge). line-height 1.4 kept. ── */
#add-queue-btn {
    align-self: flex-end !important;
    height: 42px !important;
    min-height: 0 !important;
}
.gradio-container .form:has(+ #add-queue-btn),
.gradio-container .form:has(+ #add-queue-btn) .block.padded {
    padding-bottom: 0 !important;
}
.gradio-container .form:has(+ #add-queue-btn) span[data-testid="block-info"] {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Tabs: ghost inactive, accent active (.tab-container = Gradio 4.44,
   .tab-nav = older/newer layouts). Radius on ALL corners (panel-consistent). ── */
.gradio-container .tab-nav,
.gradio-container .tab-container {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(255,255,255,.06) !important;
    border-radius: 20px !important;
    gap: 4px;
}
.gradio-container .tab-nav button,
.gradio-container .tab-container button {
    color: #9C988B !important;
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
}
.gradio-container .tab-nav button.selected,
.gradio-container .tab-container button.selected {
    color: #ffffff !important;
    background: rgba(193,95,60,.22) !important;
    box-shadow: inset 0 -2px 0 #CC785C !important;
}

footer { display: none; }

/* Gallery items on dark */
.gradio-container .gallery-item {
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    background: rgba(24,22,19,.5) !important;
}

/* ── Todo 33: preview video capped. A 9:16 render is taller than the
   viewport → without a cap the page scrolls sideways into nowhere and the
   player stretches the layout. 70vh ceiling, intrinsic ratio preserved via
   width:auto, centered in its container; container itself never exceeds
   the panel so no page stretch. ── */
.preview-video video {
    max-height: 70vh;
    width: auto;
    max-width: 100%;
    display: block;
    margin: 0 auto;
}
.preview-video[data-testid="video"] > div,
.preview-video > div { overflow: hidden; }

/* Scrollbars */
*::-webkit-scrollbar { width: 8px; height: 8px; }
*::-webkit-scrollbar-thumb { background: rgba(255,255,255,.14); border-radius: 8px; }
*::-webkit-scrollbar-thumb:hover { background: rgba(204,120,92,.5); }

/* ── Todo 50 v4 widths: full inventory per screenshots (1920px viewport).
    BEFORE: outer .block widths already == panelContent (1414) but inner
    insets differed — hint Markdown pad 0 (x253) vs .tc-range textarea
    pad12 (input x265/1390) vs gr-group heading pad0 (x253) vs checkbox
    rows pad12 (x253 block but label at x265). Visually three different
    left edges. Group content not unified. Subtitle dropdown outer
    already == panelContent (1390) — verified, no fix needed there.
    AFTER: force listed controls to 100% of container and unify group
    content to 12px horizontal padding. Multi-column rows (banner Top/
    Bottom 699px, Size/Outline 687px) remain half-width by design — documented
    exception: flex Row columns fill 50% each + 16px gap (todo 38). ── */
.gradio-container .tc-range.block {
    width: 100% !important;
    box-sizing: border-box !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
}
.gradio-container .tc-range.block textarea {
    width: 100% !important;
    box-sizing: border-box !important;
}
/* Hint Markdown block (hide-container, pad 0 before → unify to 12).
   Specificity bump: .padded is present on the element (class="block ... padded hide-container")
   so include it to outrank theme's .block.padded rule. */
.gradio-container .tabitem .block.hide-container.padded,
.gradio-container .tabitem .block.hide-container {
    padding-left: 12px !important;
    padding-right: 12px !important;
    box-sizing: border-box !important;
}
/* Group content unified to 12px (heading was 0, rows were 12) */
.gradio-container .gr-group .block,
.gradio-container .gr-group .block.hide-container {
    padding-left: 12px !important;
    padding-right: 12px !important;
    box-sizing: border-box !important;
}
/* ── Todo 11 R7b-3: hide Gradio built-in top progress bar that flickers
    on every yield (0.3→0.6s debounce below). _make_progress_html is the
    only progress UI we want. ── */
.gradio-container .progress,
.gradio-container .progress-bar,
.gradio-container .wrap .progress {
    display: none !important;
}
"""

# Force Gradio's dark palette regardless of OS/browser preference: first run
# redirects to ?__theme=dark, after the reload the guard makes it a no-op.
# The same function installs the todo-22 delegated 'input' listener masking
# .tc-range fields client-side — digits accumulate through the template
# DDDDDD-DDDDDD ("000130000230" -> "00:01:30 - 00:02:30"); delegation on
# document covers dynamically revealed fields. Mirrors _mask_timecode_range(),
# which stays authoritative.
# Todo 10 addition: Gradio's dropdown popup (ul.options) is position:fixed and
# does NOT follow its anchor when the page scrolls under it -> the open list
# visually tears away from its field. CSS overscroll-behavior stops the wheel
# from chaining INTO the page while the cursor is over the list, but a scroll
# started elsewhere still moves the page. Pure CSS can neither reposition nor
# close the popup, so on the first scroll tick we synthesize Escape on the
# dropdown input — Gradio closes the list natively (graceful, no tear).
# NOTE: Gradio compiles this string as `await (${JS})();` — it MUST be a
# single function expression, not statements/IIFE.
FORCE_DARK_JS = """
async () => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
    // CAPTURE phase: must run BEFORE Gradio's own input binding, otherwise
    // svelte re-binds the raw unmasked value over ours.
    document.addEventListener('input', function(e) {
        const el = e.target;
        if (!el || (el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA')) return;
        // Gradio puts elem_classes on the component's wrapper div, not the
        // native input — resolve via closest().
        if (!el.closest('.tc-range')) return;
        // Todo 22 range mask: digits are re-extracted from the value on every
        // input, so backspace and pasted full ranges ("00:01:30 - 00:02:30")
        // reflow naturally through the DDDDDD-DDDDDD template.
        const d = (el.value || '').replace(/\\D/g, '').slice(0, 12);
        const stamp = s => s.replace(/(..)/g, '$1:').replace(/:$/, '');
        let out = stamp(d.slice(0, 6));
        if (d.length > 6) out += ' - ' + stamp(d.slice(6));
        el.value = out;  // caret jumps to end after reformat — accepted behavior
        if (d.length === 12) {
            // Autofocus next field. The server reveal (.change chain) can take
            // >1s — longer than any fixed poll — so watch DOM mutations
            // instead and focus the moment the next .tc-range wrapper renders
            // visible. Gradio's own value-sync then yanks focus back to the
            // edited field (implicit focus via setSelectionRange — verified
            // live: focusout with zero .focus() calls), so re-assert briefly;
            // every watcher self-stops (5s ceiling), nothing leaks.
            const wraps = Array.prototype.slice.call(document.querySelectorAll('.tc-range'));
            const nxt = wraps[wraps.indexOf(el.closest('.tc-range')) + 1];
            if (nxt) {
                // Gradio 4.44 renders Textbox as <textarea>, not <input>.
                const tryFocus = function() {
                    const visible = nxt.offsetParent !== null ||
                        nxt.getBoundingClientRect().height > 0;
                    if (!visible) return false;
                    const inp = nxt.querySelector('input,textarea');
                    if (inp && document.activeElement !== inp) inp.focus();
                    return !!inp;
                };
                let ticks = 0;
                const guard = setInterval(function() {
                    const ae = document.activeElement;
                    // Focus settled in ANY .tc-range (ours, or the user
                    // deliberately moved on) -> stand down; 50x100ms ceiling.
                    if ((ae && ae.closest && ae.closest('.tc-range')) ||
                        ++ticks >= 50) { clearInterval(guard); return; }
                    tryFocus();
                }, 100);
                if (!tryFocus()) {
                    const mo = new MutationObserver(function() {
                        if (tryFocus()) mo.disconnect();
                    });
                    mo.observe(document.body,
                        { childList: true, subtree: true, attributes: true });
                    setTimeout(function() { mo.disconnect(); }, 5000);
                }
            }
        }
    }, true);
    // Todo 10: close any open dropdown popup as soon as the page scrolls.
    document.addEventListener('scroll', function() {
        const ul = document.querySelector('ul.options');
        if (!ul) return;
        const block = ul.closest('.block');
        const inp = (block && block.querySelector('input')) || document.activeElement;
        if (inp) {
            inp.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Escape', code: 'Escape', keyCode: 27, which: 27, bubbles: true
            }));
        }
    }, { passive: true });
}
"""


def parse_timestamps(text: str) -> List[Tuple[str, str]]:
    pairs = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r"(\d{1,2}:\d{2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2}:\d{2})", line
        )
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


def _mask_timecode_range(text) -> str:
    """Range timecode mask (todo 22): keep digits only, max 12; first 6
    digits form "HH:MM:SS", the next 6 form the second stamp after " - ".

    "000130000230"->"00:01:30 - 00:02:30",
    "123456000030"->"12:34:56 - 00:00:30"; partial states mask by position
    ("1234"->"12:34", "1234560"->"12:34:56 - 0"); garbage stripped the same
    way. The client-side JS mask mirrors this exact algorithm; this server
    version stays authoritative.
    """
    digits = re.sub(r"\D", "", str(text or ""))[:12]
    if not digits:
        return ""

    def stamp(d):
        return ":".join(d[i:i + 2] for i in range(0, len(d), 2))

    first = stamp(digits[:6])
    rest = digits[6:]
    if not rest:
        return first
    return f"{first} - {stamp(rest)}"


# ---------------------------------------------------------------------------
# UI language dictionary
# ---------------------------------------------------------------------------

UI = {
    "ru": {
        "app_tagline": "Автоматическая нарезка фильмов на YouTube Shorts",
        "tab_manual": "Ручной режим",
        "tab_auto": "Автоматический режим",
        "settings": "Настройки",
        "process": "Обработать",
        "auto_process": "Анализировать и обработать очередь",
        "file_label": "Выберите видео",
        "movie_title": "Название фильма (необязательно)",
        "tc_range": "Таймкод",
        "tc_hint": "Таймкоды добавляются полями: заполнили интервал — появится следующее поле",
        "max_len": "Макс. длина клипа (сек)",
        "processing_opts": "Опции монтажа",
        "subs_label": "Субтитры",
        "subs_info": "Распознаёт речь и накладывает субтитры на видео",
        "face_label": "Smart centering (Face tracking)",
        "face_info": "Анализирует положение лиц и центрирует кадр по ним",
        "banner_label": "Баннерные поля",
        "banner_info": "Добавляет поля сверху/снизу для баннеров в YouTube редакторе",
        "banner_top": "Верхнее поле (px)",
        "banner_bottom": "Нижнее поле (px)",
        "blur_label": "Размытый фон",
        "blur_info": "Заполняет пустое пространство размытой копией видео (формат 9:16)",
        "anti_label": "Anti-copyright",
        "anti_info": "Небольшие искажения (отражение, контраст, яркость) для обхода Content ID",
        "film_language": "Язык фильма",
        "film_language_info": "Язык диалогов в фильме. Определяет язык транскрипции и субтитров.",
        "llm_provider": "LLM Провайдер",
        "analysis_mode": "Режим анализа",
        "analysis_mode_info": "Стандартный: детекция сцен + LLM скоринг | Контекстный: LLM видит текст всех сцен → сам выбирает лучшие",
    "analysis_mode_std": "Стандартный",
    "analysis_mode_ctx": "Контекстный (ИИ)",
        "min_len": "Мин. длина (сек)",
        "num_clips": "Количество клипов",
        "score_threshold": "Порог оценки",
        "score_threshold_info": "Минимальная оценка сцены (1-10) для включения в результат",
        "waiting": "Ожидание запуска...",
        "error_no_file": "Ошибка: загрузите видеофайл.",
        "error_no_ts": "Ошибка: укажите хотя бы один промежуток",
        "console": "Консоль отладки",
        "subtitle_editor": "Редактор субтитров",
        "font": "Шрифт",
"sub_font_label": "Шрифт субтитров",
  "font_no_cyr_suffix": " (без кириллицы)",
  "font_size": "Размер",
        "font_color": "Цвет",
        "outline": "Обводка",
        "bold": "Жирный",
        "italic": "Курсив",
        "shadow": "Тень",
        "position_y": "Отступ от низа (px)",
        "preview": "Предпросмотр",
        "preview_text": "Пример текста субтитров",
        "test_video": "Тестовое видео",
        "preview_no_video": "Тестовое видео не найдено — положите файл (mp4/avi/mkv) в output/test_video/",
        "preview_bad_path": "Файл не найден: {path}",
        "preview_ok": "✅ Предпросмотр обновлён",
        "preview_fail": "❌ Ошибка предпросмотра: {err}",
        "preview_long_warning": "⏳ Рендер полного ролика может занять несколько минут…",
        "auto_cleanup": "Авто-очистка temp",
        "auto_cleanup_info": "Удалять временные файлы после обработки каждого фильма",
        "ui_language": "Язык интерфейса",
        "batch_files": "Выберите фильмы (можно несколько)",
        "process_all": "Обработать все",
        "process_queue": "Очередь обработки",
        "wait_start": "Ожидание запуска...",
        "apply_lang": "Применить язык",
        "save": "Сохранить",
        "general": "Общие",
        "not_saved": "не сохранено",
        "saved_ok": "✅ Настройки сохранены",
        "saved_sub_ok": "✅ Настройки субтитров сохранены",
        "reset_defaults": "Сбросить к дефолтным",
        "reset_sub_ok": "✅ Настройки субтитров сброшены к заводским",
        "add_to_queue": "Добавить в очередь",
        "queue_empty": "Очередь пуста",
        "restart_for_lang": "✅ Настройки сохранены (перезапустите для смены языка)",
        "support_author": "Поддержать автора:",
        "file_label_suffix": " — перетащите или нажмите для выбора",
        "movie_placeholder": "Например: Матрица, 1+1, Побег из Шоушенка...",
        "queue_header": "Очередь ({n})",
        "llm_provider_info": "Gemini (Google AI), Yandex AI Studio, OpenRouter или OpenCode Zen",
        "lang_russian": "Русский",
        "lang_english": "English",
        "tab_gemini": "Google AI (Gemini)",
        "tab_yandex": "Yandex AI Studio",
        "tab_openrouter": "OpenRouter",
        "tab_opencode_zen": "OpenCode Zen",
        "opencode_zen_model_label": "Модель OpenCode Zen",
        "opencode_zen_model_info": "",
        "api_key_label": "API ключ {provider}",
        "save_key": "Сохранить ключ",
        "check_key": "Проверить ключ",
        "status_not_checked": "⏳ не проверен",
        "folder_id_label": "Folder ID",
        "yandex_model_label": "Модель Yandex",
        "openrouter_model_label": "Модель OpenRouter",
        "no_api_key_title": "Без API-ключа",
        "no_api_key_desc": "Приложение будет работать, но сцены будут выбираться случайно — без анализа содержания.",
        "no_api_key_works": "Что работает:",
        "no_api_key_works_list": "детекция сцен, транскрипция (Whisper), нарезка клипов, субтитры, face tracking",
        "no_api_key_lost": "Что теряется:",
        "no_api_key_lost_list": "интеллектуальный отбор лучших сцен (LLM скоринг), контекстный режим, генерация названий моментов",
        "no_api_key_where": "Где взять ключ:",
        "color_white": "Белый",
        "color_yellow": "Жёлтый",
        "color_black": "Чёрный",
        "color_red": "Красный",
        "color_cyan": "Голубой",
        "color_green": "Зелёный",
        "file_label_x": "Файл: {name}",
        "timestamps_count": "Таймкодов: {n}",
        "options_label": "Опции: subs={subs}, face={face}, blur={blur}, anti={anti}",
        "processing_elapsed": "Обработка... прошло {time}",
        "progress_done": "Готово: {done}/{total} клипов",
        "progress_error": "Ошибка!",
        "update_available": "Доступна новая версия {tag}",
        "watch_release": "Смотреть",
        "error_generic": "ОШИБКА: {msg}",
        "error_short": "Ошибка!",
        "done_count": "Готово: {ok}/{total} клипов",
        "clip_error": "Клип {n}: ошибка обработки",
        "done_count_files": "Готово: {ok}/{total} клипов (файлов: {files})",
        "no_results": "Ошибка: нет результата",
        "no_scenes": "Не найдено подходящих сцен.",
        "no_scenes_short": "Не найдено сцен",
        "total_time": "⏱  Общее время: {time}",
        "movie_header": "MOVIESHORT AI — ФАЙЛ {i}/{total}",
        "movie_file": "Файл: {name}",
        "movie_title_label": "Фильм: {title}",
        "provider_mode": "Провайдер: {prov}, Режим: {mode}",
        "film_lang_label": "Язык: {lang}",
        "no_api_key_warn": "⚠️  API-ключ не задан — сцены будут выбраны случайно!",
        "mode_context": "Контекстный",
        "mode_standard": "Стандартный",
        "processing_file": "[{i}/{total}] {name} — прошло {time}",
        "key_saved": "API ключ сохранён ({provider})",
        "key_saved_check": "Сохранено. Проверка: {error}",
        "api_ok": "✅ API работает",
        "key_valid_quota": "⚠️ Ключ валиден, но {error}",
        "api_error": "❌ {error}",
        "reset_to_defaults": "Сбросить к дефолтным",
        "status_unknown": "статус неизвестен",
    },
    "en": {
        "app_tagline": "Automatic movie clipping for YouTube Shorts",
        "tab_manual": "Manual mode",
        "tab_auto": "Automatic mode",
        "settings": "Settings",
        "process": "Process",
        "auto_process": "Analyze & process queue",
        "file_label": "Select video file",
        "movie_title": "Movie title (optional)",
        "tc_range": "Timecode",
        "tc_hint": "Timecodes are added progressively: complete an interval and the next field appears",
        "max_len": "Max clip length (s)",
        "processing_opts": "Processing options",
        "subs_label": "Subtitles",
        "subs_info": "Recognizes speech and overlays subtitles on video",
        "face_label": "Smart centering (Face tracking)",
        "face_info": "Analyzes face positions and centers the frame on them",
        "banner_label": "Banner padding",
        "banner_info": "Adds top/bottom padding for banners in YouTube editor",
        "banner_top": "Top padding (px)",
        "banner_bottom": "Bottom padding (px)",
        "blur_label": "Blurred background",
        "blur_info": "Fills empty space with a blurred copy of the video (9:16 format)",
        "anti_label": "Anti-copyright",
        "anti_info": "Subtle transformations (mirror, contrast, brightness) to bypass Content ID",
        "film_language": "Film language",
        "film_language_info": "Language of dialogue in the movie. Determines transcription and subtitle language.",
        "llm_provider": "LLM Provider",
        "analysis_mode": "Analysis mode",
        "analysis_mode_info": "Standard: scene detection + LLM scoring | Context: LLM sees all scene transcripts → picks best",
    "analysis_mode_std": "Standard",
    "analysis_mode_ctx": "Context (AI)",
        "min_len": "Min length (s)",
        "num_clips": "Number of clips",
        "score_threshold": "Score threshold",
        "score_threshold_info": "Minimum scene score (1-10) to include in results",
        "waiting": "Waiting...",
        "error_no_file": "Error: please upload a video file.",
        "error_no_ts": "Error: please enter at least one time range",
        "console": "Debug console",
        "subtitle_editor": "Subtitle Editor",
        "font": "Font",
"sub_font_label": "Subtitle font",
  "font_no_cyr_suffix": " (no Cyrillic)",
  "font_size": "Size",
        "font_color": "Color",
        "outline": "Outline",
        "bold": "Bold",
        "italic": "Italic",
        "shadow": "Shadow",
        "position_y": "Bottom offset (px)",
        "preview": "Preview",
        "preview_text": "Subtitle text preview",
        "test_video": "Test video",
        "preview_no_video": "No test video found — drop a file (mp4/avi/mkv) into output/test_video/",
        "preview_bad_path": "File not found: {path}",
        "preview_ok": "✅ Preview updated",
        "preview_fail": "❌ Preview failed: {err}",
        "preview_long_warning": "⏳ Rendering the full clip may take several minutes…",
        "auto_cleanup": "Auto-cleanup temp",
        "auto_cleanup_info": "Delete temporary files after processing each movie",
        "ui_language": "Interface language",
        "batch_files": "Select movie files (multiple allowed)",
        "process_all": "Process all",
        "process_queue": "Processing queue",
        "wait_start": "Waiting...",
        "apply_lang": "Apply language",
        "save": "Save",
        "general": "General",
        "not_saved": "not saved",
        "saved_ok": "✅ Settings saved",
        "saved_sub_ok": "✅ Subtitle settings saved",
        "reset_defaults": "Reset to defaults",
        "reset_sub_ok": "✅ Subtitle settings reset to defaults",
        "add_to_queue": "Add to queue",
        "queue_empty": "Queue is empty",
        "restart_for_lang": "✅ Settings saved (restart to apply language)",
        "support_author": "Support the author:",
        "file_label_suffix": " — drag & drop or click to select",
        "movie_placeholder": "e.g. The Matrix, The Shawshank Redemption...",
        "queue_header": "Queue ({n})",
        "llm_provider_info": "Gemini (Google AI), Yandex AI Studio, OpenRouter or OpenCode Zen",
        "lang_russian": "Russian",
        "lang_english": "English",
        "tab_gemini": "Google AI (Gemini)",
        "tab_yandex": "Yandex AI Studio",
        "tab_openrouter": "OpenRouter",
        "tab_opencode_zen": "OpenCode Zen",
        "opencode_zen_model_label": "OpenCode Zen Model",
        "opencode_zen_model_info": "",
        "api_key_label": "{provider} API key",
        "save_key": "Save key",
        "check_key": "Check key",
        "status_not_checked": "⏳ not checked",
        "folder_id_label": "Folder ID",
        "yandex_model_label": "Yandex Model",
        "openrouter_model_label": "OpenRouter Model",
        "no_api_key_title": "Without API key",
        "no_api_key_desc": "The app will still work, but scenes will be selected randomly — without content analysis.",
        "no_api_key_works": "What works:",
        "no_api_key_works_list": "scene detection, transcription (Whisper), clip cutting, subtitles, face tracking",
        "no_api_key_lost": "What is lost:",
        "no_api_key_lost_list": "intelligent scene selection (LLM scoring), context mode, scene title generation",
        "no_api_key_where": "Where to get the key:",
        "color_white": "White",
        "color_yellow": "Yellow",
        "color_black": "Black",
        "color_red": "Red",
        "color_cyan": "Cyan",
        "color_green": "Green",
        "file_label_x": "File: {name}",
        "timestamps_count": "Timestamps: {n}",
        "options_label": "Options: subs={subs}, face={face}, blur={blur}, anti={anti}",
        "processing_elapsed": "Processing... elapsed {time}",
        "progress_done": "Done: {done}/{total} clips",
        "progress_error": "Error!",
        "update_available": "New version available {tag}",
        "watch_release": "View",
        "error_generic": "ERROR: {msg}",
        "error_short": "Error!",
        "done_count": "Done: {ok}/{total} clips",
        "clip_error": "Clip {n}: processing error",
        "done_count_files": "Done: {ok}/{total} clips (files: {files})",
        "no_results": "Error: no result",
        "no_scenes": "No suitable scenes found.",
        "no_scenes_short": "No scenes found",
        "total_time": "⏱  Total time: {time}",
        "movie_header": "MOVIESHORT AI — FILE {i}/{total}",
        "movie_file": "File: {name}",
        "movie_title_label": "Movie: {title}",
        "provider_mode": "Provider: {prov}, Mode: {mode}",
        "film_lang_label": "Language: {lang}",
        "no_api_key_warn": "⚠️  No API key set — scenes will be selected randomly!",
        "mode_context": "Context",
        "mode_standard": "Standard",
        "processing_file": "[{i}/{total}] {name} — elapsed {time}",
        "key_saved": "API key saved ({provider})",
        "key_saved_check": "Saved. Check: {error}",
        "api_ok": "✅ API is working",
        "key_valid_quota": "⚠️ Key is valid, but {error}",
        "api_error": "❌ {error}",
        "reset_to_defaults": "Reset to defaults",
        "status_unknown": "status unknown",
    },
}

LANG_RU = "ru"
LANG_EN = "en"


def _t(key, lang="ru", default=None):
    """Translate a UI string by key and language.

    Args:
        key: translation key
        lang: language code ('ru' or 'en')
        default: fallback if key not found (defaults to key itself)
    """
    fallback = key if default is None else default
    return UI.get(lang, UI["ru"]).get(key, fallback)


def _get_font_style(cfg):
    """Build font_style dict from user config."""
    return {
        "font": cfg.get("subtitle_font", "Arial"),
        "size": cfg.get("subtitle_size", 13),
        "color": cfg.get("subtitle_color", "&H00FFFFFF"),
        "outline": cfg.get("subtitle_outline", 1),
        "bold": cfg.get("subtitle_bold", True),
        "italic": cfg.get("subtitle_italic", False),
        "shadow": cfg.get("subtitle_shadow", False),
        "position_y": cfg.get("subtitle_position_y", 400),
    }


def _make_progress_html(pct: float, label: str = "") -> str:
    """Build an inline HTML progress bar with label."""
    # Same crail→kraft gradient as button.primary — one accent family everywhere.
    pct_clamped = max(0, min(100, pct))
    escaped_label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    bar_style = (
        f"width:{pct_clamped:.0f}%;height:100%;"
        f"background:#C15F3C;border-radius:12px;"
        f"transition:width 0.5s ease;"
    )
    return f"""<div style="margin:12px 0;">
  <div style="display:flex;justify-content:space-between;font-size:13px;color:#C2C0B6;margin-bottom:2px;">
    <span>{escaped_label}</span>
    <span style="font-weight:bold;">{pct_clamped:.0f}%</span>
  </div>
  <div style="width:100%;height:24px;background:rgba(255,255,255,0.08);border-radius:12px;overflow:hidden;box-shadow:inset 0 2px 4px rgba(0,0,0,0.35);">
    <div style="{bar_style}"></div>
  </div>
</div>"""


def _provider_note_html(title, desc, works, works_list, lost, lost_list,
                        where_url, where_label, where_heading=""):
    """Unified indigo 'works without API key' note for every provider tab
    (todo 9): one color scheme instead of per-provider orange/green."""
    heading_html = f"<strong style=\"color:#ffffff !important;\">{where_heading}</strong>\n  " if where_heading else ""
    return f"""<div style="margin-top:12px;padding:16px;background:rgba(193,95,60,0.08);border-left:4px solid #CC785C;border-radius:10px;font-size:14px;line-height:1.7;color:#F5F4EE !important;">
  <strong style="color:#ffffff !important;font-size:15px;">{title}</strong><br>
  <span style="color:#C2C0B6 !important;">{desc}</span>
  <br><br>
  <strong style="color:#ffffff !important;">{works}</strong><span style="color:#C2C0B6 !important;"> {works_list}</span>
  <br>
  <strong style="color:#ffffff !important;">{lost}</strong><span style="color:#C2C0B6 !important;"> {lost_list}</span>
  <br><br>
  {heading_html}<a href="{where_url}" target="_blank" style="color:#D4A27F !important;font-weight:500;">{where_label}</a>
</div>"""


def _check_for_update(lang="ru"):
    """Check GitHub for newer release. Returns banner HTML or empty string."""
    try:
        import httpx
        import html as html_mod
        resp = httpx.get(
            "https://api.github.com/repos/zhistokoepvpp-ctrl/MovieShort-AI/releases/latest",
            timeout=5,
        )
        if resp.status_code != 200:
            return ""

        data = resp.json()
        tag_name = data.get("tag_name", "")
        if not tag_name.startswith("v"):
            return ""

        remote_ver = tag_name.lstrip("v")
        local_ver = getattr(app_config, "APP_VERSION", "0.0.0")

        remote_parts = [int(x) for x in remote_ver.split(".")]
        local_parts = [int(x) for x in local_ver.split(".")]
        while len(remote_parts) < 3:
            remote_parts.append(0)
        while len(local_parts) < 3:
            local_parts.append(0)

        if remote_parts <= local_parts:
            return ""

        html_url = data.get("html_url", "#")
        body = data.get("body", "")[:300]

        safe_tag = html_mod.escape(tag_name)
        safe_body = html_mod.escape(body[:200].strip())
        safe_url = html_mod.escape(html_url)

        return f"""<div id="update-banner" style="
background:#C15F3C;
color:white;padding:14px 20px;border-radius:10px;
margin-bottom:16px;display:flex;align-items:center;
justify-content:space-between;flex-wrap:wrap;gap:10px;
box-shadow:0 4px 15px rgba(193,95,60,0.4);
">
<div style="display:flex;align-items:center;gap:12px;flex:1;min-width:200px;">
    <span style="font-size:24px;">🎬</span>
    <div>
        <div style="font-weight:bold;font-size:15px;">
            {_t("update_available", lang).format(tag=safe_tag)}
        </div>
        <div style="font-size:13px;opacity:0.9;margin-top:2px;">
            {safe_body}
        </div>
    </div>
</div>
<div style="display:flex;gap:8px;align-items:center;">
    <a href="{safe_url}" target="_blank"
       style="display:inline-block;padding:8px 20px;
              background:white;color:#C15F3C;border-radius:6px;
               text-decoration:none;font-weight:600;font-size:14px;">
        👀 {_t("watch_release", lang)}
    </a>
    <button onclick="this.parentElement.parentElement.style.display='none'"
            style="background:transparent;color:white;border:1px solid rgba(255,255,255,0.5);
                   border-radius:6px;padding:8px 12px;cursor:pointer;font-size:16px;">
        ✕
    </button>
</div>
</div>"""
    except Exception:
        return ""


# --- Gradio Temp cleanup (T14) ---
def _is_gradio_temp_path(p: str) -> bool:
    """True if path is inside system Temp/gradio."""
    try:
        import tempfile
        gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio").lower()
        pl = p.lower()
        if gradio_tmp in pl:
            return True
        # separator-aware fallback: must have /gradio/ segment
        if os.path.sep + "gradio" + os.path.sep in pl:
            return True
        if pl.endswith(os.path.sep + "gradio"):
            return True
        # also handle Temp+gradio combo safety
        if (os.path.sep + "temp" + os.path.sep in pl) and (os.path.sep + "gradio" + os.path.sep in pl):
            return True
        return False
    except Exception:
        return "gradio" in p.lower()


def _try_remove_gradio_temp(p: str) -> None:
    """Remove Gradio temp file if it matches Temp/gradio pattern."""
    try:
        if p and _is_gradio_temp_path(p) and os.path.isfile(p):
            import os as _os2  # noqa: F811
            _os2.unlink(p)
            print(f"  \U0001f9f9 Gradio temp {os.path.basename(p)} удалён")
    except Exception:
        pass


def cleanup_gradio_temp(max_age_seconds: int = 86400) -> int:
    """Стартап-чистка Temp/gradio старше max_age_seconds. Returns removed count."""
    try:
        import tempfile
        import time
        import shutil

        gradio_tmp = os.path.join(tempfile.gettempdir(), "gradio")
        if not os.path.isdir(gradio_tmp):
            return 0
        now = time.time()
        removed = 0
        for f in os.listdir(gradio_tmp):
            fp = os.path.join(gradio_tmp, f)
            try:
                if now - os.path.getmtime(fp) > max_age_seconds:
                    if os.path.isdir(fp):
                        shutil.rmtree(fp)
                    else:
                        os.unlink(fp)
                    removed += 1
            except Exception:
                pass
        if removed:
            print(f"\U0001f9f9 Gradio temp: удалено {removed} старых файлов")
        return removed
    except Exception:
        return 0


def create_app() -> gr.Blocks:
    # стартап-чистка Gradio temp
    try:
        cleanup_gradio_temp()
    except Exception:
        pass
    # Load persisted settings
    cfg = user_config.load()
    # Actual cost per minute from billing data: 31 руб / 246 мин = 0.13 руб/мин
    cfg["cost_per_minute"] = 0.13
    ui_lang = cfg.get("ui_language", "ru")

    # Sync API key to runtime config
    import config as cfg_module
    if cfg.get("api_key"):
        cfg_module.GEMINI_API_KEY = cfg["api_key"]

    with gr.Blocks(
        title="MovieShort AI",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.orange,
            neutral_hue=gr.themes.colors.stone,
            font=["Segoe UI", "system-ui", "sans-serif"],
        )        .set(
            # Components blend into their parent panel: no per-block box,
            # no per-checkbox/radio pill. Panels are styled in _GLASS_CSS.
            block_background_fill_dark="transparent",
            block_border_color_dark="transparent",
            input_background_fill_dark="transparent",
            checkbox_label_background_fill_dark="transparent",
            checkbox_label_border_color_dark="transparent",
        ),
        css=_GLASS_CSS,
        js=FORCE_DARK_JS,
    ) as app:
        # Header card — leaf surface (no dropdown inside) → carries real frost
        gr.HTML(
            value=f"""<div>
  <h1>🎬 MovieShort AI</h1>
  <p>{_t("app_tagline", ui_lang)}</p>
</div>""",
            elem_id="header-card",
        )

        # Donation buttons — always visible above tabs
        with gr.Row():
            donate_text = _t("support_author", ui_lang)
            gr.HTML(
                value=f"""<div style="text-align:center;margin:-8px 0 6px 0;font-size:14px;">
  <span style="color:#9C988B;">{donate_text}</span>
   <a href="https://donatex.gg/donate/nzeronfourme" target="_blank"
     style="display:inline-block;padding:4px 14px;margin:0 6px;
            background:#C15F3C;color:white;border-radius:6px;
            text-decoration:none;font-weight:500;font-size:13px;">
    💸 DonateX
  </a>
  <a href="https://boosty.to/nzeronfourme/donate" target="_blank"
     style="display:inline-block;padding:4px 14px;margin:0 6px;
            background:#C15F3C;color:white;border-radius:6px;
            text-decoration:none;font-weight:500;font-size:13px;">
    🎗 Boosty
  </a>
</div>"""
            )

        # Update notification banner (checked synchronously at startup)
        _banner_html = _check_for_update(ui_lang)
        if _banner_html:
            gr.HTML(value=_banner_html)

        with gr.Tabs():
            # ── Tab 1: Manual ──────────────────────────────────
            with gr.Tab(_t("tab_manual", ui_lang)):
                video_file = gr.File(
                    label=_t("file_label", ui_lang) + _t("file_label_suffix", ui_lang),
                    file_types=[".mp4", ".avi", ".mkv", ".mov"],
                )
                # Progressive timecode ranges (todo 22): ONE field per interval
                # "HH:MM:SS - HH:MM:SS"; field N+1 appears once field N carries
                # all 12 digits; client JS masks digits and autofocuses the
                # newly revealed field.
                tc_boxes = []
                for _i in range(12):
                    tc_boxes.append(gr.Textbox(
                        label=_t("tc_range", ui_lang),
                        placeholder="00:01:30 - 00:02:30",
                        elem_classes="tc-range",
                        visible=(_i == 0),
                    ))
                gr.Markdown(f'<span style="font-size:12px;color:#9C988B">{_t("tc_hint", ui_lang)}</span>')

                def _tc_reveal(val):
                    """Show field N+1 once field N holds all 12 digits."""
                    if len(re.sub(r"\D", "", str(val or ""))) >= 12:
                        return gr.update(visible=True)
                    return gr.update()

                for _i in range(len(tc_boxes) - 1):
                    tc_boxes[_i].change(
                        fn=_tc_reveal, inputs=tc_boxes[_i],
                        outputs=tc_boxes[_i + 1])
                with gr.Group():
                    gr.Markdown(f"### {_t('processing_opts', ui_lang)}")
                    m_subs = gr.Checkbox(value=cfg.get("subtitles", True),
                        label=_t("subs_label", ui_lang),
                        info=_t("subs_info", ui_lang))
                    m_face = gr.Checkbox(value=cfg.get("face_tracking", True),
                        label=_t("face_label", ui_lang),
                        info=_t("face_info", ui_lang))
                    m_banner = gr.Checkbox(value=True,
                        label=_t("banner_label", ui_lang),
                        info=_t("banner_info", ui_lang))
                    with gr.Row():
                        m_banner_top = gr.Slider(0, 500, value=cfg.get("banner_top", 300),
                            label=_t("banner_top", ui_lang))
                        m_banner_bottom = gr.Slider(0, 500, value=cfg.get("banner_bottom", 300),
                            label=_t("banner_bottom", ui_lang))
                    m_blur = gr.Checkbox(value=cfg.get("blur_background", True),
                        label=_t("blur_label", ui_lang),
                        info=_t("blur_info", ui_lang))
                    m_anti = gr.Checkbox(value=cfg.get("anti_copyright", True),
                        label=_t("anti_label", ui_lang),
                        info=_t("anti_info", ui_lang))
                run_btn = gr.Button(_t("process", ui_lang), variant="primary")
                manual_progress = gr.HTML(
                    value=_make_progress_html(0, _t("wait_start", ui_lang)),
                    elem_id="manual-progress",
                )
                manual_log = gr.Textbox(
                    label=_t("console", ui_lang), lines=10, max_lines=20,
                    interactive=False, elem_id="console-log",
                    value=_t("wait_start", ui_lang) + "\n"
                )

                def on_process(file, subs, face, banner, bt, bb, blur, anti,
                               sub_font_name, sub_size, sub_outline, sub_color_name,
                               sub_bold, sub_italic, sub_shadow, sub_position,
                               *tc_values, _lang=ui_lang):
                    if file is None:
                        yield (_t("error_no_file", _lang),
                               _make_progress_html(0, _t("error_no_file", _lang)))
                        return
                    # Assemble "start - end" lines from filled range-fields
                    # (the mask already yields that shape); parse_timestamps
                    # stays the final validator (todo 22).
                    tc_lines = []
                    for val in tc_values:
                        masked = _mask_timecode_range(val)
                        if " - " in masked:
                            tc_lines.append(masked)
                    pairs = parse_timestamps("\n".join(tc_lines))
                    if not pairs:
                        yield (_t("error_no_ts", _lang),
                               _make_progress_html(0, _t("error_no_ts", _lang)))
                        return

                    video_path = file.name if hasattr(file, 'name') else str(file)
                    # --- subtitle style from Editor (R7b-7): must flow to pipeline ---
                    _cmap_ru = {"Белый": "&H00FFFFFF", "Жёлтый": "&H0000FFFF", "Чёрный": "&H00000000", "Красный": "&H000000FF", "Голубой": "&H00FFFF00", "Зелёный": "&H0000FF00"}
                    _cmap_en = {"White": "&H00FFFFFF", "Yellow": "&H0000FFFF", "Black": "&H00000000", "Red": "&H000000FF", "Cyan": "&H00FFFF00", "Green": "&H0000FF00"}
                    _cmap = _cmap_en if _lang == "en" else _cmap_ru
                    _cval = _cmap.get(sub_color_name, "&H00FFFFFF")
                    try:
                        _font_family = ensure_font(sub_font_name, FONTS_DIR)
                    except Exception:
                        _font_family = sub_font_name or "Arial"
                    _font_style = {"font": _font_family, "size": int(sub_size) if sub_size is not None else 13, "color": _cval, "outline": int(sub_outline) if sub_outline is not None else 1, "bold": bool(sub_bold), "italic": bool(sub_italic), "shadow": bool(sub_shadow), "position_y": int(sub_position) if sub_position is not None else 400}
                    _options = {
                        "subtitles": subs,
                        "face_tracking": face,
                        "anti_copyright": anti,
                        "blur_background": blur,
                        "banner_top": bt,
                        "banner_bottom": bb,
                        "subtitle_font_name": sub_font_name,
                        "subtitle_font": _font_family,
                        "subtitle_size": int(sub_size) if sub_size is not None else 13,
                        "subtitle_outline": int(sub_outline) if sub_outline is not None else 1,
                        "subtitle_color": _cval,
                        "subtitle_bold": bool(sub_bold),
                        "subtitle_italic": bool(sub_italic),
                        "subtitle_shadow": bool(sub_shadow),
                        "subtitle_position_y": int(sub_position) if sub_position is not None else 400,
                        "font_style": _font_style,
                    }

                    capture = LogCapture()
                    capture.start_capture()

                    print(f"Файл: {os.path.basename(video_path)}")
                    print(f"Таймкодов: {len(pairs)}")
                    print(f"Опции: subs={subs}, face={face}, blur={blur}, anti={anti}")

                    results_container = []
                    error_container = []

                    def worker():
                        try:
                            results = process_multiple(video_path, pairs, _options)
                            results_container.append(results)
                        except Exception as e:
                            error_container.append(str(e))
                            import traceback
                            error_container.append(traceback.format_exc())

                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()

                    all_lines = []
                    start_ts = time_module.time()
                    last_pct = -1
                    while thread.is_alive():
                        new_lines = capture.get_new_lines()
                        has_new = bool(new_lines)
                        if new_lines:
                            all_lines.extend(new_lines)
                        elapsed = time_module.time() - start_ts
                        pct = min(95, int(elapsed / 120 * 100))
                        if has_new or abs(pct - last_pct) >= 1:
                            label = _t("processing_elapsed", _lang).format(time=_fmt_duration(elapsed))
                            yield ("\n".join(all_lines[-40:]),
                                   _make_progress_html(pct, label))
                            last_pct = pct
                        time_module.sleep(0.6)

                    thread.join(timeout=2)
                    new_lines = capture.get_new_lines()
                    if new_lines:
                        all_lines.extend(new_lines)

                    # T14: автоочистка Gradio temp после обработки
                    try:
                        if file and hasattr(file, 'name'):
                            p = str(file.name)
                            if os.path.sep + "Temp" + os.path.sep in p or "gradio" in p.lower():
                                import os as _os2
                                _os2.unlink(p)
                                print(f"  \U0001f9f9 Gradio temp {os.path.basename(p)} удалён")
                    except Exception:
                        pass
                    # fallback via helper (covers edge where p already in video_path)
                    try:
                        _try_remove_gradio_temp(video_path)
                    except Exception:
                        pass

                    if error_container:
                        print(f"\nОШИБКА: {error_container[0]}")
                        label = _t("progress_error", _lang)
                    elif results_container:
                        results = results_container[0]
                        successes = [r for r in results if r is not None]
                        print(f"\nГотово: {len(successes)}/{len(results)} клипов")
                        for r in successes:
                            print(f"  + {os.path.basename(r)}")
                        for i, r in enumerate(results):
                            if r is None:
                                print(f"  - Клип {i+1}: ошибка обработки")
                        label = _t("progress_done", _lang).format(
                            done=len(successes), total=len(results))
                        elapsed_total = time_module.time() - start_ts
                        print(f"\n⏱  Общее время: {_fmt_duration(elapsed_total)}")
                    else:
                        print("\nОшибка: нет результата")
                        label = _t("progress_error", _lang)

                    capture.stop_capture()
                    yield ("\n".join(capture.get_all()[-40:]),
                           _make_progress_html(100, label))

                # R7b-7: run_btn binding deferred after subtitle editor (needs sub_* inputs) — see bottom of subtitle editor tab

            # ── Tab 2: Automatic ───────────────────────────────
            with gr.Tab(_t("tab_auto", ui_lang)):
                queue_state = gr.State([])

                auto_file = gr.File(
                    label=_t("file_label", ui_lang) + _t("file_label_suffix", ui_lang),
                    file_count="single",
                )
                with gr.Row():
                    movie_title_box = gr.Textbox(
                        label=_t("movie_title", ui_lang),
                        placeholder=_t("movie_placeholder", ui_lang),
                        scale=3,
                    )
                    add_queue_btn = gr.Button("➕ " + _t("add_to_queue", ui_lang),
                        variant="secondary", scale=1, elem_id="add-queue-btn")
                queue_display = gr.HTML(value='<div style="color:#9C988B;padding:8px"><i>' + _t("queue_empty", ui_lang) + '</i></div>')

                def _render_queue(q):
                    if not q:
                        return '<div style="color:#9C988B;padding:8px"><i>' + _t("queue_empty", ui_lang) + '</i></div>'
                    items = ""
                    for i, item in enumerate(q):
                        title_part = f" — {item['title']}" if item.get("title") else ""
                        items += f'<div style="padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.08);display:flex;align-items:center">'
                        items += f'<span style="margin-right:8px;font-weight:bold;color:#D4A27F">{i+1}.</span>'
                        items += f'<span>📁 {item["name"]}{title_part}</span></div>'
                    n = len(q)
                    label = _t("queue_header", ui_lang).format(n=n)
                    return f'<div style="border:1px solid rgba(255,255,255,0.12);border-radius:10px;max-height:200px;overflow-y:auto;background:rgba(26,24,20,0.5)"><div style="padding:6px 10px;background:rgba(255,255,255,0.06);font-weight:bold;border-bottom:1px solid rgba(255,255,255,0.12)">{label}</div>{items}</div>'

                def _add_to_queue(q, file, title):
                    if file is None:
                        return q, _render_queue(q), None, title
                    import os
                    fpath = file.name if hasattr(file, "name") else str(file)
                    new_item = {"path": fpath, "title": title or "", "name": os.path.basename(fpath)}
                    new_q = list(q or []) + [new_item]
                    return new_q, _render_queue(new_q), None, ""

                add_queue_btn.click(
                    fn=_add_to_queue,
                    inputs=[queue_state, auto_file, movie_title_box],
                    outputs=[queue_state, queue_display, auto_file, movie_title_box],
                )
                llm_provider = gr.Radio(
                    choices=["Gemini", "Yandex", "OpenRouter", "OpenCode Zen"],
                    value={"gemini": "Gemini", "yandex": "Yandex", "openrouter": "OpenRouter", "opencode_zen": "OpenCode Zen"}.get(
                        cfg.get("llm_provider", "gemini"), "Gemini"
                    ),
                    label=_t("llm_provider", ui_lang),
                    info=_t("llm_provider_info", ui_lang),
                )
                analysis_mode = gr.Radio(
                    choices=[_t("analysis_mode_std", ui_lang, "Стандартный"),
                             _t("analysis_mode_ctx", ui_lang, "Контекстный (ИИ)")],
                    value=_t("analysis_mode_ctx", ui_lang, "Контекстный (ИИ)") if cfg.get("analysis_mode", "context") == "context" else _t("analysis_mode_std", ui_lang, "Стандартный"),
                    label=_t("analysis_mode", ui_lang),
                    info=_t("analysis_mode_info", ui_lang),
                )
                film_lang = gr.Radio(
                    choices=[_t("lang_russian", ui_lang), _t("lang_english", ui_lang)],
                    value=_t("lang_english", ui_lang) if cfg.get("film_language", "ru") == "en" else _t("lang_russian", ui_lang),
                    label=_t("film_language", ui_lang),
                    info=_t("film_language_info", ui_lang),
                )
                with gr.Row():
                    min_dur = gr.Slider(
                        minimum=15, maximum=60,
                        value=cfg.get("min_duration", 15),
                        label=_t("min_len", ui_lang)
                    )
                    max_dur2 = gr.Slider(
                        minimum=30, maximum=120,
                        value=cfg.get("max_duration", 60),
                        label=_t("max_len", ui_lang)
                    )
                with gr.Row():
                    num_clips = gr.Slider(
                        minimum=5, maximum=20, step=1,
                        value=cfg.get("num_clips", 10),
                        label=_t("num_clips", ui_lang)
                    )
                    score_thresh = gr.Slider(
                        minimum=1, maximum=10, step=0.5,
                        value=cfg.get("score_threshold", 7.0),
                        label=_t("score_threshold", ui_lang),
                        info=_t("score_threshold_info", ui_lang),
                    )
                with gr.Group():
                    gr.Markdown(f"### {_t('processing_opts', ui_lang)}")
                    a_subs = gr.Checkbox(value=cfg.get("subtitles", True),
                        label=_t("subs_label", ui_lang),
                        info=_t("subs_info", ui_lang))
                    a_face = gr.Checkbox(value=cfg.get("face_tracking", True),
                        label=_t("face_label", ui_lang),
                        info=_t("face_info", ui_lang))
                    a_banner = gr.Checkbox(value=True,
                        label=_t("banner_label", ui_lang),
                        info=_t("banner_info", ui_lang))
                    with gr.Row():
                        a_banner_top = gr.Slider(0, 500, value=cfg.get("banner_top", 300),
                            label=_t("banner_top", ui_lang))
                        a_banner_bottom = gr.Slider(0, 500, value=cfg.get("banner_bottom", 300),
                            label=_t("banner_bottom", ui_lang))
                    a_blur = gr.Checkbox(value=cfg.get("blur_background", True),
                        label=_t("blur_label", ui_lang),
                        info=_t("blur_info", ui_lang))
                    a_anti = gr.Checkbox(value=cfg.get("anti_copyright", True),
                        label=_t("anti_label", ui_lang),
                        info=_t("anti_info", ui_lang))
                auto_progress = gr.HTML(
                    value=_make_progress_html(0, _t("wait_start", ui_lang)),
                    elem_id="auto-progress",
                )
                auto_btn = gr.Button(_t("auto_process", ui_lang), variant="primary")
                auto_log = gr.Textbox(
                    label=_t("console", ui_lang), lines=12, max_lines=20,
                    interactive=False, elem_id="console-log",
                    value=_t("wait_start", ui_lang) + "\n"
                )

                def on_auto_process(queue, min_d, max_d,
                                    n_clips, s_thresh,
                                    subs, face, banner, bt, bb, blur, anti,
                                    provider, mode, film_lang_val,
                                    sub_font_name, sub_size, sub_outline, sub_color_name,
                                    sub_bold, sub_italic, sub_shadow, sub_position,
                                    _lang=ui_lang):
                    if not queue:
                        yield (_t("error_no_file", _lang),
                               _make_progress_html(0, _t("error_no_file", _lang)))
                        return

                    # Save current settings as defaults
                    cfg_save = user_config.load()
                    cfg_save["min_duration"] = min_d
                    cfg_save["max_duration"] = max_d
                    cfg_save["subtitles"] = subs
                    cfg_save["face_tracking"] = face
                    cfg_save["banner_top"] = bt
                    cfg_save["banner_bottom"] = bb
                    cfg_save["blur_background"] = blur
                    cfg_save["anti_copyright"] = anti
                    cfg_save["num_clips"] = n_clips
                    cfg_save["score_threshold"] = s_thresh
                    cfg_save["llm_provider"] = {"Gemini": "gemini", "Yandex": "yandex", "OpenRouter": "openrouter", "OpenCode Zen": "opencode_zen"}.get(provider, "gemini")
                    is_context = mode and ("Контекстный" in mode or "Context" in mode)
                    cfg_save["analysis_mode"] = "context" if is_context else "standard"
                    cfg_save["film_language"] = "en" if film_lang_val and "English" in film_lang_val else "ru"
                    cleanup = cfg_save.get("auto_cleanup", True)
                    user_config.save(cfg_save)

                    # Get API key from runtime config
                    import config as cfg_runtime
                    llm_provider_val = {"Gemini": "gemini", "Yandex": "yandex", "OpenRouter": "openrouter", "OpenCode Zen": "opencode_zen"}.get(provider, "gemini")
                    analysis_mode_val = "context" if is_context else "standard"
                    if llm_provider_val == "yandex":
                        api_key = cfg_runtime.YANDEX_API_KEY
                    elif llm_provider_val == "openrouter":
                        api_key = getattr(cfg_runtime, 'OPENROUTER_API_KEY', '')
                    elif llm_provider_val == "opencode_zen":
                        api_key = getattr(cfg_runtime, 'OPENCODE_ZEN_API_KEY', '')
                    else:
                        api_key = cfg_runtime.GEMINI_API_KEY
                    film_language = "en" if film_lang_val and "English" in film_lang_val else "ru"

                    # Process each file in queue
                    all_results = []
                    total_files = len(queue)
                    for file_idx, item in enumerate(queue):
                        video_path = item["path"]
                        movie_title = item.get("title", "")

                        # subtitle style (R7b-7) — Editor controls flow to pipeline
                        _cmap_ru2 = {"Белый": "&H00FFFFFF", "Жёлтый": "&H0000FFFF", "Чёрный": "&H00000000", "Красный": "&H000000FF", "Голубой": "&H00FFFF00", "Зелёный": "&H0000FF00"}
                        _cmap_en2 = {"White": "&H00FFFFFF", "Yellow": "&H0000FFFF", "Black": "&H00000000", "Red": "&H000000FF", "Cyan": "&H00FFFF00", "Green": "&H0000FF00"}
                        _cmap2 = _cmap_en2 if _lang == "en" else _cmap_ru2
                        _cval2 = _cmap2.get(sub_color_name, "&H00FFFFFF")
                        try:
                            _font_family2 = ensure_font(sub_font_name, FONTS_DIR)
                        except Exception:
                            _font_family2 = sub_font_name or "Arial"
                        _font_style2 = {"font": _font_family2, "size": int(sub_size) if sub_size is not None else 13, "color": _cval2, "outline": int(sub_outline) if sub_outline is not None else 1, "bold": bool(sub_bold), "italic": bool(sub_italic), "shadow": bool(sub_shadow), "position_y": int(sub_position) if sub_position is not None else 400}
                        settings = {
                            "min_duration": min_d,
                            "max_duration": max_d,
                            "subtitles": subs,
                            "face_tracking": face,
                            "anti_copyright": anti,
                            "blur_background": blur,
                            "banner_top": bt,
                            "banner_bottom": bb,
                            "num_clips": n_clips,
                            "score_threshold": s_thresh,
                            "api_key": api_key,
                            "movie_title": movie_title,
                            "llm_provider": llm_provider_val,
                            "analysis_mode": analysis_mode_val,
                            "film_language": film_language,
                            "auto_cleanup": cleanup,
                            "subtitle_font_name": sub_font_name,
                            "subtitle_font": _font_family2,
                            "subtitle_size": int(sub_size) if sub_size is not None else 13,
                            "subtitle_outline": int(sub_outline) if sub_outline is not None else 1,
                            "subtitle_color": _cval2,
                            "subtitle_bold": bool(sub_bold),
                            "subtitle_italic": bool(sub_italic),
                            "subtitle_shadow": bool(sub_shadow),
                            "subtitle_position_y": int(sub_position) if sub_position is not None else 400,
                            "font_style": _font_style2,
                        }

                        capture = LogCapture()
                        capture.start_capture()

                        file_label = os.path.basename(video_path)
                        print("=" * 60)
                        print(_t("movie_header", _lang).format(i=file_idx+1, total=total_files))
                        print(f"{_t('file_label', _lang)}: {file_label}")
                        print("=" * 60)
                        if movie_title:
                            print(f"{_t('movie_title', _lang)}: {movie_title}")
                        print(f"{_t('llm_provider', _lang)}: {provider}, {_t('analysis_mode', _lang)}: {_t('analysis_mode_ctx' if is_context else 'analysis_mode_std', _lang)}")
                        print(f"{_t('film_language', _lang)}: {film_language}")
                        if not api_key:
                            print(f"⚠️ {_t('no_api_key_warn', _lang)}")
                        print()

                        results_container = []
                        error_container = []

                        def worker():
                            try:
                                results = process_movie(video_path, settings)
                                results_container.append(results)
                            except Exception as e:
                                error_container.append(str(e))
                                import traceback
                                error_container.append(traceback.format_exc())

                        thread = threading.Thread(target=worker, daemon=True)
                        thread.start()

                        all_lines = []
                        start_ts = time_module.time()
                        EST_TOTAL = 2100
                        last_pct = -1

                        while thread.is_alive():
                            new_lines = capture.get_new_lines()
                            has_new = bool(new_lines)
                            if new_lines:
                                all_lines.extend(new_lines)
                            elapsed = time_module.time() - start_ts
                            pct = min(97, int(elapsed / EST_TOTAL * 100))
                            if has_new or abs(pct - last_pct) >= 1:
                                label = _t("processing_file", _lang).format(i=file_idx+1, total=total_files, name=file_label, time=_fmt_duration(elapsed))
                                yield ("\n".join(all_lines[-40:]),
                                       _make_progress_html(pct, label))
                                last_pct = pct
                            time_module.sleep(0.6)

                        thread.join(timeout=2)
                        new_lines = capture.get_new_lines()
                        if new_lines:
                            all_lines.extend(new_lines)

                        # T14: автоочистка Gradio temp для пакетного режима
                        try:
                            p = str(video_path)
                            if os.path.sep + "Temp" + os.path.sep in p or "gradio" in p.lower():
                                import os as _os2
                                _os2.unlink(p)
                                print(f"  \U0001f9f9 Gradio temp {os.path.basename(p)} удалён")
                        except Exception:
                            pass
                        try:
                            _try_remove_gradio_temp(video_path)
                        except Exception:
                            pass

                        if results_container:
                            all_results.extend(results_container[0])

                    # Final summary
                    if error_container:
                        print(f"\n{_t('error_generic', _lang).format(msg=error_container[0])}")
                        label = _t("error_short", _lang)
                    elif all_results:
                        s = [r for r in all_results if r is not None]
                        print(f"\n{_t('done_count', _lang).format(ok=len(s), total=len(all_results))}")
                        for r in s:
                            print(f"  + {os.path.basename(r)}")
                        label = _t("done_count_files", _lang).format(ok=len(s), total=len(all_results), files=total_files)
                    else:
                        print(f"\n{_t('no_scenes', _lang)}")
                        label = _t("no_scenes_short", _lang)

                    yield ("\n".join(capture.get_all()[-40:]),
                           _make_progress_html(100, label))

                # Clear queue after processing; _render_queue is already defined above
                def _clear_queue():
                    return [], _render_queue([])

                # R7b-7: auto_btn binding deferred after subtitle editor — see below

        # ── Settings ───────────────────────────────────────────
        with gr.Accordion(_t("settings", ui_lang), open=False):
            with gr.Tabs():
                with gr.Tab(_t("tab_gemini", ui_lang)):
                    gemini_key_box = gr.Textbox(
                        label=_t("api_key_label", ui_lang).format(provider="Gemini"),
                        type="password",
                        value=cfg.get("api_key", ""),
                    )
                    with gr.Row():
                        save_gemini_btn = gr.Button(_t("save_key", ui_lang))
                        check_gemini_btn = gr.Button(_t("check_key", ui_lang), variant="secondary")
                    gemini_status = gr.HTML(
                        value='<span style="color:#9C988B">' + _t("status_not_checked", ui_lang) + '</span>',
                    )
                    gr.HTML(
                        value=_provider_note_html(
                            _t("no_api_key_title", ui_lang),
                            _t("no_api_key_desc", ui_lang),
                            _t("no_api_key_works", ui_lang),
                            _t("no_api_key_works_list", ui_lang),
                            _t("no_api_key_lost", ui_lang),
                            _t("no_api_key_lost_list", ui_lang),
                            "https://aistudio.google.com/apikey",
                            "Google AI Studio",
                            where_heading=_t("no_api_key_where", ui_lang),
                        ),
                    )

                with gr.Tab(_t("tab_yandex", ui_lang)):
                    with gr.Row():
                        yandex_key_box = gr.Textbox(
                            label=_t("api_key_label", ui_lang).format(provider="Yandex"),
                            type="password",
                            value=cfg.get("yandex_api_key", ""),
                        )
                        yandex_folder_box = gr.Textbox(
                            label=_t("folder_id_label", ui_lang),
                            value=cfg.get("yandex_folder_id", ""),
                        )
                        yandex_model_dropdown = gr.Dropdown(
                            choices=app_config.YANDEX_MODEL_LIST,
                            label=_t("yandex_model_label", ui_lang),
                            value=cfg.get("yandex_model", "yandexgpt-lite"),
                        )
                    with gr.Row():
                        save_yandex_btn = gr.Button(_t("save_key", ui_lang))
                        check_yandex_btn = gr.Button(_t("check_key", ui_lang), variant="secondary")
                    yandex_status = gr.HTML(
                        value='<span style="color:#9C988B">' + _t("status_not_checked", ui_lang) + '</span>',
                    )
                    gr.HTML(
                        value=_provider_note_html(
                            _t("no_api_key_title", ui_lang),
                            _t("no_api_key_desc", ui_lang),
                            _t("no_api_key_works", ui_lang),
                            _t("no_api_key_works_list", ui_lang),
                            _t("no_api_key_lost", ui_lang),
                            _t("no_api_key_lost_list", ui_lang),
                            "https://aistudio.yandex.cloud/platform/",
                            "Yandex AI Studio",
                            where_heading=_t("no_api_key_where", ui_lang),
                        ),
                    )

                with gr.Tab(_t("tab_openrouter", ui_lang)):
                    with gr.Row():
                        openrouter_key_box = gr.Textbox(
                            label=_t("api_key_label", ui_lang).format(provider="OpenRouter"),
                            type="password",
                            value=cfg.get("openrouter_api_key", ""),
                        )
                        openrouter_model_dropdown = gr.Dropdown(
                            choices=app_config.OPENROUTER_MODEL_LIST,
                            label=_t("openrouter_model_label", ui_lang),
                            value=cfg.get("openrouter_model", app_config.OPENROUTER_MODEL),
                        )
                    with gr.Row():
                        save_openrouter_btn = gr.Button(_t("save_key", ui_lang))
                        check_openrouter_btn = gr.Button(_t("check_key", ui_lang), variant="secondary")
                    openrouter_status = gr.HTML(
                        value='<span style="color:#9C988B">' + _t("status_not_checked", ui_lang) + '</span>',
                    )
                    gr.HTML(
                        value=_provider_note_html(
                            _t("no_api_key_title", ui_lang),
                            _t("no_api_key_desc", ui_lang),
                            _t("no_api_key_works", ui_lang),
                            _t("no_api_key_works_list", ui_lang),
                            _t("no_api_key_lost", ui_lang),
                            _t("no_api_key_lost_list", ui_lang),
                            "https://openrouter.ai/keys",
                            "OpenRouter",
                            where_heading=_t("no_api_key_where", ui_lang),
                        ),
                    )

                with gr.Tab(_t("tab_opencode_zen", ui_lang)):
                    with gr.Row():
                        opencode_zen_key_box = gr.Textbox(
                            label=_t("api_key_label", ui_lang).format(provider="OpenCode Zen"),
                            type="password",
                            value=cfg.get("opencode_zen_api_key", ""),
                        )
                        opencode_zen_model_dropdown = gr.Dropdown(
                            choices=app_config.OPENCODE_ZEN_MODEL_LIST,
                            label=_t("opencode_zen_model_label", ui_lang),
                            value=cfg.get("opencode_zen_model", app_config.OPENCODE_ZEN_MODEL),
                        )
                    with gr.Row():
                        save_opencode_zen_btn = gr.Button(_t("save_key", ui_lang))
                        check_opencode_zen_btn = gr.Button(_t("check_key", ui_lang), variant="secondary")
                    opencode_zen_status = gr.HTML(
                        value='<span style="color:#9C988B">' + _t("status_not_checked", ui_lang) + '</span>',
                    )
                    gr.HTML(
                        value=_provider_note_html(
                            _t("no_api_key_title", ui_lang),
                            _t("no_api_key_desc", ui_lang),
                            _t("no_api_key_works", ui_lang),
                            _t("no_api_key_works_list", ui_lang),
                            _t("no_api_key_lost", ui_lang),
                            _t("no_api_key_lost_list", ui_lang),
                            "https://opencode.ai/auth",
                            "OpenCode Zen",
                            where_heading=_t("no_api_key_where", ui_lang),
                        ),
                    )
                # ── Subtitle Editor tab ──
                with gr.Tab(_t("subtitle_editor", ui_lang)):
                    initial_fs = _get_font_style(cfg)
                    sub_font_dd = gr.Dropdown(
                        # (label, value) tuples: non-Cyrillic fonts annotated in the
                        # UI only; handlers/save still receive the clean font name.
                        choices=[(n + _t("font_no_cyr_suffix", ui_lang), n)
                                 if not meta["cyrillic"] else (n, n)
                                 for n, meta in POPULAR_FONTS.items()],
                        value=cfg.get("subtitle_font_name") or "Bebas Neue",
                        label=_t("sub_font_label", ui_lang),
                    )
                    with gr.Row():
                        sub_size = gr.Slider(8, 48, value=initial_fs["size"], step=1,
                            label=_t("font_size", ui_lang))
                        sub_outline = gr.Slider(0, 5, value=initial_fs["outline"], step=1,
                            label=_t("outline", ui_lang))
                    _COLOR_MAP_RU = {
                        "Белый": "&H00FFFFFF",
                        "Жёлтый": "&H0000FFFF",
                        "Чёрный": "&H00000000",
                        "Красный": "&H000000FF",
                        "Голубой": "&H00FFFF00",
                        "Зелёный": "&H0000FF00",
                    }
                    _COLOR_MAP_EN = {
                        _t("color_white", "en"): "&H00FFFFFF",
                        _t("color_yellow", "en"): "&H0000FFFF",
                        _t("color_black", "en"): "&H00000000",
                        _t("color_red", "en"): "&H000000FF",
                        _t("color_cyan", "en"): "&H00FFFF00",
                        _t("color_green", "en"): "&H0000FF00",
                    }
                    _COLOR_MAP = _COLOR_MAP_EN if ui_lang == "en" else _COLOR_MAP_RU
                    _COLOR_TO_NAME = {v: k for k, v in _COLOR_MAP.items()}
                    _initial_color_name = _COLOR_TO_NAME.get(initial_fs["color"], list(_COLOR_MAP.keys())[0])
                    sub_color = gr.Dropdown(
                        choices=list(_COLOR_MAP.keys()),
                        value=_initial_color_name, label=_t("font_color", ui_lang),
                    )
                    with gr.Row():
                        sub_bold = gr.Checkbox(value=initial_fs["bold"], label=_t("bold", ui_lang))
                        sub_italic = gr.Checkbox(value=initial_fs["italic"], label=_t("italic", ui_lang))
                        sub_shadow = gr.Checkbox(value=initial_fs["shadow"], label=_t("shadow", ui_lang))
                    sub_position = gr.Slider(50, 800, value=initial_fs["position_y"], step=10,
                        label=_t("position_y", ui_lang))
                    with gr.Row():
                        save_sub_btn = gr.Button(_t("save", ui_lang), variant="primary")
                        reset_sub_btn = gr.Button(_t("reset_defaults", ui_lang), variant="secondary")
                    sub_status = gr.HTML(value=f'<span style="color:#9C988B">{_t("not_saved", ui_lang)}</span>')

                    # Real VIDEO preview through the production subtitles
                    # filter. Test video auto-detected from output/test_video/
                    # (first mp4/avi/mkv found); folder created at startup so
                    # the user only has to drop a render inside.
                    _test_dir = os.path.join("output", "test_video")
                    os.makedirs(_test_dir, exist_ok=True)
                    _test_videos = (
                        glob.glob(os.path.join(_test_dir, "*.mp4"))
                        + glob.glob(os.path.join(_test_dir, "*.avi"))
                        + glob.glob(os.path.join(_test_dir, "*.mkv"))
                    )
                    _test_video_path = _test_videos[0] if _test_videos else ""
                    if not _test_video_path:
                        gr.HTML(value='<span style="color:#fbbf24">⚠️ '
                                + _t("preview_no_video", ui_lang) + '</span>')
                    preview_frames_btn = gr.Button(_t("preview", ui_lang))
                    # Todo 33: cap the preview video so a tall vertical render
                    # can't stretch the page; elem_classes lands on the
                    # component wrapper (same mechanism as .tc-range).
                    preview_video = gr.Video(value=None, label=_t("preview", ui_lang),
                                             elem_classes="preview-video")

                    _PREVIEW_COLOR_MAP = _COLOR_MAP_EN if ui_lang == "en" else _COLOR_MAP_RU

                    def _run_video_preview(font_name, size, outline,
                                           color_name, bold, italic, shadow, pos,
                                           subs_on, face_on, banner_v, banner_top_v,
                                           banner_bottom_v, blur_on, anti_on):
                        """Render the FULL test video through the production
                        chain with the CURRENT auto-tab options + editor
                        style (todo 14b). Video path is auto-detected
                        at startup (closure).
                        TIMESTAMP INVARIANT: subtitles are generated over
                        [0, full_duration] — no fragment window exists anymore.
                        Banner semantics mirror production on_auto_process:
                        the banner checkbox is accepted but NOT used there —
                        only the banner_top/banner_bottom slider values flow
                        into the pipeline, so preview passes them verbatim.
                        Never raises — every error lands in sub_status."""
                        lang = ui_lang
                        ok_html = '<span style="color:green">{}</span>'
                        err_html = '<span style="color:red">{}</span>'
                        warn_html = '<span style="color:#fbbf24">{}</span>'
                        try:
                            vp = (_test_video_path or "").strip().strip('"')
                            if not vp:
                                yield None, err_html.format(_t("preview_no_video", lang))
                                return
                            if not os.path.isfile(vp):
                                yield None, err_html.format(
                                    _t("preview_bad_path", lang).format(path=vp))
                                return
                            cval = _PREVIEW_COLOR_MAP.get(color_name, "&H00FFFFFF")
                            family = ensure_font(font_name, FONTS_DIR)
                            fs = {"font": family, "size": size, "outline": outline,
                                  "color": cval, "bold": bold, "italic": italic,
                                  "shadow": shadow, "position_y": pos}
                            prev_dir = app_config.TEMP_DIR / "subtitle_preview"
                            prev_dir.mkdir(parents=True, exist_ok=True)
                            duration = max(_probe_duration(vp), 1.0)
                            if duration > 90:
                                # Long render ahead — warn BEFORE starting.
                                yield None, warn_html.format(
                                    _t("preview_long_warning", lang))
                            segments = [{"start": 0.0, "end": duration,
                                         "text": _t("preview_text", lang)}]
                            if not subs_on:
                                sub_path = None  # render_full_preview skips subs
                            else:
                                sub_path = str(prev_dir / "preview.srt")
                                generate_word_group_srt(segments, sub_path)
                            options = {
                                "banner_top": int(banner_top_v),
                                "banner_bottom": int(banner_bottom_v),
                                "blur": bool(blur_on),
                                "anti_copyright": bool(anti_on),
                                "face_tracking": bool(face_on),
                            }
                            out_path = render_full_preview(
                                vp, sub_path, fs, options, prev_dir)
                            yield out_path, ok_html.format(_t("preview_ok", lang))
                        except Exception as e:
                            yield None, err_html.format(
                                _t("preview_fail", lang).format(err=e))

                    preview_frames_btn.click(fn=_run_video_preview,
                        inputs=[sub_font_dd, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position,
                                a_subs, a_face, a_banner, a_banner_top,
                                a_banner_bottom, a_blur, a_anti],
                        outputs=[preview_video, sub_status])

                    _SAVE_COLOR_MAP = _COLOR_MAP_EN if ui_lang == "en" else _COLOR_MAP_RU
                    def _save_sub_settings(font_name, size, outline, color_name, bold, italic, shadow, pos):
                        cval = _SAVE_COLOR_MAP.get(color_name, "&H00FFFFFF")
                        cfg_k = user_config.load()
                        cfg_k["subtitle_font_name"] = font_name
                        try:
                            cfg_k["subtitle_font"] = ensure_font(font_name, FONTS_DIR)
                        except Exception:
                            # Offline / download failed: keep libass passthrough name,
                            # the TTF will be fetched on the next successful save.
                            cfg_k["subtitle_font"] = font_name
                        cfg_k["subtitle_size"] = size
                        cfg_k["subtitle_outline"] = outline
                        cfg_k["subtitle_color"] = cval
                        cfg_k["subtitle_bold"] = bold
                        cfg_k["subtitle_italic"] = italic
                        cfg_k["subtitle_shadow"] = shadow
                        cfg_k["subtitle_position_y"] = pos
                        user_config.save(cfg_k)
                        return f'<span style="color:green">{_t("saved_sub_ok", cfg_k.get("ui_language", "ru"))}</span>'
                    save_sub_btn.click(fn=_save_sub_settings,
                        inputs=[sub_font_dd, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[sub_status])

                    def _reset_sub_defaults(_lang=ui_lang):
                        """Reset subtitle editor to factory defaults."""
                        cfg_k = user_config.load()
                        cfg_k["subtitle_font_name"] = "Bebas Neue"
                        cfg_k["subtitle_font"] = "Bebas Neue"
                        cfg_k["subtitle_size"] = 13
                        cfg_k["subtitle_outline"] = 1
                        cfg_k["subtitle_color"] = "&H00FFFFFF"
                        cfg_k["subtitle_bold"] = True
                        cfg_k["subtitle_italic"] = False
                        cfg_k["subtitle_shadow"] = False
                        cfg_k["subtitle_position_y"] = 400
                        user_config.save(cfg_k)
                        _reset_color_map = _COLOR_MAP_EN if _lang == "en" else _COLOR_MAP_RU
                        default_color = list(_reset_color_map.keys())[0]
                        msg = f'<span style="color:green">{_t("reset_sub_ok", cfg_k.get("ui_language", "ru"))}</span>'
                        return ("Bebas Neue", 13, 1, default_color, True, False, False, 400,
                                msg)
                    reset_sub_btn.click(fn=_reset_sub_defaults,
                        inputs=[],
                        outputs=[sub_font_dd, sub_size, sub_outline, sub_color,
                                 sub_bold, sub_italic, sub_shadow, sub_position,
                                 sub_status])

                    # ── R7b-7 wiring: final clips use Editor font (not default) ──
                    # run_btn (manual) and auto_btn now include 8 subtitle controls
                    run_btn.click(
                        fn=on_process,
                        inputs=[video_file,
                                m_subs, m_face, m_banner, m_banner_top, m_banner_bottom,
                                m_blur, m_anti,
                                sub_font_dd, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position] + tc_boxes,
                        outputs=[manual_log, manual_progress],
                    )
                    auto_btn.click(
                        fn=on_auto_process,
                        inputs=[queue_state, min_dur, max_dur2,
                                num_clips, score_thresh,
                                a_subs, a_face, a_banner, a_banner_top, a_banner_bottom,
                                a_blur, a_anti,
                                llm_provider, analysis_mode, film_lang,
                                sub_font_dd, sub_size, sub_outline, sub_color,
                                sub_bold, sub_italic, sub_shadow, sub_position],
                        outputs=[auto_log, auto_progress],
                    ).then(
                        fn=_clear_queue,
                        inputs=[],
                        outputs=[queue_state, queue_display],
                    )

                # ── General Settings tab ──
                with gr.Tab(_t("general", ui_lang)):
                    auto_cleanup_cb = gr.Checkbox(
                        value=cfg.get("auto_cleanup", True),
                        label=_t("auto_cleanup", ui_lang),
                        info=_t("auto_cleanup_info", ui_lang),
                    )
                    ui_lang_radio = gr.Radio(
                        choices=[_t("lang_russian", ui_lang), _t("lang_english", ui_lang)],
                        value=_t("lang_english", ui_lang) if cfg.get("ui_language", "ru") == "en" else _t("lang_russian", ui_lang),
                        label=_t("ui_language", ui_lang),
                    )
                    save_general_btn = gr.Button(_t("save", ui_lang), variant="primary")
                    general_status = gr.HTML(value=f'<span style="color:#9C988B">{_t("not_saved", ui_lang)}</span>')

                    def _save_general(cleanup, ui_lang_val):
                        cfg_k = user_config.load()
                        cfg_k["auto_cleanup"] = cleanup
                        cfg_k["ui_language"] = "en" if ui_lang_val and "English" in ui_lang_val else "ru"
                        user_config.save(cfg_k)
                        lang_for_msg = cfg_k["ui_language"]
                        return f'<span style="color:green">{_t("restart_for_lang", lang_for_msg)}</span>'
                    save_general_btn.click(fn=_save_general,
                        inputs=[auto_cleanup_cb, ui_lang_radio],
                        outputs=[general_status])

            def _lang_for_keys():
                cfg_k = user_config.load()
                return cfg_k.get("ui_language", "ru")

            def save_gemini_key(key: str):
                cfg_k = user_config.load()
                cfg_k["api_key"] = key
                cfg_k["llm_provider"] = "gemini"
                user_config.save(cfg_k)
                import config as cfg_runtime
                cfg_runtime.GEMINI_API_KEY = key
                cfg_runtime.LLM_PROVIDER = "gemini"
                lk = _lang_for_keys()
                result = check_api_key(key, "gemini")
                if result.get("ok"):
                    return _t("key_saved", lk).format(provider="Google AI")
                return _t("key_saved_check", lk).format(error=result.get('error', _t("status_unknown", lk)))

            def verify_gemini_key(key: str):
                lk = _lang_for_keys()
                result = check_api_key(key, "gemini")
                if result["ok"]:
                    return f'<span style="color:green">{_t("api_ok", lk)}</span>'
                error = result.get("error", _t("status_unknown", lk))
                if any(x in error.lower() for x in ["лимит", "quota", "429", "запрещён", "limit", "forbidden"]):
                    return f'<span style="color:#FF8C00">{_t("key_valid_quota", lk).format(error=error)}</span>'
                return f'<span style="color:red">❌ {error}</span>'

            def save_yandex_key(key: str, folder_id: str, model: str):
                cfg_k = user_config.load()
                cfg_k["yandex_api_key"] = key
                cfg_k["yandex_folder_id"] = folder_id
                cfg_k["yandex_model"] = model
                cfg_k["llm_provider"] = "yandex"
                user_config.save(cfg_k)
                import config as cfg_runtime
                cfg_runtime.YANDEX_API_KEY = key
                cfg_runtime.YANDEX_FOLDER_ID = folder_id
                cfg_runtime.YANDEX_MODEL = model
                cfg_runtime.LLM_PROVIDER = "yandex"
                lk = _lang_for_keys()
                result = check_api_key(key, "yandex")
                if result.get("ok"):
                    return _t("key_saved", lk).format(provider="Yandex AI")
                return _t("key_saved_check", lk).format(error=result.get('error', _t("status_unknown", lk)))

            def verify_yandex_key(key: str):
                lk = _lang_for_keys()
                result = check_api_key(key, "yandex")
                if result["ok"]:
                    return f'<span style="color:green">{_t("api_ok", lk)}</span>'
                error = result.get("error", _t("status_unknown", lk))
                return f'<span style="color:red">❌ {error}</span>'

            def verify_openrouter_key(key: str):
                lk = _lang_for_keys()
                result = check_api_key(key, provider="openrouter")
                if result["ok"]:
                    return f'<span style="color:green">{_t("api_ok", lk)}</span>'
                error = result.get("error", _t("status_unknown", lk))
                return f'<span style="color:red">❌ {error}</span>'

            def save_openrouter_key(key: str, model: str):
                cfg_k = user_config.load()
                cfg_k["openrouter_api_key"] = key
                cfg_k["openrouter_model"] = model
                cfg_k["llm_provider"] = "openrouter"
                user_config.save(cfg_k)
                import config as cfg_runtime
                cfg_runtime.OPENROUTER_API_KEY = key
                cfg_runtime.OPENROUTER_MODEL = model
                cfg_runtime.LLM_PROVIDER = "openrouter"
                lk = _lang_for_keys()
                return _t("key_saved", lk).format(provider="OpenRouter")

            def verify_opencode_zen_key(key: str):
                lk = _lang_for_keys()
                result = check_api_key(key, provider="opencode_zen")
                if result["ok"]:
                    return f'<span style="color:green">{_t("api_ok", lk)}</span>'
                error = result.get("error", _t("status_unknown", lk))
                return f'<span style="color:red">❌ {error}</span>'

            def save_opencode_zen_key(key: str, model: str):
                cfg_k = user_config.load()
                cfg_k["opencode_zen_api_key"] = key
                cfg_k["opencode_zen_model"] = model
                cfg_k["llm_provider"] = "opencode_zen"
                user_config.save(cfg_k)
                import config as cfg_runtime
                cfg_runtime.OPENCODE_ZEN_API_KEY = key
                cfg_runtime.OPENCODE_ZEN_MODEL = model
                cfg_runtime.LLM_PROVIDER = "opencode_zen"
                lk = _lang_for_keys()
                return _t("key_saved", lk).format(provider="OpenCode Zen")


            save_gemini_btn.click(fn=save_gemini_key, inputs=[gemini_key_box], outputs=[])
            check_gemini_btn.click(fn=verify_gemini_key, inputs=[gemini_key_box], outputs=[gemini_status])
            save_yandex_btn.click(fn=save_yandex_key, inputs=[yandex_key_box, yandex_folder_box, yandex_model_dropdown], outputs=[])
            check_yandex_btn.click(fn=verify_yandex_key, inputs=[yandex_key_box], outputs=[yandex_status])
            save_openrouter_btn.click(fn=save_openrouter_key, inputs=[openrouter_key_box, openrouter_model_dropdown], outputs=[])
            check_openrouter_btn.click(fn=verify_openrouter_key, inputs=[openrouter_key_box], outputs=[openrouter_status])
            save_opencode_zen_btn.click(fn=save_opencode_zen_key, inputs=[opencode_zen_key_box, opencode_zen_model_dropdown], outputs=[])
            check_opencode_zen_btn.click(fn=verify_opencode_zen_key, inputs=[opencode_zen_key_box], outputs=[opencode_zen_status])

            # Load keys into runtime config on startup (without verifying)
            if cfg.get("api_key"):
                import config as cfg_runtime
                cfg_runtime.GEMINI_API_KEY = cfg["api_key"]
            if cfg.get("yandex_api_key"):
                import config as cfg_runtime
                cfg_runtime.YANDEX_API_KEY = cfg["yandex_api_key"]
                cfg_runtime.YANDEX_FOLDER_ID = cfg.get("yandex_folder_id", "")
                cfg_runtime.YANDEX_MODEL = cfg.get("yandex_model", "yandexgpt-lite")
            if cfg.get("openrouter_api_key"):
                import config as cfg_runtime
                cfg_runtime.OPENROUTER_API_KEY = cfg["openrouter_api_key"]
                cfg_runtime.OPENROUTER_MODEL = cfg.get(
                    "openrouter_model", getattr(cfg_runtime, "OPENROUTER_MODEL", "")
                )
            if cfg.get("opencode_zen_api_key"):
                import config as cfg_runtime
                cfg_runtime.OPENCODE_ZEN_API_KEY = cfg["opencode_zen_api_key"]
                cfg_runtime.OPENCODE_ZEN_MODEL = cfg.get(
                    "opencode_zen_model", getattr(cfg_runtime, "OPENCODE_ZEN_MODEL", "")
                )

    return app


def _fmt_duration(seconds: float) -> str:
    return fmt_duration(seconds)


if __name__ == "__main__":
    app = create_app()
    app.launch(server_port=7860)
