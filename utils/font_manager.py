"""Font management for subtitle rendering.
Downloads TTF files from Google Fonts to a local fonts directory
when not present. No system installation needed — ffmpeg/libass
uses fontsdir option to find them."""

import os
import urllib.request
from pathlib import Path

FONTS_DIR = os.path.join("output", "fonts")

POPULAR_FONTS = {
    "Oswald": {
        "family": "Oswald",
        # Variable font kept: empirically verified — libass matches family
        # nameID1="Oswald" and renders Cyrillic from Oswald[wght].ttf.
        "url": "https://github.com/google/fonts/raw/main/ofl/oswald/Oswald%5Bwght%5D.ttf",
        "file": "Oswald.ttf",
        "cyrillic": True,
    },
    "Bebas Neue": {
        # «Bebas Neue Cyrillic» per user choice (fonts-online.ru/fonts/bebas-neue-cyrillic).
        # License on source page: all rights reserved (Ryoichi Tsunekawa,
        # Cyrillic by AA) -> runtime download ONLY, never commit the TTF.
        # Fallback mirror (family there is "Bebas Neue", not "...Cyrillic"):
        #   https://github.com/Scrum/font-bebas-neue/raw/master/fonts/BebasNeue.ttf
        "family": "Bebas Neue Cyrillic",
        "url": "https://fonts-online.ru/sites/default/files/2021-09/bebasneuecyrillic.ttf",
        "file": "BebasNeueCyrillic.ttf",
        "cyrillic": True,
    },
    "Montserrat": {
        "family": "Montserrat ExtraBold",
        # Static instance REQUIRED: variable Montserrat[wght].ttf name-table
        # says family="Montserrat" -> libass lookup for "Montserrat ExtraBold"
        # fails. google/fonts ships no static TTF (404) -> upstream repo.
        "url": "https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf",
        "file": "Montserrat-ExtraBold.ttf",
        "cyrillic": True,
    },
    "Anton": {
        "family": "Anton",
        "url": "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
        "file": "Anton-Regular.ttf",
        "cyrillic": False,
    },
    "Archivo Black": {
        "family": "Archivo Black",
        "url": "https://github.com/google/fonts/raw/main/ofl/archivoblack/ArchivoBlack-Regular.ttf",
        "file": "ArchivoBlack-Regular.ttf",
        "cyrillic": False,
    },
    "Poppins": {
        "family": "Poppins SemiBold",
        "url": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-SemiBold.ttf",
        "file": "Poppins-SemiBold.ttf",
        "cyrillic": False,
    },
    "Impact": {
        "family": "Impact",
        "url": None,  # system font on Windows, no download needed
        "file": None,
        "cyrillic": True,  # Windows system Impact covers Cyrillic
    },
}


def ensure_font(display_name: str, fonts_dir: str = FONTS_DIR) -> str:
    """Ensure font TTF exists locally. Returns libass-compatible family name.

    Downloads from Google Fonts if missing. Creates fonts_dir if needed.
    System fonts (Impact) are returned as-is without download.
    """
    entry = POPULAR_FONTS.get(display_name)
    if entry is None:
        # Unknown font: user may have it installed on the system — pass through.
        return display_name
    family = entry["family"]
    url = entry["url"]
    if url is None:
        return family  # system font (e.g. Impact), nothing to download
    Path(fonts_dir).mkdir(parents=True, exist_ok=True)
    target = os.path.join(fonts_dir, entry["file"])
    if not os.path.exists(target):
        urllib.request.urlretrieve(url, target)
    return family


def get_available_fonts() -> list:
    """Sorted display names for GUI dropdown."""
    return sorted(POPULAR_FONTS.keys())
