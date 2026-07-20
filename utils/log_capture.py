"""
MovieShort AI — Debug log capture utility.
Redirects print() output to both console and a queue for GUI display.
"""
import sys
import queue


class LogCapture:
    """Captures print() output into a queue for real-time GUI display.

    Handles:
    - Normal print() calls with newlines
    - Partial lines (accumulated in buffer until newline)
    - Carriage-return-based progress updates (\r)
    """

    def __init__(self):
        self._queue = queue.Queue()
        self._lines = []
        self._buffer = ""
        self._original_stdout = None

    def write(self, text: str):
        # Always write through to real stdout
        if self._original_stdout:
            self._original_stdout.write(text)
            self._original_stdout.flush()

        if not text:
            return

        # Accumulate into buffer and split on newlines
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            # Strip carriage returns (used by progress bars like tqdm)
            line = line.strip("\r").strip()
            if line:
                self._queue.put(line)

        # Handle carriage returns: replace current line
        if "\r" in self._buffer:
            parts = self._buffer.split("\r")
            # Take content after last \r as current partial line
            last_part = parts[-1].strip()
            if last_part:
                # Don't queue it (no newline yet), just track it
                pass

    def flush(self):
        if self._original_stdout:
            self._original_stdout.flush()

    def start_capture(self):
        self._original_stdout = sys.stdout
        sys.stdout = self

    def stop_capture(self):
        if self._original_stdout:
            sys.stdout = self._original_stdout
            self._original_stdout = None

    def get_new_lines(self):
        """Get all complete lines since last call."""
        lines = []
        while True:
            try:
                lines.append(self._queue.get_nowait())
            except queue.Empty:
                break
        self._lines.extend(lines)
        return lines

    def get_all(self):
        """Get all captured lines so far."""
        return self._lines
