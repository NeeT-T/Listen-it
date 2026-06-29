import os
from datetime import datetime
from pathlib import Path


class MarkdownOutput:
    def __init__(self, mode: str, model: str, lang: str, outputs_dir: str = "outputs"):
        Path(outputs_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = Path(outputs_dir) / f"{timestamp}.md"
        self._file = self.path.open("w", encoding="utf-8")
        display_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"# Transcrição — {display_ts}\n\n")
        self._file.write(f"**Modo:** {mode}  \n")
        self._file.write(f"**Modelo:** {model}  \n")
        self._file.write(f"**Idioma:** {lang}  \n\n")
        self._file.write("---\n\n")
        self._file.flush()

    def append(self, timestamp: str, text: str) -> None:
        self._file.write(f"[{timestamp}] {text.strip()}\n\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
