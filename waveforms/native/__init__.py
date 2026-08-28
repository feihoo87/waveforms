"""Installed sources and public header for the language-neutral native core."""

from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
HEADER = SOURCE_DIR / "wf_native.h"
SOURCE = SOURCE_DIR / "wf_native.c"
FORMAT_SPEC = SOURCE_DIR / "WNF4_FORMAT.md"

__all__ = ["SOURCE_DIR", "HEADER", "SOURCE", "FORMAT_SPEC"]
