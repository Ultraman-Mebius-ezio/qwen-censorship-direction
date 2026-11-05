"""Minimal pcre shim for gptqmodel (project-local, via PYTHONPATH).

gptqmodel uses pcre for ordinary regex work (include-line parsing, comment
stripping, splitting compiler flags). Python's built-in re covers every
pattern it compiles, so this shim avoids the native libpcre build on Windows.
"""
import re


class Flag:
    CASELESS = re.IGNORECASE
    IGNORECASE = re.IGNORECASE
    DOTALL = re.DOTALL
    MULTILINE = re.MULTILINE


Pattern = re.Pattern
compile = re.compile
escape = re.escape
search = re.search
split = re.split
