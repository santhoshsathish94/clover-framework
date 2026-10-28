"""Clover Hyphen Source Loader & Codec (`# coding: clover-hyphen`).

Allows Python source files and hyphenated file names (such as `clover-object-model.py`,
`generative-code-compiler.py`, `memory-knowledge-graph.py`) to write literal ASCII hyphens `-`
in role-prefixed identifiers:
  - `class-*`      (e.g. `class-task-direction`, `class-conversation-session`, `class-verified-knowledge-graph`)
  - `function-*`   (e.g. `function-encode-input-into-activation-vector`, `function-execute-complete-clover-cycle`)
  - `variable-*`   (e.g. `variable-resident-entries-by-identifier`, `variable-maximum-token-budget`)
  - `flag-*`       (e.g. `flag-has-passed-all-checks`, `flag-is-cache-currently-full`)
  - `constant-*`   (e.g. `constant-maximum-session-context-tokens`)
  - `decorator-*`  (e.g. `decorator-register-component-in-catalog`, `decorator-enforce-direction-boundary`)

How it works safely without touching strings, comments, or subtraction operators:
1. Scans the source text character-by-character, preserving string literals (`'...'`, `"..."`,
   `'''...'''`, `\"\"\"...\"\"\"`) and `# ...` comments verbatim.
2. Outside strings and comments, matches any identifier token starting with one of the 6 role prefixes
   `(class|function|variable|flag|constant|decorator)-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*` and converts its
   internal `-` characters into `_` before handing the source to Python's AST compiler.
3. Also provides `load_hyphenated_module(file_or_module_name)` and registers a `sys.meta_path` finder
   so modules named `clover-object-model.py`, `generative-code-compiler.py`, etc. can be imported
   seamlessly across the repository.
"""

from __future__ import annotations

import codecs
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import re
import sys
import types
from typing import Iterable, Optional, Sequence, Tuple


ROLE_PREFIX_HYPHEN_PATTERN = re.compile(
    r"\b(?:class|function|variable|flag|constant|decorator)-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b"
)


def translate_clover_hyphen_source(source_text: str) -> str:
    """Translates `class-*`, `function-*`, `variable-*`, `flag-*`, `constant-*`, and `decorator-*`
    identifiers containing literal ASCII `-` into valid Python identifiers while leaving comments
    and arithmetic `-` subtraction operators untouched."""
    output_lines: list[str] = []
    for raw_line in source_text.splitlines(keepends=True):
        in_single = False
        in_double = False
        comment_index = -1
        idx = 0
        while idx < len(raw_line):
            ch = raw_line[idx]
            if ch == "\\" and (in_single or in_double):
                idx += 2
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "#" and not in_single and not in_double:
                comment_index = idx
                break
            idx += 1
        if comment_index >= 0:
            code_part = raw_line[:comment_index]
            comment_part = raw_line[comment_index:]
        else:
            code_part = raw_line
            comment_part = ""
        translated_code = ROLE_PREFIX_HYPHEN_PATTERN.sub(
            lambda match_obj: match_obj.group(0).replace("-", "_"),
            code_part,
        )
        output_lines.append(translated_code + comment_part)
    return "".join(output_lines)


class CloverHyphenIncrementalDecoder(codecs.BufferedIncrementalDecoder):
    def _buffer_decode(self, input_bytes: bytes, errors: str, final: bool) -> Tuple[str, int]:
        if not final:
            return "", 0
        raw_text = input_bytes.decode("utf-8", errors=errors)
        return translate_clover_hyphen_source(raw_text), len(input_bytes)


def _decode_clover_hyphen_bytes(input_bytes: memoryview | bytes, errors: str = "strict") -> Tuple[str, int]:
    raw_bytes = bytes(input_bytes)
    raw_text = raw_bytes.decode("utf-8", errors=errors)
    return translate_clover_hyphen_source(raw_text), len(raw_bytes)


def _encode_clover_hyphen_text(input_text: str, errors: str = "strict") -> Tuple[bytes, int]:
    return input_text.encode("utf-8", errors=errors), len(input_text)


def _search_clover_hyphen_codec(encoding_name: str) -> Optional[codecs.CodecInfo]:
    normalized = encoding_name.lower().replace("_", "-")
    if normalized in ("clover-hyphen", "clover-english"):
        return codecs.CodecInfo(
            name="clover-hyphen",
            encode=_encode_clover_hyphen_text,
            decode=_decode_clover_hyphen_bytes,
            incrementaldecoder=CloverHyphenIncrementalDecoder,
        )
    return None


class CloverHyphenSourceFileLoader(importlib.machinery.SourceFileLoader):
    """SourceFileLoader that translates `class-*`, `function-*`, `variable-*`, `flag-*`,
    `constant-*`, and `decorator-*` hyphenated identifiers before compiling to bytecode."""

    def exec_module(self, module: types.ModuleType) -> None:
        if not hasattr(module, "__file__") or module.__file__ is None:
            module.__file__ = self.path
        super().exec_module(module)

    def get_code(self, fullname: str) -> Optional[types.CodeType]:
        source_path = self.get_filename(fullname)
        raw_bytes = self.get_data(source_path)
        return self.source_to_code(raw_bytes, source_path)

    def source_to_code(self, data: bytes | str, path: str = "<string>", *, _optimize: int = -1) -> types.CodeType:
        if isinstance(data, (bytes, bytearray, memoryview)):
            raw_text = bytes(data).decode("utf-8")
        else:
            raw_text = str(data)
        raw_text = re.sub(r"^#.*coding[:=]\s*clover-hyphen.*$", "# translated by clover-hyphen", raw_text, flags=re.MULTILINE)
        translated_text = translate_clover_hyphen_source(raw_text)
        return compile(translated_text, path, "exec", dont_inherit=True, optimize=_optimize)


class CloverHyphenMetaPathFinder(importlib.abc.MetaPathFinder):
    """Allows importing hyphenated files (e.g., `clover-object-model.py` as `clover_object_model`
    or `generative-code-compiler.py` as `generative_code_compiler`) from `sys.path`."""

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]] = None,
        target: Optional[types.ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        module_basename = fullname.rpartition(".")[2]
        hyphenated_filename = module_basename.replace("_", "-") + ".py"
        underscore_filename = module_basename + ".py"
        search_directories: Iterable[str] = path if path is not None else sys.path
        for directory_entry in search_directories:
            base_dir = Path(directory_entry or ".")
            for candidate_name in (hyphenated_filename, underscore_filename):
                candidate_path = base_dir / candidate_name
                if candidate_path.is_file():
                    try:
                        header_Sample = candidate_path.read_bytes()[:512].decode("utf-8", errors="ignore")
                    except Exception:
                        header_Sample = ""
                    if candidate_name == hyphenated_filename or "clover-hyphen" in header_Sample or "class-" in header_Sample:
                        loader = CloverHyphenSourceFileLoader(fullname, str(candidate_path))
                        return importlib.util.spec_from_loader(fullname, loader, origin=str(candidate_path))
        return None


def register_clover_hyphen_runtime() -> None:
    """Registers the `clover-hyphen` source codec and `CloverHyphenMetaPathFinder` idempotently."""
    codecs.register(_search_clover_hyphen_codec)
    if not any(isinstance(finder, CloverHyphenMetaPathFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, CloverHyphenMetaPathFinder())


def load_hyphenated_module(file_path_or_name: str | Path, module_alias: Optional[str] = None) -> types.ModuleType:
    """Loads any `.py` file containing literal `-` role-prefixed identifiers and registers it in `sys.modules`."""
    register_clover_hyphen_runtime()
    resolved_path = Path(file_path_or_name).resolve()
    canonical_module_name = module_alias or resolved_path.stem.replace("-", "_")
    loader = CloverHyphenSourceFileLoader(canonical_module_name, str(resolved_path))
    spec = importlib.util.spec_from_loader(canonical_module_name, loader, origin=str(resolved_path))
    if spec is None:
        raise ImportError(f"Could not create ModuleSpec for '{resolved_path}'.")
    module_obj = importlib.util.module_from_spec(spec)
    module_obj.__file__ = str(resolved_path)
    sys.modules[canonical_module_name] = module_obj
    sys.modules[resolved_path.stem] = module_obj
    loader.exec_module(module_obj)
    return module_obj


register_clover_hyphen_runtime()
