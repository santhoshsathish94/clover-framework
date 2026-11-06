"""Local-first deterministic tool packs for Clover AI Studio (Phase 1 Reliability & Inspection Pack + Phase 2
Documents, Data & Media Pack).

Phase 1 — Reliability and Inspection Pack:
  1. calculate      : Exact AST/Decimal arithmetic evaluator (no eval, bounded AST/digits/exponents).
  2. convert_units  : Deterministic unit conversion across an explicit unit catalogue (length, mass, time,
                      temperature, data size, volume, area, speed).
  3. search_files   : Bounded file-name glob and literal content search in an authorized folder (post-resolution
                      containment, skips symlinks/reparse points, vault, records, .git, and binary/oversized files).
  4. file_info      : File metadata (size, timestamps, type, line count) and optional streamed SHA-256.
  5. git_inspect    : Fixed read-only Git operations (status, diff, log, show) with argument-injection protection.

Phase 2 — Documents, Data, and Media Pack:
  6. read_document  : Paginated document reading (PDF pages, PPTX slides, XLSX/CSV sheet & row ranges, DOCX/TXT lines).
  7. analyze_table  : Fixed deterministic filter/group/aggregate/sort operations on spreadsheets, CSV, TSV, and JSON.
  8. chart_table    : Fixed deterministic chart generation (bar, line, scatter, pie, hist) saved as a chat PNG artifact.
  9. media_info     : Fixed FFprobe JSON inspection of audio, video, and image files via local_terminal.run.
 10. extract_frame  : Fixed FFmpeg video frame extraction at a specified timestamp into a chat PNG artifact.
"""
from __future__ import annotations

import ast
import csv
import datetime
import decimal
import fnmatch
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

import local_terminal
import paths

MAX_EXPR_CHARS = 400
MAX_AST_NODES = 64
MAX_LITERAL_DIGITS = 64
MAX_EXPONENT = 10_000

SEARCH_MAX_DEPTH = 12
SEARCH_MAX_FILES = 2_000
SEARCH_MAX_FILE_BYTES = 2 * 1024 * 1024
SEARCH_MAX_RESULTS_CAP = 100
SEARCH_DEFAULT_RESULTS = 25
SEARCH_TIMEOUT_S = 10.0
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "vault", "clover-ai-studio"}

HASH_MAX_BYTES = 512 * 1024 * 1024
HASH_TIMEOUT_S = 15.0

TABLE_SCAN_ROWS = 200_000


def _fn(name, description, props=None, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props or {}, "required": list(required)},
        },
    }


# ---------------------------------------------------------------------------
# Path & Authorized Root Helpers
# ---------------------------------------------------------------------------
def is_reparse_or_symlink(path: Path) -> bool:
    """True when `path` is a symlink or Windows reparse point / junction."""
    try:
        if path.is_symlink():
            return True
        st = os.lstat(path)
        if getattr(st, "st_reparse_tag", 0) != 0:
            return True
        if getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            return True
    except OSError:
        return True
    return False


def resolve_authorized_path(raw_path: str | Path | None, base_dir: Path, workdir: Path | None = None, default_to_workdir: bool = False) -> tuple[Path | None, str | None]:
    """Resolves a file or folder path, blocking Clover's vault and (when writing or searching) Clover's internal repo."""
    s = str(raw_path or "").strip()
    if not s:
        if default_to_workdir:
            p = Path(workdir) if workdir else Path.home()
        else:
            return None, "Error: path is required."
    else:
        expanded = os.path.expandvars(os.path.expanduser(s))
        p = Path(expanded)
        if not p.is_absolute():
            p = (Path(workdir) if workdir else Path.home()) / p
    try:
        resolved = p.resolve()
    except OSError as exc:
        return None, f"Error: invalid path {p}: {exc}"
    if paths.read_blocked(base_dir, resolved):
        return None, "Error: Clover's vault holds secrets; it is not read by tools."
    if paths._inside(resolved, paths.repo_dir(base_dir)) or resolved.name.lower() == "clover-ai-studio" or any(p.name.lower() == "clover-ai-studio" for p in resolved.parents):
        return None, "Error: Clover's internal studio records are not accessed by file tools."
    return resolved, None


def resolve_authorized_root(raw_root: str | Path | None, base_dir: Path, workdir: Path | None = None, user_paths: tuple | list = ()) -> tuple[Path | None, str | None]:
    """Resolves a root directory for search or Git inspection, blocking vault, Clover's internal repo, and unauthorized roots."""
    root, err = resolve_authorized_path(raw_root, base_dir, workdir, default_to_workdir=True)
    if err or root is None:
        return None, err
    if paths._inside(root, paths.repo_dir(base_dir)):
        return None, "Error: Clover's internal records repository is not searched by file tools."
    if root.name.lower() == "clover-ai-studio" or any(p.name.lower() == "clover-ai-studio" for p in root.parents):
        return None, "Error: Clover's internal studio directory is not searched by file tools."
    if not root.exists():
        return None, f"Error: the folder {root} does not exist."
    if not root.is_dir():
        return None, f"Error: {root} is a file, not a folder."
    # Section 4.3: Root must be inside the active workspace, an explicitly user-named path, or (when no workspace) home
    allowed_roots = []
    if workdir:
        try:
            allowed_roots.append(Path(workdir).resolve())
        except OSError:
            pass
    else:
        try:
            allowed_roots.append(Path.home().resolve())
        except OSError:
            pass
    for up in (user_paths or ()):
        try:
            up_res = Path(os.path.expandvars(os.path.expanduser(str(up)))).resolve()
            allowed_roots.append(up_res if up_res.is_dir() else up_res.parent)
        except OSError:
            pass
    if allowed_roots and not any(paths._inside(root, ar) for ar in allowed_roots):
        return None, f"Error: {root} is outside the authorized workspace or user-specified folder."
    return root, None


# ---------------------------------------------------------------------------
# 1. calculate — Safe Deterministic AST/Decimal Calculator
# ---------------------------------------------------------------------------
_SAFE_CONSTANTS = {
    "pi": Decimal(str(math.pi)),
    "e": Decimal(str(math.e)),
    "tau": Decimal(str(math.tau)),
}


def _fmt_number(val: Decimal | int | float) -> str:
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError("result is not finite")
        if val.is_integer() and abs(val) < 1e18:
            return str(int(val))
        return f"{val:.12g}"
    d = val.normalize()
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def calculate(expression: str) -> str:
    """Safely evaluates an arithmetic expression without Python eval()."""
    raw = str(expression or "").strip()
    if not raw:
        return "Error: expression is required."
    if len(raw) > MAX_EXPR_CHARS:
        return f"Error: expression exceeds {MAX_EXPR_CHARS} characters."

    norm = (
        raw.replace("\u00d7", "*")
        .replace("\u00f7", "/")
        .replace("\u2212", "-")
        .replace("^", "**")
    )
    norm = re.sub(r"(?<=\d),(?=\d{3}\b)", "", norm)
    norm = re.sub(r"^\s*(?:what\s+is|calculate|compute|evaluate)\s+", "", norm, flags=re.I)
    norm = re.sub(r"[?=]+\s*$", "", norm).strip()

    for m in re.finditer(r"\d+", norm):
        if len(m.group(0)) > MAX_LITERAL_DIGITS:
            return f"Error: numeric literal exceeds {MAX_LITERAL_DIGITS} digits."

    try:
        tree = ast.parse(norm, mode="eval")
    except SyntaxError as exc:
        return f"Error: invalid arithmetic expression ({exc.msg})."

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        return f"Error: expression is too complex ({len(nodes)} operations; max {MAX_AST_NODES})."

    with decimal.localcontext() as ctx:
        ctx.prec = 50

        def _eval_node(node) -> Decimal:
            if isinstance(node, ast.Expression):
                return _eval_node(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    raise ValueError(f"unsupported literal {node.value!r}")
                return Decimal(str(node.value))
            if isinstance(node, ast.Name):
                key = node.id.lower()
                if key not in _SAFE_CONSTANTS:
                    raise ValueError(f"unknown identifier {node.id!r}")
                return _SAFE_CONSTANTS[key]
            if isinstance(node, ast.UnaryOp):
                v = _eval_node(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +v
                if isinstance(node.op, ast.USub):
                    return -v
                raise ValueError("unsupported unary operator")
            if isinstance(node, ast.BinOp):
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    if right == 0:
                        raise ZeroDivisionError("division by zero")
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    if right == 0:
                        raise ZeroDivisionError("division by zero")
                    return Decimal(int(left // right))
                if isinstance(node.op, ast.Mod):
                    if right == 0:
                        raise ZeroDivisionError("modulo by zero")
                    return left % right
                if isinstance(node.op, ast.Pow):
                    if abs(right) > MAX_EXPONENT:
                        raise ValueError(f"exponent {right} exceeds maximum ({MAX_EXPONENT})")
                    if abs(left) > Decimal("1e6") and abs(right) > 64:
                        raise ValueError("power is too large")
                    if right == int(right):
                        return left ** int(right)
                    return Decimal(str(math.pow(float(left), float(right))))
                raise ValueError("unsupported binary operator")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.keywords:
                    raise ValueError("only simple math functions are allowed")
                fn = node.func.id.lower()
                args = [_eval_node(a) for a in node.args]
                if fn == "sqrt" and len(args) == 1:
                    if args[0] < 0:
                        raise ValueError("sqrt of negative number")
                    return args[0].sqrt()
                if fn == "abs" and len(args) == 1:
                    return abs(args[0])
                if fn == "round" and len(args) in (1, 2):
                    nd = int(args[1]) if len(args) == 2 else 0
                    return Decimal(str(round(float(args[0]), nd)))
                if fn == "min" and len(args) >= 1:
                    return min(args)
                if fn == "max" and len(args) >= 1:
                    return max(args)
                if fn == "floor" and len(args) == 1:
                    return Decimal(math.floor(args[0]))
                if fn == "ceil" and len(args) == 1:
                    return Decimal(math.ceil(args[0]))
                if fn in ("sin", "cos", "tan", "log", "log10", "log2") and len(args) == 1:
                    fv = float(args[0])
                    if fn in ("log", "log10", "log2") and fv <= 0:
                        raise ValueError(f"{fn} requires a positive number")
                    res_f = getattr(math, fn)(fv)
                    return Decimal(str(res_f))
                if fn == "factorial" and len(args) == 1:
                    n_int = int(args[0])
                    if args[0] != n_int or n_int < 0 or n_int > 200:
                        raise ValueError("factorial requires an integer in 0..200")
                    return Decimal(math.factorial(n_int))
                if fn == "gcd" and len(args) == 2:
                    return Decimal(math.gcd(int(args[0]), int(args[1])))
                raise ValueError(f"unsupported function {node.func.id!r}")
            raise ValueError(f"unsupported syntax {type(node).__name__}")

        try:
            result = _eval_node(tree)
            return _fmt_number(result)
        except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError) as exc:
            return f"Error: {exc}"


# ---------------------------------------------------------------------------
# 2. convert_units — Deterministic Unit Catalogue Converter
# ---------------------------------------------------------------------------
# Each non-temperature unit maps to (category, factor_to_base_in_Decimal, canonical_symbol)
_UNIT_RAW = {
    # Length (base: meter)
    "nm": ("length", "1e-9", "nm"), "nanometer": ("length", "1e-9", "nm"), "nanometers": ("length", "1e-9", "nm"),
    "um": ("length", "1e-6", "um"), "micrometer": ("length", "1e-6", "um"), "micrometers": ("length", "1e-6", "um"),
    "mm": ("length", "0.001", "mm"), "millimeter": ("length", "0.001", "mm"), "millimeters": ("length", "0.001", "mm"), "millimetre": ("length", "0.001", "mm"), "millimetres": ("length", "0.001", "mm"),
    "cm": ("length", "0.01", "cm"), "centimeter": ("length", "0.01", "cm"), "centimeters": ("length", "0.01", "cm"), "centimetre": ("length", "0.01", "cm"), "centimetres": ("length", "0.01", "cm"),
    "m": ("length", "1", "m"), "meter": ("length", "1", "m"), "meters": ("length", "1", "m"), "metre": ("length", "1", "m"), "metres": ("length", "1", "m"),
    "km": ("length", "1000", "km"), "kilometer": ("length", "1000", "km"), "kilometers": ("length", "1000", "km"), "kilometre": ("length", "1000", "km"), "kilometres": ("length", "1000", "km"),
    "in": ("length", "0.0254", "in"), "inch": ("length", "0.0254", "in"), "inches": ("length", "0.0254", "in"),
    "ft": ("length", "0.3048", "ft"), "foot": ("length", "0.3048", "ft"), "feet": ("length", "0.3048", "ft"),
    "yd": ("length", "0.9144", "yd"), "yard": ("length", "0.9144", "yd"), "yards": ("length", "0.9144", "yd"),
    "mi": ("length", "1609.344", "mi"), "mile": ("length", "1609.344", "mi"), "miles": ("length", "1609.344", "mi"),
    "nmi": ("length", "1852", "nmi"), "nautical_mile": ("length", "1852", "nmi"), "nautical_miles": ("length", "1852", "nmi"),

    # Mass (base: kilogram)
    "mg": ("mass", "1e-6", "mg"), "milligram": ("mass", "1e-6", "mg"), "milligrams": ("mass", "1e-6", "mg"),
    "g": ("mass", "0.001", "g"), "gram": ("mass", "0.001", "g"), "grams": ("mass", "0.001", "g"),
    "kg": ("mass", "1", "kg"), "kilogram": ("mass", "1", "kg"), "kilograms": ("mass", "1", "kg"), "kilo": ("mass", "1", "kg"), "kilos": ("mass", "1", "kg"),
    "t": ("mass", "1000", "t"), "tonne": ("mass", "1000", "t"), "tonnes": ("mass", "1000", "t"), "metric_ton": ("mass", "1000", "t"), "metric_tons": ("mass", "1000", "t"),
    "oz": ("mass", "0.028349523125", "oz"), "ounce": ("mass", "0.028349523125", "oz"), "ounces": ("mass", "0.028349523125", "oz"),
    "lb": ("mass", "0.45359237", "lb"), "lbs": ("mass", "0.45359237", "lb"), "pound": ("mass", "0.45359237", "lb"), "pounds": ("mass", "0.45359237", "lb"),
    "st": ("mass", "6.35029318", "st"), "stone": ("mass", "6.35029318", "st"), "stones": ("mass", "6.35029318", "st"),
    "short_ton": ("mass", "907.18474", "short_ton"), "us_ton": ("mass", "907.18474", "short_ton"), "us_tons": ("mass", "907.18474", "short_ton"),

    # Time (base: second)
    "ns": ("time", "1e-9", "ns"), "nanosecond": ("time", "1e-9", "ns"), "nanoseconds": ("time", "1e-9", "ns"),
    "us": ("time", "1e-6", "us"), "microsecond": ("time", "1e-6", "us"), "microseconds": ("time", "1e-6", "us"),
    "ms": ("time", "0.001", "ms"), "millisecond": ("time", "0.001", "ms"), "milliseconds": ("time", "0.001", "ms"),
    "s": ("time", "1", "s"), "sec": ("time", "1", "s"), "secs": ("time", "1", "s"), "second": ("time", "1", "s"), "seconds": ("time", "1", "s"),
    "min": ("time", "60", "min"), "mins": ("time", "60", "min"), "minute": ("time", "60", "min"), "minutes": ("time", "60", "min"),
    "h": ("time", "3600", "h"), "hr": ("time", "3600", "h"), "hrs": ("time", "3600", "h"), "hour": ("time", "3600", "h"), "hours": ("time", "3600", "h"),
    "d": ("time", "86400", "d"), "day": ("time", "86400", "d"), "days": ("time", "86400", "d"),
    "wk": ("time", "604800", "wk"), "week": ("time", "604800", "wk"), "weeks": ("time", "604800", "wk"),
    "yr": ("time", "31557600", "yr"), "year": ("time", "31557600", "yr"), "years": ("time", "31557600", "yr"),

    # Data size (base: byte)
    "bit": ("data", "0.125", "bit"), "bits": ("data", "0.125", "bit"),
    "kbit": ("data", "125", "kbit"), "mbit": ("data", "125000", "mbit"), "gbit": ("data", "125000000", "gbit"),
    "b": ("data", "1", "B"), "byte": ("data", "1", "B"), "bytes": ("data", "1", "B"),
    "kb": ("data", "1000", "KB"), "kilobyte": ("data", "1000", "KB"), "kilobytes": ("data", "1000", "KB"),
    "mb": ("data", "1000000", "MB"), "megabyte": ("data", "1000000", "MB"), "megabytes": ("data", "1000000", "MB"),
    "gb": ("data", "1000000000", "GB"), "gigabyte": ("data", "1000000000", "GB"), "gigabytes": ("data", "1000000000", "GB"),
    "tb": ("data", "1000000000000", "TB"), "terabyte": ("data", "1000000000000", "TB"), "terabytes": ("data", "1000000000000", "TB"),
    "pb": ("data", "1000000000000000", "PB"),
    "kib": ("data", "1024", "KiB"), "kibibyte": ("data", "1024", "KiB"), "kibibytes": ("data", "1024", "KiB"),
    "mib": ("data", "1048576", "MiB"), "mebibyte": ("data", "1048576", "MiB"), "mebibytes": ("data", "1048576", "MiB"),
    "gib": ("data", "1073741824", "GiB"), "gibibyte": ("data", "1073741824", "GiB"), "gibibytes": ("data", "1073741824", "GiB"),
    "tib": ("data", "1099511627776", "TiB"), "tebibyte": ("data", "1099511627776", "TiB"), "tebibytes": ("data", "1099511627776", "TiB"),

    # Volume (base: liter)
    "ml": ("volume", "0.001", "mL"), "milliliter": ("volume", "0.001", "mL"), "milliliters": ("volume", "0.001", "mL"), "millilitre": ("volume", "0.001", "mL"), "millilitres": ("volume", "0.001", "mL"), "cc": ("volume", "0.001", "mL"),
    "l": ("volume", "1", "L"), "liter": ("volume", "1", "L"), "liters": ("volume", "1", "L"), "litre": ("volume", "1", "L"), "litres": ("volume", "1", "L"),
    "m3": ("volume", "1000", "m^3"), "cubic_meter": ("volume", "1000", "m^3"), "cubic_meters": ("volume", "1000", "m^3"),
    "tsp": ("volume", "0.00492892159375", "tsp"), "teaspoon": ("volume", "0.00492892159375", "tsp"), "teaspoons": ("volume", "0.00492892159375", "tsp"),
    "tbsp": ("volume", "0.01478676478125", "tbsp"), "tablespoon": ("volume", "0.01478676478125", "tbsp"), "tablespoons": ("volume", "0.01478676478125", "tbsp"),
    "fl_oz": ("volume", "0.0295735295625", "fl oz"), "floz": ("volume", "0.0295735295625", "fl oz"), "fluid_ounce": ("volume", "0.0295735295625", "fl oz"), "fluid_ounces": ("volume", "0.0295735295625", "fl oz"),
    "cup": ("volume", "0.2365882365", "cup"), "cups": ("volume", "0.2365882365", "cup"),
    "pt": ("volume", "0.473176473", "pt"), "pint": ("volume", "0.473176473", "pt"), "pints": ("volume", "0.473176473", "pt"),
    "qt": ("volume", "0.946352946", "qt"), "quart": ("volume", "0.946352946", "qt"), "quarts": ("volume", "0.946352946", "qt"),
    "gal": ("volume", "3.785411784", "gal"), "gallon": ("volume", "3.785411784", "gal"), "gallons": ("volume", "3.785411784", "gal"),

    # Area (base: m^2)
    "mm2": ("area", "1e-6", "mm^2"), "sqmm": ("area", "1e-6", "mm^2"),
    "cm2": ("area", "1e-4", "cm^2"), "sqcm": ("area", "1e-4", "cm^2"),
    "m2": ("area", "1", "m^2"), "sqm": ("area", "1", "m^2"), "square_meter": ("area", "1", "m^2"), "square_meters": ("area", "1", "m^2"),
    "ha": ("area", "10000", "ha"), "hectare": ("area", "10000", "ha"), "hectares": ("area", "10000", "ha"),
    "km2": ("area", "1000000", "km^2"), "sqkm": ("area", "1000000", "km^2"), "square_kilometer": ("area", "1000000", "km^2"), "square_kilometers": ("area", "1000000", "km^2"),
    "in2": ("area", "0.00064516", "in^2"), "sqin": ("area", "0.00064516", "in^2"),
    "ft2": ("area", "0.09290304", "ft^2"), "sqft": ("area", "0.09290304", "ft^2"), "square_foot": ("area", "0.09290304", "ft^2"), "square_feet": ("area", "0.09290304", "ft^2"),
    "yd2": ("area", "0.83612736", "yd^2"), "sqyd": ("area", "0.83612736", "yd^2"),
    "ac": ("area", "4046.8564224", "ac"), "acre": ("area", "4046.8564224", "ac"), "acres": ("area", "4046.8564224", "ac"),
    "mi2": ("area", "2589988.110336", "mi^2"), "sqmi": ("area", "2589988.110336", "mi^2"),

    # Speed (base: m/s)
    "m/s": ("speed", "1", "m/s"), "mps": ("speed", "1", "m/s"),
    "km/h": ("speed", "0.27777777777777777778", "km/h"), "kph": ("speed", "0.27777777777777777778", "km/h"), "kmh": ("speed", "0.27777777777777777778", "km/h"),
    "mph": ("speed", "0.44704", "mph"),
    "ft/s": ("speed", "0.3048", "ft/s"), "fps": ("speed", "0.3048", "ft/s"),
    "kn": ("speed", "0.51444444444444444444", "kn"), "knot": ("speed", "0.51444444444444444444", "kn"), "knots": ("speed", "0.51444444444444444444", "kn"),
}

_TEMP_ALIASES = {
    "c": ("C", "°C"), "celsius": ("C", "°C"), "centigrade": ("C", "°C"), "degc": ("C", "°C"), "°c": ("C", "°C"),
    "f": ("F", "°F"), "fahrenheit": ("F", "°F"), "degf": ("F", "°F"), "°f": ("F", "°F"),
    "k": ("K", "K"), "kelvin": ("K", "K"), "kelvins": ("K", "K"), "degk": ("K", "K"), "°k": ("K", "K"),
}


def _norm_unit(u: str) -> str:
    s = str(u or "").strip().lower()
    s = re.sub(r"^degrees?\s+", "", s)
    s = s.replace("^", "").replace("²", "2").replace("³", "3")
    s = re.sub(r"\s+", "_", s)
    return s


def convert_units(value, from_unit: str, to_unit: str) -> str:
    """Deterministically converts `value` from `from_unit` to `to_unit`."""
    try:
        s_val = str(value).replace(",", "").strip()
        val_d = Decimal(s_val)
        if not val_d.is_finite() or abs(val_d) > Decimal("1e100"):
            return "Error: value is out of range."
    except (InvalidOperation, ValueError, TypeError):
        return f"Error: invalid numeric value {value!r}."

    u_from = _norm_unit(from_unit)
    u_to = _norm_unit(to_unit)

    with decimal.localcontext() as ctx:
        ctx.prec = 36
        if u_from in _TEMP_ALIASES or u_to in _TEMP_ALIASES:
            if u_from not in _TEMP_ALIASES or u_to not in _TEMP_ALIASES:
                return f"Error: incompatible units {from_unit!r} and {to_unit!r}."
            k_from, sym_from = _TEMP_ALIASES[u_from]
            k_to, sym_to = _TEMP_ALIASES[u_to]
            # Convert to Kelvin first
            if k_from == "C":
                kelvin = val_d + Decimal("273.15")
            elif k_from == "F":
                kelvin = (val_d - Decimal("32")) * Decimal("5") / Decimal("9") + Decimal("273.15")
            else:
                kelvin = val_d
            if kelvin < Decimal("-1e-9"):
                return f"Error: temperature {_fmt_number(val_d)} {sym_from} is below absolute zero (0 K)."
            if k_to == "C":
                out_d = kelvin - Decimal("273.15")
            elif k_to == "F":
                out_d = (kelvin - Decimal("273.15")) * Decimal("9") / Decimal("5") + Decimal("32")
            else:
                out_d = kelvin
            out_round = Decimal(str(round(float(out_d), 8)))
            return f"{_fmt_number(val_d)} {sym_from} = {_fmt_number(out_round)} {sym_to}"

        if u_from not in _UNIT_RAW:
            return f"Error: unknown unit {from_unit!r}."
        if u_to not in _UNIT_RAW:
            return f"Error: unknown unit {to_unit!r}."

        cat_f, fac_f, sym_f = _UNIT_RAW[u_from]
        cat_t, fac_t, sym_t = _UNIT_RAW[u_to]
        if cat_f != cat_t:
            return f"Error: incompatible unit dimensions ({from_unit!r} is {cat_f}, {to_unit!r} is {cat_t})."

        base_val = val_d * Decimal(fac_f)
        out_d = base_val / Decimal(fac_t)
        out_round = Decimal(str(round(float(out_d), 10))) if abs(out_d) >= Decimal("1e-6") else out_d
        return f"{_fmt_number(val_d)} {sym_f} = {_fmt_number(out_round)} {sym_t}"


# ---------------------------------------------------------------------------
# 3. search_files — Bounded File-Name & Literal Content Search
# ---------------------------------------------------------------------------
def search_files(
    root: str | Path | None,
    query: str = "",
    mode: str = "content",
    glob_pat: str = "*",
    max_results: int = SEARCH_DEFAULT_RESULTS,
    *,
    base_dir: Path,
    workdir: Path | None = None,
    user_paths: tuple | list = (),
    cancel=None,
) -> str:
    root_dir, err = resolve_authorized_root(root, base_dir, workdir, user_paths=user_paths)
    if err or root_dir is None:
        return err or "Error: invalid root folder."
    q = str(query or "").strip()
    g_pat = str(glob_pat or "*").strip() or "*"
    if not q and g_pat != "*":
        mode = "name"
    if not q and mode != "name":
        return "Error: query is required for content search."
    search_mode = "name" if str(mode or "").strip().lower() == "name" else "content"
    try:
        cap = max(1, min(SEARCH_MAX_RESULTS_CAP, int(max_results or SEARCH_DEFAULT_RESULTS)))
    except (TypeError, ValueError):
        cap = SEARCH_DEFAULT_RESULTS

    q_low = q.lower()
    t0 = time.monotonic()
    matches: list[str] = []
    files_scanned = 0
    truncated = False
    stop_reason = ""

    def _walk(cur_dir: Path, depth: int):
        nonlocal files_scanned, truncated, stop_reason
        if depth > SEARCH_MAX_DEPTH or truncated:
            return
        if cancel is not None and cancel.is_set():
            truncated = True
            stop_reason = "cancelled"
            return
        if time.monotonic() - t0 > SEARCH_TIMEOUT_S:
            truncated = True
            stop_reason = "time limit reached"
            return
        try:
            entries = sorted(os.scandir(cur_dir), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
        except OSError:
            return

        for ent in entries:
            if truncated:
                return
            ent_path = Path(ent.path)
            if is_reparse_or_symlink(ent_path):
                continue
            try:
                resolved = ent_path.resolve()
            except OSError:
                continue
            if not paths._inside(resolved, root_dir) or paths.read_blocked(base_dir, resolved) or paths._inside(resolved, paths.repo_dir(base_dir)):
                continue

            if ent.is_dir(follow_symlinks=False):
                if ent.name in SKIP_DIRS or ent.name.startswith(".git"):
                    continue
                _walk(resolved, depth + 1)
                continue

            if not ent.is_file(follow_symlinks=False):
                continue
            if g_pat != "*" and not fnmatch.fnmatch(ent.name.lower(), g_pat.lower()):
                continue

            files_scanned += 1
            if files_scanned > SEARCH_MAX_FILES:
                truncated = True
                stop_reason = f"scanned {SEARCH_MAX_FILES} files"
                return

            rel = str(resolved.relative_to(root_dir)).replace("\\", "/")
            if search_mode == "name":
                if not q_low or q_low in ent.name.lower() or fnmatch.fnmatch(ent.name.lower(), f"*{q_low}*"):
                    try:
                        sz = ent.stat(follow_symlinks=False).st_size
                    except OSError:
                        sz = 0
                    matches.append(f"{rel}  ({sz} bytes)")
                    if len(matches) >= cap:
                        truncated = True
                        stop_reason = f"reached {cap} results"
                        return
                continue

            # Content search mode
            try:
                st = ent.stat(follow_symlinks=False)
                if st.st_size == 0 or st.st_size > SEARCH_MAX_FILE_BYTES:
                    continue
                raw = resolved.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            text = raw.decode("utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), 1):
                if q_low in line.lower():
                    snippet = " ".join(line.strip().split())[:160]
                    matches.append(f"{rel}:{line_no}: {snippet}")
                    if len(matches) >= cap:
                        truncated = True
                        stop_reason = f"reached {cap} results"
                        return

    _walk(root_dir, 0)
    header = (
        f"Search ({search_mode}) in {root_dir} for {q!r} (glob={g_pat!r}): "
        f"{len(matches)} match(es) across {files_scanned} file(s) scanned, truncated={str(truncated).lower()}"
        + (f" ({stop_reason})" if stop_reason else "")
    )
    if not matches:
        return header + "\n(no matches)"
    return header + "\n" + "\n".join(matches)


# ---------------------------------------------------------------------------
# 4. file_info — Metadata & Optional Streamed SHA-256
# ---------------------------------------------------------------------------
def file_info(raw_path: str | Path | None, sha256: bool = False, *, base_dir: Path, workdir: Path | None = None) -> str:
    path, err = resolve_authorized_path(raw_path, base_dir, workdir, default_to_workdir=False)
    if err or path is None:
        return err or "Error: invalid path."
    if not path.exists():
        return f"Error: {path} does not exist."
    try:
        st = path.stat()
    except OSError as exc:
        return f"Error: could not stat {path}: {exc}"

    kind = "directory" if path.is_dir() else ("file" if path.is_file() else "other")
    mtime_utc = datetime.datetime.fromtimestamp(st.st_mtime, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ctime_utc = datetime.datetime.fromtimestamp(st.st_ctime, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    info = {
        "path": str(path),
        "name": path.name,
        "type": kind,
        "extension": path.suffix.lower() if path.is_file() else "",
        "size_bytes": st.st_size if path.is_file() else 0,
        "modified_utc": mtime_utc,
        "created_utc": ctime_utc,
    }
    if path.is_file():
        try:
            with open(path, "rb") as fh:
                head = fh.read(4096)
            is_bin = b"\x00" in head
            info["encoding_kind"] = "binary" if is_bin else "text"
            if not is_bin and st.st_size <= 10 * 1024 * 1024:
                info["lines"] = len(path.read_bytes().decode("utf-8", errors="replace").splitlines())
        except OSError:
            pass

        if sha256:
            if st.st_size > HASH_MAX_BYTES:
                info["sha256_error"] = f"file exceeds {HASH_MAX_BYTES >> 20} MB hash limit"
            else:
                h = hashlib.sha256()
                t0 = time.monotonic()
                try:
                    timed_out = False
                    with open(path, "rb") as fh:
                        while True:
                            chunk = fh.read(1 << 20)
                            if not chunk:
                                break
                            h.update(chunk)
                            if time.monotonic() - t0 > HASH_TIMEOUT_S:
                                info["sha256_error"] = f"hash timed out after {HASH_TIMEOUT_S:.0f} s"
                                timed_out = True
                                break
                    if not timed_out:
                        info["sha256"] = h.hexdigest()
                except OSError as exc:
                    info["sha256_error"] = str(exc)

    return json.dumps(info, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 5. git_inspect — Fixed Read-Only Git Status, Diff, Log, and Show
# ---------------------------------------------------------------------------
def _find_git_exe(base_dir: Path) -> str:
    mingit = Path(base_dir) / "tools" / "git" / "cmd" / ("git.exe" if os.name == "nt" else "git")
    if mingit.is_file():
        return str(mingit)
    return shutil.which("git") or "git"


def git_inspect(
    operation: str,
    path: str | Path | None = None,
    target: str = "",
    limit: int = 10,
    *,
    base_dir: Path,
    workdir: Path | None = None,
    user_paths: tuple | list = (),
    cancel=None,
) -> str:
    op = str(operation or "status").strip().lower()
    if op not in ("status", "diff", "log", "show"):
        return f"Error: operation must be one of status, diff, log, show (got {operation!r})."

    repo_root, err = resolve_authorized_root(path, base_dir, workdir, user_paths=user_paths)
    if err or repo_root is None:
        return err or "Error: invalid repository path."

    # Verify it is inside a Git working tree (and not Clover's internal repo)
    cur = repo_root
    has_git = False
    while True:
        if (cur / ".git").exists():
            if paths._inside(cur, paths.repo_dir(base_dir)):
                return "Error: Clover's internal records repository is not inspected by git_inspect."
            has_git = True
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if not has_git:
        return f"Error: {repo_root} is not a Git repository."

    tgt = str(target or "").strip()
    if tgt:
        if tgt.startswith("-"):
            return f"Error: invalid target {tgt!r} (flags are not allowed in target)."
        if any(c in tgt for c in ("\n", "\r", "\x00")):
            return "Error: target contains invalid control characters."

    try:
        n_lim = max(1, min(50, int(limit or 10)))
    except (TypeError, ValueError):
        n_lim = 10

    git_exe = _find_git_exe(base_dir)
    if op == "status":
        cmd = [git_exe, "--no-pager", "status", "--short", "--branch"]
        if tgt:
            cmd += ["--", tgt]
    elif op == "diff":
        cmd = [git_exe, "--no-pager", "diff", "--no-ext-diff", "--stat=120", "-p"]
        if tgt:
            cmd += [tgt]
    elif op == "log":
        cmd = [git_exe, "--no-pager", "log", "--no-decorate", f"-n{n_lim}", "--oneline"]
        if tgt:
            cmd += [tgt]
    else:  # show
        ref = tgt or "HEAD"
        cmd = [git_exe, "--no-pager", "show", "--no-ext-diff", "--stat=120", ref]

    env = dict(os.environ, **local_terminal.QUIET_GIT)
    out, code, how = local_terminal.run(cmd, cwd=str(repo_root), timeout_s=30, cancel=cancel, env=env)
    return local_terminal.result_text(out, code, how, 30)


# ---------------------------------------------------------------------------
# 6. read_document — Paginated Document Reader (PDF, PPTX, XLSX/CSV, DOCX, TXT)
# ---------------------------------------------------------------------------
def read_document(
    raw_path: str | Path | None,
    page: int = 1,
    end_page: int = 0,
    sheet: str = "",
    start_row: int = 1,
    max_rows: int = 50,
    *,
    base_dir: Path,
    workdir: Path | None = None,
) -> str:
    import documents

    path, err = resolve_authorized_path(raw_path, base_dir, workdir, default_to_workdir=False)
    if err or path is None:
        return err or "Error: invalid document path."
    if not path.is_file():
        return f"Error: file {path} does not exist."

    ext = path.suffix.lower()
    if ext not in documents.KINDS:
        return f"Error: unsupported document extension {ext!r} (supported: {', '.join(sorted(documents.KINDS))})."

    try:
        p_start = max(1, int(page or 1))
        p_end = max(p_start, int(end_page or p_start + 4))
        p_end = min(p_end, p_start + 9)
        r_start = max(1, int(start_row or 1))
        r_count = max(1, min(200, int(max_rows or 50)))
    except (TypeError, ValueError):
        p_start, p_end, r_start, r_count = 1, 5, 1, 50

    if ext == ".pdf":
        from pypdf import PdfReader
        rd = PdfReader(str(path))
        total_pages = len(rd.pages)
        p_end = min(total_pages, p_end)
        out = [f"PDF: {path.name} (total pages: {total_pages}, showing {p_start}..{p_end})"]
        for i in range(p_start, p_end + 1):
            txt = re.sub(r"[ \t]+", " ", rd.pages[i - 1].extract_text() or "").strip()
            out.append(f"[page {i}]\n{txt or '(no extractable text on this page)'}")
        return "\n\n".join(out)

    if ext == ".pptx":
        from pptx import Presentation
        pr = Presentation(str(path))
        total_slides = len(pr.slides)
        p_end = min(total_slides, p_end)
        out = [f"PowerPoint: {path.name} (total slides: {total_slides}, showing {p_start}..{p_end})"]
        for i in range(p_start, p_end + 1):
            slide = pr.slides[i - 1]
            texts = [sh.text_frame.text.strip() for sh in slide.shapes if getattr(sh, "has_text_frame", False) and sh.text_frame.text.strip()]
            out.append(f"[slide {i}] " + (" / ".join(texts) if texts else "(empty slide)"))
        return "\n".join(out)

    if ext in (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".json"):
        df, sheet_names, sel_sheet, total_rows, truncated_scan = _load_dataframe(path, sheet)
        if isinstance(df, str):
            return df
        end_idx = min(len(df), r_start - 1 + r_count)
        sub = df.iloc[r_start - 1 : end_idx]
        head_info = (
            f"Document: {path.name}"
            + (f" (sheet={sel_sheet!r}, sheets={sheet_names})" if sheet_names else "")
            + f" | rows {r_start}..{end_idx} of {total_rows} (truncated_scan={str(truncated_scan).lower()})"
        )
        cols = [str(c) for c in df.columns]
        lines = [head_info, "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for idx_r, row in sub.iterrows():
            lines.append("| " + " | ".join(documents._cell(row[c]) for c in df.columns) + " |")
        return "\n".join(lines)

    if ext == ".docx":
        import docx
        d = docx.Document(str(path))
        paras = [p.text.strip() for p in d.paragraphs if p.text.strip()]
        end_p = min(len(paras), r_start - 1 + r_count)
        body = "\n".join(f"{i}: {paras[i - 1]}" for i in range(r_start, end_p + 1))
        return f"Word: {path.name} (paragraphs {r_start}..{end_p} of {len(paras)})\n{body}"

    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    end_l = min(len(lines), r_start - 1 + r_count)
    body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(r_start, end_l + 1))
    return f"Text: {path.name} (lines {r_start}..{end_l} of {len(lines)})\n{body}"


# ---------------------------------------------------------------------------
# 7 & 8. analyze_table & chart_table — Fixed Deterministic Table Operations
# ---------------------------------------------------------------------------
def _load_dataframe(path: Path, sheet: str = ""):
    import pandas as pd

    ext = path.suffix.lower()
    sheet_names: list[str] = []
    sel_sheet = ""
    try:
        if ext in (".xlsx", ".xlsm", ".xls"):
            xl = pd.ExcelFile(path)
            sheet_names = list(xl.sheet_names)
            sel_sheet = sheet.strip() if (sheet and sheet.strip() in sheet_names) else sheet_names[0]
            df = xl.parse(sel_sheet, nrows=TABLE_SCAN_ROWS + 1)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=TABLE_SCAN_ROWS + 1)
        elif ext == ".csv":
            df = pd.read_csv(path, nrows=TABLE_SCAN_ROWS + 1)
        elif ext == ".json":
            df = pd.read_json(path)
        else:
            return f"Error: {path.name} is not a tabular file (supported: .xlsx, .xls, .csv, .tsv, .json).", [], "", 0, False
    except Exception as exc:
        return f"Error reading table {path.name}: {type(exc).__name__}: {exc}", [], "", 0, False

    truncated = len(df) > TABLE_SCAN_ROWS
    if truncated:
        df = df.iloc[:TABLE_SCAN_ROWS]
    return df, sheet_names, sel_sheet, len(df), truncated


def _resolve_col(df, col_name: str):
    c = str(col_name or "").strip()
    if not c:
        return None
    if c in df.columns:
        return c
    low_map = {str(k).lower(): k for k in df.columns}
    return low_map.get(c.lower())


def analyze_table(
    raw_path: str | Path | None,
    sheet: str = "",
    filter_column: str = "",
    filter_op: str = "==",
    filter_value: str = "",
    group_by: str = "",
    target_column: str = "",
    agg: str = "sum",
    sort_by: str = "",
    descending: bool = True,
    limit: int = 20,
    *,
    base_dir: Path,
    workdir: Path | None = None,
) -> str:
    import pandas as pd

    path, err = resolve_authorized_path(raw_path, base_dir, workdir, default_to_workdir=False)
    if err or path is None:
        return err or "Error: invalid table path."
    if not path.is_file():
        return f"Error: file {path} does not exist."

    df, sheet_names, sel_sheet, rows_read, truncated = _load_dataframe(path, sheet)
    if isinstance(df, str):
        return df

    if filter_column:
        fcol = _resolve_col(df, filter_column)
        if fcol is None:
            return f"Error: filter_column {filter_column!r} not in columns {list(df.columns)}."
        op = str(filter_op or "==").strip().lower()
        val_s = str(filter_value or "").strip()
        series = df[fcol]
        try:
            val_num = float(val_s.replace(",", ""))
            num_s = pd.to_numeric(series, errors="coerce")
            is_num_cmp = num_s.notna().any()
        except ValueError:
            val_num, num_s, is_num_cmp = None, None, False

        if op in ("==", "=", "eq"):
            df = df[num_s == val_num] if is_num_cmp and val_num is not None else df[series.astype(str).str.lower() == val_s.lower()]
        elif op in ("!=", "<>", "ne"):
            df = df[num_s != val_num] if is_num_cmp and val_num is not None else df[series.astype(str).str.lower() != val_s.lower()]
        elif op in (">", "gt") and val_num is not None:
            df = df[num_s > val_num]
        elif op in (">=", "gte") and val_num is not None:
            df = df[num_s >= val_num]
        elif op in ("<", "lt") and val_num is not None:
            df = df[num_s < val_num]
        elif op in ("<=", "lte") and val_num is not None:
            df = df[num_s <= val_num]
        elif op == "contains":
            df = df[series.astype(str).str.contains(val_s, case=False, na=False, regex=False)]
        else:
            return f"Error: unsupported filter_op {filter_op!r} (use ==, !=, >, >=, <, <=, or contains)."

    agg_op = str(agg or "sum").strip().lower()
    if agg_op not in ("sum", "mean", "average", "count", "min", "max", "median"):
        return f"Error: agg must be one of sum, mean, count, min, max, median (got {agg!r})."
    if agg_op == "average":
        agg_op = "mean"

    try:
        n_lim = max(1, min(100, int(limit or 20)))
    except (TypeError, ValueError):
        n_lim = 20

    tcol = _resolve_col(df, target_column) if target_column else None
    if target_column and tcol is None:
        return f"Error: target_column {target_column!r} not in columns {list(df.columns)}."

    gcol = _resolve_col(df, group_by) if group_by else None
    if group_by and gcol is None:
        return f"Error: group_by {group_by!r} not in columns {list(df.columns)}."

    result_obj = {
        "file": path.name,
        "sheet": sel_sheet or None,
        "rows_read": rows_read,
        "rows_matched": int(len(df)),
        "truncated": truncated,
        "columns": [str(c) for c in df.columns],
    }

    if gcol is not None:
        if tcol is not None:
            work = df.copy()
            if agg_op != "count":
                work[tcol] = pd.to_numeric(work[tcol], errors="coerce")
            grouped = getattr(work.groupby(gcol, dropna=False)[tcol], agg_op)().reset_index()
            scol = _resolve_col(grouped, sort_by) or tcol
            grouped = grouped.sort_values(by=scol, ascending=not bool(descending)).head(n_lim)
            result_obj["grouped_by"] = str(gcol)
            result_obj["target_column"] = str(tcol)
            result_obj["agg"] = agg_op
            result_obj["rows"] = [
                {str(gcol): str(r[gcol]), str(tcol): round(float(r[tcol]), 4) if pd.notna(r[tcol]) else None}
                for _, r in grouped.iterrows()
            ]
        else:
            counts = df.groupby(gcol, dropna=False).size().reset_index(name="count")
            counts = counts.sort_values(by="count", ascending=not bool(descending)).head(n_lim)
            result_obj["grouped_by"] = str(gcol)
            result_obj["agg"] = "count"
            result_obj["rows"] = [{str(gcol): str(r[gcol]), "count": int(r["count"])} for _, r in counts.iterrows()]
    elif tcol is not None:
        if agg_op == "count":
            val = int(df[tcol].notna().sum())
        else:
            s_num = pd.to_numeric(df[tcol], errors="coerce").dropna()
            val = round(float(getattr(s_num, agg_op)()), 6) if len(s_num) else None
        result_obj["target_column"] = str(tcol)
        result_obj["agg"] = agg_op
        result_obj["value"] = val
    else:
        scol = _resolve_col(df, sort_by)
        sub = df.sort_values(by=scol, ascending=not bool(descending)).head(n_lim) if scol else df.head(n_lim)
        result_obj["rows"] = json.loads(sub.to_json(orient="records", date_format="iso"))

    return json.dumps(result_obj, indent=2, ensure_ascii=False)


def chart_table(
    raw_path: str | Path | None,
    sheet: str = "",
    chart_type: str = "bar",
    x_column: str = "",
    y_column: str = "",
    agg: str = "sum",
    title: str = "",
    limit: int = 15,
    *,
    base_dir: Path,
    workdir: Path | None = None,
    turn_images: list | None = None,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    path, err = resolve_authorized_path(raw_path, base_dir, workdir, default_to_workdir=False)
    if err or path is None:
        return err or "Error: invalid table path."
    if not path.is_file():
        return f"Error: file {path} does not exist."

    df, _, sel_sheet, rows_read, truncated = _load_dataframe(path, sheet)
    if isinstance(df, str):
        return df

    kind = str(chart_type or "bar").strip().lower()
    if kind not in ("bar", "line", "scatter", "pie", "hist"):
        return f"Error: chart_type must be one of bar, line, scatter, pie, hist (got {chart_type!r})."

    xcol = _resolve_col(df, x_column) if x_column else (df.columns[0] if len(df.columns) else None)
    ycol = _resolve_col(df, y_column) if y_column else (df.columns[1] if len(df.columns) > 1 else xcol)
    if xcol is None or (kind != "hist" and ycol is None):
        return f"Error: column not found in {list(df.columns)}."

    try:
        n_lim = max(2, min(40, int(limit or 15)))
    except (TypeError, ValueError):
        n_lim = 15

    agg_op = str(agg or "sum").strip().lower()
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    try:
        if kind == "hist":
            vals = pd.to_numeric(df[xcol], errors="coerce").dropna()
            ax.hist(vals, bins=min(20, max(5, len(vals) // 5 or 5)), color="#176B2C", edgecolor="#FFFDF8")
            ax.set_xlabel(str(xcol))
            ax.set_ylabel("Count")
        else:
            work = df.copy()
            work[ycol] = pd.to_numeric(work[ycol], errors="coerce")
            work = work.dropna(subset=[ycol])
            if agg_op in ("sum", "mean", "count", "min", "max", "median") and kind in ("bar", "pie", "line"):
                plot_df = getattr(work.groupby(xcol, dropna=False)[ycol], agg_op)().reset_index()
            else:
                plot_df = work[[xcol, ycol]]
            plot_df = plot_df.head(n_lim)
            labels = [str(v)[:28] for v in plot_df[xcol]]
            values = [float(v) for v in plot_df[ycol]]
            if kind == "bar":
                ax.bar(labels, values, color="#176B2C")
                plt.xticks(rotation=30, ha="right")
                ax.set_ylabel(f"{ycol} ({agg_op})")
            elif kind == "line":
                ax.plot(labels, values, marker="o", color="#176B2C", linewidth=2)
                plt.xticks(rotation=30, ha="right")
                ax.set_ylabel(str(ycol))
            elif kind == "scatter":
                xv = pd.to_numeric(plot_df[xcol], errors="coerce")
                ax.scatter(xv if xv.notna().all() else labels, values, color="#176B2C")
                ax.set_xlabel(str(xcol))
                ax.set_ylabel(str(ycol))
            elif kind == "pie":
                ax.pie([max(0.0, v) for v in values], labels=labels, autopct="%1.1f%%")

        chart_title = title.strip() if title else f"{ycol} by {xcol} ({path.name})"
        ax.set_title(chart_title)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        png_bytes = buf.getvalue()
    finally:
        plt.close(fig)

    media = paths.media_dir(base_dir)
    media.mkdir(parents=True, exist_ok=True)
    fname = hashlib.sha256(png_bytes).hexdigest()[:32] + ".png"
    out_path = media / fname
    out_path.write_bytes(png_bytes)
    if turn_images is not None:
        turn_images.append({
            "file": fname,
            "prompt": chart_title,
            "model": "chart_table",
            "seconds": 0,
            "width": 1050,
            "height": 630,
        })
    return (
        f"Created {kind} chart {chart_title!r} from {path.name} (rows_read={rows_read}, truncated={str(truncated).lower()}); "
        f"saved as {out_path.name} and shown to the user in the chat."
    )


# ---------------------------------------------------------------------------
# 9 & 10. media_info & extract_frame — Bounded FFprobe / FFmpeg Media Tools
# ---------------------------------------------------------------------------
def _ffmpeg_tool_path(base_dir: Path | None, binary_name: str) -> str | None:
    exe_name = f"{binary_name}.exe" if os.name == "nt" else binary_name
    if base_dir:
        ff_txt = Path(base_dir) / "tools" / "ffmpeg" / "BIN.txt"
        if ff_txt.is_file():
            d = Path(ff_txt.read_text(encoding="utf-8").strip())
            if (d / exe_name).is_file():
                return str(d / exe_name)
    return shutil.which(binary_name)


find_media_binary = _ffmpeg_tool_path


def media_info(raw_path: str | Path | None, *, base_dir: Path, workdir: Path | None = None, cancel=None) -> str:
    path, err = resolve_authorized_path(raw_path, base_dir, workdir, default_to_workdir=False)
    if err or path is None:
        return err or "Error: invalid media path."
    if not path.is_file():
        return f"Error: file {path} does not exist."

    is_img = path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
    img_rep = None
    if is_img:
        try:
            import image_inspector
            img_rep = image_inspector.inspect_image(path)
        except Exception:
            img_rep = None

    ffprobe = _ffmpeg_tool_path(base_dir, "ffprobe")
    if not ffprobe:
        if img_rep is not None:
            return json.dumps(img_rep, indent=2, ensure_ascii=False)
        return "Error: media_info needs ffprobe (run Clover AI Studio setup again)."

    cmd = [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    out, code, how = local_terminal.run(cmd, timeout_s=20, cancel=cancel)
    if how != "exit" or code != 0:
        if img_rep is not None:
            return json.dumps(img_rep, indent=2, ensure_ascii=False)
        return f"Error: ffprobe failed ({how}, code={code}): {out[:400]}"
    try:
        raw = json.loads(out)
    except ValueError:
        if img_rep is not None:
            return json.dumps(img_rep, indent=2, ensure_ascii=False)
        return "Error: ffprobe returned invalid JSON."

    fmt = raw.get("format") or {}
    streams = []
    for s in (raw.get("streams") or [])[:8]:
        streams.append({
            k: s.get(k)
            for k in ("index", "codec_type", "codec_name", "width", "height", "sample_rate", "channels", "r_frame_rate", "duration")
            if s.get(k) is not None
        })
    summary = {
        "file": path.name,
        "path": str(path),
        "format_name": fmt.get("format_name"),
        "duration_s": round(float(fmt["duration"]), 3) if fmt.get("duration") else None,
        "size_bytes": int(fmt["size"]) if fmt.get("size") else path.stat().st_size,
        "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        "streams": streams,
    }
    if img_rep is not None:
        summary["visual_inspection"] = {
            "width": img_rep["width"],
            "height": img_rep["height"],
            "aspect_ratio": img_rep["aspect_ratio"],
            "palette": img_rep["palette"],
            "elements": img_rep["elements"],
            "ocr_text": img_rep["ocr_text"],
            "low_contrast_refs": img_rep["low_contrast_refs"],
        }
    return json.dumps(summary, indent=2, ensure_ascii=False)


def extract_frame(
    raw_path: str | Path | None,
    timestamp_s: float = 0.0,
    max_width: int = 960,
    *,
    base_dir: Path,
    workdir: Path | None = None,
    turn_images: list | None = None,
    cancel=None,
) -> str:
    path, err = resolve_authorized_path(raw_path, base_dir, workdir, default_to_workdir=False)
    if err or path is None:
        return err or "Error: invalid video path."
    if not path.is_file():
        return f"Error: file {path} does not exist."

    try:
        raw_ts = float(timestamp_s if timestamp_s is not None else 0.0)
        if raw_ts < 0:
            return "Error: timestamp_s must be >= 0."
        ts = raw_ts
        mw = max(128, min(1280, int(max_width or 960)))
    except (TypeError, ValueError):
        return "Error: timestamp_s must be a non-negative number."

    ffmpeg = _ffmpeg_tool_path(base_dir, "ffmpeg")
    if not ffmpeg:
        return "Error: extract_frame needs ffmpeg (run Clover AI Studio setup again)."

    media = paths.media_dir(base_dir)
    out_dir = media / "out" / uuid.uuid4().hex[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_png = out_dir / "frame.png"
    try:
        cmd = [
            ffmpeg, "-y", "-loglevel", "error", "-ss", f"{ts:.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale='min({mw},iw)':-1", str(tmp_png),
        ]
        out, code, how = local_terminal.run(cmd, timeout_s=30, cancel=cancel)
        if how != "exit" or code != 0 or not tmp_png.is_file() or tmp_png.stat().st_size == 0:
            return f"Error: could not extract frame at {ts:.2f}s ({out.strip()[:300] or how})."
        png_bytes = tmp_png.read_bytes()
        fname = hashlib.sha256(png_bytes).hexdigest()[:32] + ".png"
        (media / fname).write_bytes(png_bytes)
        if turn_images is not None:
            turn_images.append({
                "file": fname,
                "prompt": f"frame from {path.name} at {ts:.2f}s",
                "model": "ffmpeg",
                "seconds": round(ts, 2),
                "width": None,
                "height": None,
            })
        return f"Extracted frame from {path.name} at {ts:.2f} s ({len(png_bytes):,} bytes); saved as {fname} and shown in the chat."
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tool Schemas & Permissions Catalogue
# ---------------------------------------------------------------------------
PACK_TOOLS = [
    _fn(
        "calculate",
        "Evaluate an arithmetic or mathematical expression deterministically (+, -, *, /, //, %, **, sqrt, abs, round, "
        "min, max, floor, ceil, sin, cos, tan, log, log10, log2, factorial, gcd, pi, e).",
        {"expression": {"type": "string", "description": "arithmetic expression, e.g. '137 * 29 + 46'"}},
        ["expression"],
    ),
    _fn(
        "convert_units",
        "Convert a numeric quantity between compatible units (length, mass, time, temperature, data size, volume, area, speed).",
        {
            "value": {"type": "number", "description": "numeric value to convert"},
            "from_unit": {"type": "string", "description": "source unit, e.g. 'km', 'lb', 'C', 'GiB'"},
            "to_unit": {"type": "string", "description": "target unit, e.g. 'mi', 'kg', 'F', 'MB'"},
        },
        ["value", "from_unit", "to_unit"],
    ),
    _fn(
        "search_files",
        "Search for files by name or literal text content inside an authorized folder (bounded depth and result count).",
        {
            "root": {"type": "string", "description": "folder to search (default: active workspace or home folder)"},
            "query": {"type": "string", "description": "file name substring or literal text to search for"},
            "mode": {"type": "string", "description": "'content' (default) or 'name'"},
            "glob": {"type": "string", "description": "optional filename glob filter, e.g. '*.py'"},
            "max_results": {"type": "integer", "description": "maximum matches to return (1-100, default 25)"},
        },
        ["query"],
    ),
    _fn(
        "file_info",
        "Read a file's metadata (size, timestamps, type, line count) and optionally compute its streamed SHA-256 hash.",
        {
            "path": {"type": "string", "description": "file or folder path"},
            "sha256": {"type": "boolean", "description": "true to compute the file's SHA-256 checksum"},
        },
        ["path"],
    ),
    _fn(
        "git_inspect",
        "Run a fixed read-only Git inspection (status, diff, log, or show) in an authorized Git repository.",
        {
            "operation": {"type": "string", "description": "'status', 'diff', 'log', or 'show'"},
            "path": {"type": "string", "description": "repository folder (default: active workspace)"},
            "target": {"type": "string", "description": "optional commit ref or relative file path (no flags)"},
            "limit": {"type": "integer", "description": "number of log entries for 'log' (1-50, default 10)"},
        },
        ["operation"],
    ),
    _fn(
        "read_document",
        "Read specific pages, slides, sheets, or row ranges of a document (PDF, PowerPoint, Excel, CSV, Word, JSON, text).",
        {
            "path": {"type": "string", "description": "path to the document"},
            "page": {"type": "integer", "description": "1-based start page or slide (for PDF/PPTX)"},
            "end_page": {"type": "integer", "description": "1-based end page or slide (up to 10 pages per call)"},
            "sheet": {"type": "string", "description": "sheet name for Excel workbooks"},
            "start_row": {"type": "integer", "description": "1-based start row or paragraph"},
            "max_rows": {"type": "integer", "description": "maximum rows or paragraphs to return (1-200, default 50)"},
        },
        ["path"],
    ),
    _fn(
        "analyze_table",
        "Run a deterministic filter, group-by, aggregation (sum, mean, count, min, max, median), or sort on a table or spreadsheet.",
        {
            "path": {"type": "string", "description": "path to .xlsx, .xls, .csv, .tsv, or .json file"},
            "sheet": {"type": "string", "description": "optional sheet name for Excel workbooks"},
            "filter_column": {"type": "string", "description": "optional column to filter on"},
            "filter_op": {"type": "string", "description": "==, !=, >, >=, <, <=, or contains"},
            "filter_value": {"type": "string", "description": "value to compare against"},
            "group_by": {"type": "string", "description": "optional column to group by"},
            "target_column": {"type": "string", "description": "numeric column to aggregate"},
            "agg": {"type": "string", "description": "sum, mean, count, min, max, or median"},
            "sort_by": {"type": "string", "description": "optional column to sort results by"},
            "descending": {"type": "boolean", "description": "sort descending (default true)"},
            "limit": {"type": "integer", "description": "max rows to return (1-100, default 20)"},
        },
        ["path"],
    ),
    _fn(
        "chart_table",
        "Create a bar, line, scatter, pie, or histogram PNG chart from a spreadsheet or CSV table and show it in the chat.",
        {
            "path": {"type": "string", "description": "path to .xlsx, .xls, .csv, .tsv, or .json file"},
            "sheet": {"type": "string", "description": "optional sheet name for Excel workbooks"},
            "chart_type": {"type": "string", "description": "bar, line, scatter, pie, or hist"},
            "x_column": {"type": "string", "description": "column for the X axis or categories"},
            "y_column": {"type": "string", "description": "numeric column for the Y axis"},
            "agg": {"type": "string", "description": "sum, mean, count, min, max, median, or none"},
            "title": {"type": "string", "description": "chart title"},
            "limit": {"type": "integer", "description": "max categories/points (2-40, default 15)"},
        },
        ["path"],
    ),
    _fn(
        "media_info",
        "Inspect an audio, video, or image file's format, duration, bit rate, and streams with FFprobe.",
        {"path": {"type": "string", "description": "path to the media file"}},
        ["path"],
    ),
    _fn(
        "extract_frame",
        "Extract a single video frame at a timestamp (in seconds) as a PNG image and show it in the chat.",
        {
            "path": {"type": "string", "description": "path to the video file"},
            "timestamp_s": {"type": "number", "description": "timestamp in seconds (default 0.0)"},
            "max_width": {"type": "integer", "description": "maximum image width in pixels (128-1280, default 960)"},
        },
        ["path"],
    ),
]

PACK_PERMISSIONS = {
    "calculate": None,
    "convert_units": None,
    "search_files": "files",
    "file_info": "files",
    "git_inspect": "files",
    "read_document": "files",
    "analyze_table": "files",
    "chart_table": "files",
    "media_info": "files",
    "extract_frame": "files",
}


def tool_detail(name: str, args: dict, workdir: Path | None = None) -> str:
    a = args or {}
    loc = str(a.get("path") or a.get("root") or workdir or Path.home())
    if name == "calculate":
        return f"Calculate: {a.get('expression', '')}"
    if name == "convert_units":
        return f"Convert {a.get('value', '')} {a.get('from_unit', '')} to {a.get('to_unit', '')}"
    if name == "search_files":
        return f"Search ({a.get('mode') or 'content'}) for {a.get('query')!r} in {loc}"
    if name == "file_info":
        return f"Read file info{' and SHA-256 checksum' if a.get('sha256') else ''}: {loc}"
    if name == "git_inspect":
        return f"Inspect Git ({a.get('operation') or 'status'} {a.get('target') or ''}) in {loc}".strip()
    if name == "read_document":
        return f"Read document: {loc}"
    if name == "analyze_table":
        return f"Analyze table ({a.get('agg') or 'sum'} {a.get('target_column') or ''}): {loc}"
    if name == "chart_table":
        return f"Chart table ({a.get('chart_type') or 'bar'}): {loc}"
    if name == "media_info":
        return f"Inspect media metadata: {loc}"
    if name == "extract_frame":
        return f"Extract video frame at {a.get('timestamp_s', 0)}s from {loc}"
    return f"{name}: {loc}"


def run_pack_tool(name: str, args: dict, engine) -> str:
    a = args or {}
    base_dir = Path(getattr(engine, "base_dir", Path.cwd()))
    workdir = getattr(engine, "workdir", None)
    user_paths = getattr(engine, "turn_paths", None) or getattr(engine, "user_paths", ()) or ()
    cancel = getattr(engine, "turn_cancel", None)
    turn_images = getattr(engine, "turn_images", None)

    if name == "calculate":
        return calculate(a.get("expression", ""))
    if name == "convert_units":
        return convert_units(a.get("value"), a.get("from_unit", ""), a.get("to_unit", ""))
    if name == "search_files":
        return search_files(
            a.get("root"),
            a.get("query", ""),
            mode=a.get("mode", "content"),
            glob_pat=a.get("glob") or a.get("pattern") or "*",
            max_results=a.get("max_results", SEARCH_DEFAULT_RESULTS),
            base_dir=base_dir,
            workdir=workdir,
            user_paths=user_paths,
            cancel=cancel,
        )
    if name == "file_info":
        return file_info(a.get("path"), sha256=bool(a.get("sha256", False)), base_dir=base_dir, workdir=workdir)
    if name == "git_inspect":
        return git_inspect(
            a.get("operation", "status"),
            path=a.get("path"),
            target=a.get("target", ""),
            limit=a.get("limit") or a.get("max_count") or 10,
            base_dir=base_dir,
            workdir=workdir,
            user_paths=user_paths,
            cancel=cancel,
        )
    if name == "read_document":
        return read_document(
            a.get("path"),
            page=a.get("page", 1),
            end_page=a.get("end_page", 0),
            sheet=a.get("sheet", ""),
            start_row=a.get("start_row", 1),
            max_rows=a.get("max_rows", 50),
            base_dir=base_dir,
            workdir=workdir,
        )
    if name == "analyze_table":
        return analyze_table(
            a.get("path"),
            sheet=a.get("sheet", ""),
            filter_column=a.get("filter_column", ""),
            filter_op=a.get("filter_op", "=="),
            filter_value=a.get("filter_value", ""),
            group_by=a.get("group_by", ""),
            target_column=a.get("target_column", ""),
            agg=a.get("agg", "sum"),
            sort_by=a.get("sort_by", ""),
            descending=bool(a.get("descending", True)),
            limit=a.get("limit", 20),
            base_dir=base_dir,
            workdir=workdir,
        )
    if name == "chart_table":
        return chart_table(
            a.get("path"),
            sheet=a.get("sheet", ""),
            chart_type=a.get("chart_type", "bar"),
            x_column=a.get("x_column") or a.get("x") or "",
            y_column=a.get("y_column") or a.get("y") or "",
            agg=a.get("agg", "sum"),
            title=a.get("title", ""),
            limit=a.get("limit", 15),
            base_dir=base_dir,
            workdir=workdir,
            turn_images=turn_images,
        )
    if name == "media_info":
        return media_info(a.get("path"), base_dir=base_dir, workdir=workdir, cancel=cancel)
    if name == "extract_frame":
        return extract_frame(
            a.get("path"),
            timestamp_s=a.get("timestamp_s", 0.0),
            max_width=a.get("max_width", 960),
            base_dir=base_dir,
            workdir=workdir,
            turn_images=turn_images,
            cancel=cancel,
        )
    raise ValueError(f"Unknown pack tool: {name}")
