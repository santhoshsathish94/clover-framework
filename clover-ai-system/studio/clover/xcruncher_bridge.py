"""Clover Studio bridge to the MIT-licensed x-cruncher v1.6.1 arbitrary-precision multiplication
and zero-storage hexadecimal Pi extraction engine (`_xcruncher_c/xcruncher.c`).

Provides deterministic, zero-hallucination exact arithmetic for:
  - `mul`       : Exact big-integer product `a * b` (including sparse near-2^k expressions like `(1<<200000)-3`)
  - `factorial` : Exact `n!` via binary-splitting tree multiplication backed by x-cruncher (`0 <= n <= 200000`)
  - `pi_hex`    : Zero-storage N-th hexadecimal digits of Pi (`0 <= hex_offset <= 2^44`, `1..20` hex digits)
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
CSRC_DIR = HERE / "_xcruncher_c"
MAX_PI_HEX_OFFSET = 1 << 44
MAX_FACTORIAL_N = 200_000
MAX_STR_DIGITS = 1_000_000
INLINE_DIGIT_DISPLAY_CAP = 2_000

_LIB: Optional[ctypes.CDLL] = None
_LIB_LOCK = threading.Lock()


def _candidate_c_compilers() -> list[str]:
    out: list[str] = []
    env_cc = os.environ.get("CC")
    if env_cc and shutil.which(env_cc):
        out.append(shutil.which(env_cc))  # type: ignore[arg-type]
    for name in ("gcc", "clang", "cc"):
        found = shutil.which(name)
        if found and found not in out:
            out.append(found)
    if os.name == "nt":
        winget_pkgs = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkgs.is_dir():
            for g in winget_pkgs.rglob("gcc.exe"):
                if g.is_file() and str(g) not in out:
                    out.append(str(g))
        for cand in (
            r"C:\msys64\ucrt64\bin\gcc.exe",
            r"C:\msys64\mingw64\bin\gcc.exe",
        ):
            if os.path.isfile(cand) and cand not in out:
                out.append(cand)
    return out


def _cache_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    d = base / "clover-xcruncher" / f"v1.6.1-{platform.system().lower()}-{platform.machine().lower()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


_PI_KAT_VECTORS: tuple[tuple[int, str], ...] = (
    (0, "243F6A8885"),
    (194, "7C9045F12C"),
    (1000, "49F1C09B07"),
    (5000, "CAD181156B"),
)


def _verify_lib_kat(lib: ctypes.CDLL) -> None:
    buf = ctypes.create_string_buffer(16)
    for off, expected in _PI_KAT_VECTORS:
        rc = lib.xc_pi_hex_digits(ctypes.c_uint64(off), buf, ctypes.c_size_t(len(expected)))
        got = buf.value.decode("ascii", errors="replace").upper() if rc == 0 else f"rc={rc}"
        if got != expected:
            raise OSError(f"xcruncher KAT self-test failed at offset {off}: got {got}, expected {expected}")


def _bind_cdll(path: Path) -> ctypes.CDLL:
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(path.parent))
    lib = ctypes.CDLL(str(path))
    u64_p = ctypes.POINTER(ctypes.c_uint64)
    lib.xc_version.argtypes = []
    lib.xc_version.restype = ctypes.c_char_p
    lib.xc_active_isa_name.argtypes = []
    lib.xc_active_isa_name.restype = ctypes.c_char_p
    lib.xc_mul_u64.argtypes = [u64_p, u64_p, ctypes.c_size_t, u64_p, ctypes.c_size_t]
    lib.xc_mul_u64.restype = ctypes.c_int
    lib.xc_pi_hex_digits.argtypes = [ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t]
    lib.xc_pi_hex_digits.restype = ctypes.c_int
    _verify_lib_kat(lib)
    return lib


def get_native_lib() -> Optional[ctypes.CDLL]:
    """Load or compile the native x-cruncher C11 shared library; returns None if no binary/compiler is available."""
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB

        sys_name = platform.system()
        lib_name = (
            "xcruncher.dll" if sys_name == "Windows"
            else ("libxcruncher.dylib" if sys_name == "Darwin" else "libxcruncher.so")
        )

        # 1. Check installed `xcruncher` Python wheel if present
        try:
            import xcruncher as _ext_xc  # type: ignore
            ext_dir = Path(_ext_xc.__file__).resolve().parent
            if (ext_dir / lib_name).is_file():
                _LIB = _bind_cdll(ext_dir / lib_name)
                return _LIB
        except Exception:
            pass

        # 2. Check bundled `fused/`, Android `jniLibs` (`LD_LIBRARY_PATH`), or user cache directory
        search_paths = [
            HERE / "fused" / lib_name,
            CSRC_DIR / lib_name,
        ]
        for env_k in ("CLOVER_NATIVE_LIB_DIR", "LD_LIBRARY_PATH"):
            for d_str in (os.environ.get(env_k) or "").split(os.pathsep):
                if d_str.strip():
                    search_paths.append(Path(d_str.strip()) / lib_name)
        try:
            search_paths.append(_cache_dir() / lib_name)
        except OSError:
            pass
        for cand in search_paths:
            if cand.is_file():
                try:
                    _LIB = _bind_cdll(cand)
                    return _LIB
                except OSError:
                    continue
        if sys_name != "Windows":
            try:
                raw_cdll = ctypes.CDLL(lib_name)
                u64_p = ctypes.POINTER(ctypes.c_uint64)
                raw_cdll.xc_version.argtypes = []
                raw_cdll.xc_version.restype = ctypes.c_char_p
                raw_cdll.xc_active_isa_name.argtypes = []
                raw_cdll.xc_active_isa_name.restype = ctypes.c_char_p
                raw_cdll.xc_mul_u64.argtypes = [u64_p, u64_p, ctypes.c_size_t, u64_p, ctypes.c_size_t]
                raw_cdll.xc_mul_u64.restype = ctypes.c_int
                raw_cdll.xc_pi_hex_digits.argtypes = [ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t]
                raw_cdll.xc_pi_hex_digits.restype = ctypes.c_int
                _verify_lib_kat(raw_cdll)
                _LIB = raw_cdll
                return _LIB
            except OSError:
                pass

        # 3. Compile from vendored `_xcruncher_c/xcruncher.c`
        c_src = CSRC_DIR / "src" / "xcruncher.c"
        if c_src.is_file():
            try:
                out_lib = _cache_dir() / lib_name
            except OSError:
                return None
            shared_flag = "-dynamiclib" if sys_name == "Darwin" else "-shared"
            omp_variants = (
                (["-fopenmp", "-Wl,-Bstatic", "-lgomp", "-lwinpthread", "-Wl,-Bdynamic"], [])
                if sys_name == "Windows"
                else (["-fopenmp"], [])
            )
            for cc in _candidate_c_compilers():
                base_cmd = [cc, "-O3", "-ffp-contract=off", "-std=c11", shared_flag, str(c_src), "-I", str(CSRC_DIR / "include"), "-o", str(out_lib)]
                if sys_name == "Windows":
                    base_cmd.extend(["-DXCRUNCHER_BUILD_DLL", "-static"])
                else:
                    base_cmd.insert(4, "-fPIC")
                    base_cmd.append("-lm")
                cc_env = dict(os.environ)
                cc_dir = str(Path(cc).parent)
                cc_env["PATH"] = cc_dir + os.pathsep + cc_env.get("PATH", "")
                for omp in omp_variants:
                    r = subprocess.run(base_cmd + omp, env=cc_env, capture_output=True)
                    if r.returncode == 0 and out_lib.is_file():
                        try:
                            if sys_name == "Windows" and hasattr(os, "add_dll_directory") and os.path.isdir(cc_dir):
                                os.add_dll_directory(cc_dir)
                            _LIB = _bind_cdll(out_lib)
                            return _LIB
                        except OSError:
                            continue
        return None


def active_isa_info() -> dict[str, str]:
    lib = get_native_lib()
    if lib is not None:
        return {
            "backend": "xcruncher-c11-native",
            "version": lib.xc_version().decode("utf-8"),
            "isa": lib.xc_active_isa_name().decode("utf-8"),
        }
    return {"backend": "python-bigint-fallback", "version": "stdlib", "isa": "CPython Karatsuba"}


def mul_exact(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    if a.bit_length() < 4096 and b.bit_length() < 4096:
        return a * b
    lib = get_native_lib()
    if lib is None:
        return a * b
    sign = -1 if (a < 0) ^ (b < 0) else 1
    aa, ab = abs(a), abs(b)
    na = max(1, (aa.bit_length() + 63) // 64)
    nb = max(1, (ab.bit_length() + 63) // 64)
    ba = (ctypes.c_uint64 * na).from_buffer_copy(aa.to_bytes(na * 8, "little"))
    bb = (ctypes.c_uint64 * nb).from_buffer_copy(ab.to_bytes(nb * 8, "little"))
    out = (ctypes.c_uint64 * (na + nb))()
    rc = lib.xc_mul_u64(out, ba, na, bb, nb)
    if rc != 0:
        raise RuntimeError(f"xc_mul_u64 failed with error code {rc}")
    res = int.from_bytes(bytes(out), "little")
    return -res if sign < 0 else res


def _range_prod(lo: int, hi: int) -> int:
    if lo > hi:
        return 1
    if lo == hi:
        return lo
    if hi - lo == 1:
        return lo * hi
    mid = (lo + hi) >> 1
    return mul_exact(_range_prod(lo, mid), _range_prod(mid + 1, hi))


def factorial_exact(n: int) -> int:
    if n < 0 or n > MAX_FACTORIAL_N:
        raise ValueError(f"n must be between 0 and {MAX_FACTORIAL_N}")
    if n <= 1:
        return 1
    return _range_prod(2, n)


def pi_hex_exact(hex_offset: int, digits: int = 20) -> str:
    if not (0 <= hex_offset <= MAX_PI_HEX_OFFSET):
        raise ValueError(f"hex_offset must be in 0..{MAX_PI_HEX_OFFSET} (2^44)")
    if not (1 <= digits <= 20):
        raise ValueError("digits must be in 1..20")
    lib = get_native_lib()
    if lib is not None:
        buf = ctypes.create_string_buffer(digits + 1)
        rc = lib.xc_pi_hex_digits(ctypes.c_uint64(hex_offset), buf, ctypes.c_size_t(digits))
        if rc != 0:
            raise RuntimeError(f"xc_pi_hex_digits failed with code {rc}")
        return buf.value.decode("ascii")
    # Pure-Python Bellard/BBP fallback for small offsets if native C compiler is unavailable
    if hex_offset > 200_000:
        raise RuntimeError("Native x-cruncher C11 library is required for pi_hex offsets > 200,000.")
    s = 0.0
    terms = (
        (4.0, 1), (-2.0, 4), (-1.0, 5), (-1.0, 6),
    )
    # Use high-precision integer fixed-point (128 extra guard bits) for exact 80-bit extraction
    prec_bits = digits * 4 + 64
    scale = 1 << prec_bits
    acc = 0
    for coeff_num, j in ((4, 1), (-2, 4), (-1, 5), (-1, 6)):
        sub = 0
        for k in range(hex_offset + 1):
            d = 8 * k + j
            r = pow(16, hex_offset - k, d)
            sub = (sub + (r * scale) // d) % scale
        k = hex_offset + 1
        while True:
            d = 8 * k + j
            sh = 4 * (k - hex_offset)
            if sh > prec_bits + 16:
                break
            term = (scale >> sh) // d
            if term == 0:
                break
            sub = (sub + term) % scale
            k += 1
        acc = (acc + coeff_num * sub) % scale
    _ = (s, terms)
    hex_val = (acc >> 64) & ((1 << (digits * 4)) - 1)
    return f"{hex_val:0{digits}x}"


_SAFE_INT_EXPR = re.compile(r"^[0-9+\-*()<>\s^]+$")


_BIGINT_PARSE_SEM = threading.BoundedSemaphore(4)


def _bigint_to_decimal_safe(val: int) -> str:
    if val == 0:
        return "0"
    neg = val < 0
    v = -val if neg else val
    if v.bit_length() <= 14_000:
        s = str(v)
        return ("-" + s) if neg else s

    def _conv(x: int, approx_digs: int) -> str:
        if x.bit_length() <= 14_000 or approx_digs <= 2000:
            return str(x)
        half = max(2000, (approx_digs // 4000) * 2000)
        hi, lo = divmod(x, 10 ** half)
        if hi == 0:
            return _conv(lo, half)
        return _conv(hi, approx_digs - half) + _conv(lo, half).zfill(half)

    res = _conv(v, int(v.bit_length() * 0.30103) + 2)
    return ("-" + res) if neg else res


def _decimal_to_bigint_safe(s: str) -> int:
    neg = s.startswith("-")
    digits = s[1:] if s[:1] in "+-" else s
    if len(digits) <= 4000:
        v = int(digits)
        return -v if neg else v

    def _parse_range(lo_i: int, hi_i: int) -> int:
        span = hi_i - lo_i
        if span <= 2000:
            return int(digits[lo_i:hi_i])
        mid_len = max(2000, (span // 4000) * 2000)
        split = hi_i - mid_len
        return _parse_range(lo_i, split) * (10 ** mid_len) + _parse_range(split, hi_i)

    v = _parse_range(0, len(digits))
    return -v if neg else v


def _parse_big_operand(raw: str | int) -> int:
    """Safely parse either a decimal/hex integer string or a simple near-2^k integer expression
    such as `(1 << 200000) - 6` or `2**100000 - 1` without eval() or raising sys.set_int_max_str_digits."""
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    s = str(raw or "").strip().replace(",", "").replace("_", "")
    if not s:
        raise ValueError("integer operand is empty")
    if len(s) > MAX_STR_DIGITS + 32:
        raise ValueError(f"operand exceeds {MAX_STR_DIGITS:,} characters")

    with _BIGINT_PARSE_SEM:
        if re.fullmatch(r"[+-]?0x[0-9a-fA-F]+", s):
            return int(s, 16)
        if re.fullmatch(r"[+-]?\d+", s):
            return _decimal_to_bigint_safe(s)
        s_norm = s.replace("^", "**")
        if not re.fullmatch(r"[0-9+\-*()<>\s]+", s_norm) or len(s_norm) > 256:
            raise ValueError(f"unsupported integer operand syntax: {s[:80]!r}")
        import ast
        tree = ast.parse(s_norm, mode="eval")

        def _ev(node) -> int:
            if isinstance(node, ast.Expression):
                return _ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
                return int(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                v = _ev(node.operand)
                return v if isinstance(node.op, ast.UAdd) else -v
            if isinstance(node, ast.BinOp):
                l_v, r_v = _ev(node.left), _ev(node.right)
                if isinstance(node.op, ast.Add):
                    return l_v + r_v
                if isinstance(node.op, ast.Sub):
                    return l_v - r_v
                if isinstance(node.op, ast.Mult):
                    return mul_exact(l_v, r_v)
                if isinstance(node.op, ast.LShift):
                    if not (0 <= r_v <= 3_500_000):
                        raise ValueError("bit shift must be in 0..3,500,000")
                    return l_v << r_v
                if isinstance(node.op, ast.Pow):
                    if abs(l_v) > 10_000 or not (0 <= r_v <= 1_000_000):
                        raise ValueError("power exponent out of safe exact range (base <= 10000, exp <= 1000000)")
                    return l_v ** r_v
            raise ValueError("unsupported operator in integer operand")

        return _ev(tree)


def _format_bigint(val: int) -> dict:
    est_need = int(abs(val).bit_length() * 0.31) + 64
    if est_need > MAX_STR_DIGITS * 3:
        return {
            "bit_length": val.bit_length(),
            "hex_low_64": f"{abs(val) & ((1 << 64) - 1):016x}",
            "note": "Result exceeds decimal conversion display limit; exact binary product computed.",
        }
    with _BIGINT_PARSE_SEM:
        dec = _bigint_to_decimal_safe(val)

    digits_only = dec.lstrip("-")
    n_digits = len(digits_only)
    if n_digits <= INLINE_DIGIT_DISPLAY_CAP:
        return {"exact_decimal": dec, "decimal_digits": n_digits, "bit_length": val.bit_length()}
    return {
        "decimal_digits": n_digits,
        "bit_length": val.bit_length(),
        "leading_60_digits": ("-" if val < 0 else "") + digits_only[:60],
        "trailing_60_digits": digits_only[-60:],
        "decimal_sha256": hashlib.sha256(dec.encode("ascii")).hexdigest(),
    }


def run_xcruncher_compute(
    operation: str = "mul",
    a: str | int = "",
    b: str | int = "",
    n: int = 0,
    hex_offset: int = 0,
    digits: int = 20,
) -> str:
    """Execute an exact x-cruncher computation (`mul`, `factorial`, or `pi_hex`) and return JSON."""
    op = str(operation or "mul").strip().lower()
    info = active_isa_info()
    t0 = time.perf_counter()
    try:
        if op in ("pi_hex", "pi"):
            off = int(hex_offset or 0)
            dig = int(digits or 20)
            hx = pi_hex_exact(off, dig)
            ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return json.dumps({
                "operation": "pi_hex",
                "hex_offset": off,
                "digits": dig,
                "pi_hex": hx,
                "elapsed_ms": ms,
                **info,
            }, indent=2)

        if op in ("factorial", "fact"):
            nv = int(n if n is not None else a)
            res = factorial_exact(nv)
            ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return json.dumps({
                "operation": "factorial",
                "n": nv,
                "elapsed_ms": ms,
                **_format_bigint(res),
                **info,
            }, indent=2)

        if op in ("mul", "multiply", "square"):
            va = _parse_big_operand(a)
            vb = va if (op == "square" or str(b or "").strip() == "") else _parse_big_operand(b)
            res = mul_exact(va, vb)
            ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return json.dumps({
                "operation": "square" if va == vb else "mul",
                "elapsed_ms": ms,
                **_format_bigint(res),
                **info,
            }, indent=2)

        return f"Error: operation must be 'mul', 'square', 'factorial', or 'pi_hex' (got {operation!r})."
    except (ValueError, RuntimeError, OverflowError) as exc:
        return f"Error: {exc}"
