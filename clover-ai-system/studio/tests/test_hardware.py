"""Offline checks for hardware.py: the settings chosen for machines other than the laptop the Studio was built on, and
what is refused below the minimum (that laptop). The real probe is checked on the machine it runs on.
Usage: python tests/test_hardware.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "clover"))
import hardware  # noqa: E402

ok = 0


def check(cond, what):
    global ok
    if not cond:
        print("FAIL", what)
        raise SystemExit(1)
    ok += 1
    print("PASS", what)


def gpu(name, mb):
    return {"name": name, "memory_mb": mb, "free_mb": mb, "software": bool(hardware.SOFTWARE_GPU.search(name))}


def machine(ram, threads, cores, avx2=True, build=26200):
    return {"cpu": {"threads": threads, "cores": cores, "avx2": avx2}, "ram_gb": ram, "windows_build": build}


# the minimum: i5-1335U (10 cores, 12 threads), 15.7 GB visible, Iris Xe (shared memory, reported by Vulkan)
laptop = hardware.choose({"threads": 12, "cores": 10}, 15.7, [gpu("Intel(R) Iris(R) Xe Graphics", 8031)])
check(laptop == {"engine": "gpu", "gpu": "Intel(R) Iris(R) Xe Graphics", "ctx": 32768, "voice_threads": 10,
                 "csi_threads": 2}, f"the laptop: GPU, 32k context, 10 speech threads, 2 CSI threads ({laptop})")
check(hardware.meets_minimum(machine(15.7, 12, 10), 40) == [], "the laptop meets the minimum")

# a desktop with a discrete GPU next to the integrated one, 32 GB: the larger GPU, the longer context
desk = hardware.choose({"threads": 24, "cores": 16}, 31.8,
                       [gpu("Intel(R) UHD Graphics 770", 4096), gpu("NVIDIA GeForce RTX 4070", 12282)])
check(desk["engine"] == "gpu" and desk["gpu"] == "NVIDIA GeForce RTX 4070", f"desktop: the discrete GPU ({desk['gpu']})")
check(desk["ctx"] == 65536 and desk["voice_threads"] == 16 and desk["csi_threads"] == 4,
      f"desktop: 64k context, 16 speech threads, 4 CSI threads ({desk})")

# only a software renderer (a VM or a remote desktop) or a GPU too small for the model: the CPU
vm = hardware.choose({"threads": 8, "cores": 4}, 16.0, [gpu("llvmpipe (LLVM 15.0.7, 256 bits)", 16384)])
check(vm["engine"] == "cpu" and vm["gpu"] is None, f"software renderer only: CPU ({vm})")
small = hardware.choose({"threads": 8, "cores": 4}, 16.0, [gpu("AMD Radeon(TM) Graphics", 512)])
check(small["engine"] == "cpu", "a GPU with 512 MB: CPU")
none = hardware.choose({"threads": 16, "cores": None}, 64.0, [])
check(none == {"engine": "cpu", "gpu": None, "ctx": 65536, "voice_threads": 8, "csi_threads": 2},
      f"no GPU, cores unknown (half the threads): CPU, 64k context, 8 speech threads ({none})")
check(hardware.choose({"threads": 8, "cores": 4}, 16.0, [])["voice_threads"] == 4, "4 cores: speech keeps 4 threads")

# below the minimum: each shortfall named
low = hardware.meets_minimum(machine(7.8, 4, 2, avx2=False, build=18363), 10)
check(len(low) == 5, f"8 GB, 4 threads, no AVX2, Windows 10 1909, 10 GB free: five shortfalls ({low})")
check(any("memory 7.8 GB" in x for x in low) and any("4 CPU threads" in x for x in low)
      and any("AVX2" in x for x in low) and any("18363" in x for x in low) and any("10 GB free" in x for x in low),
      "each shortfall says what this machine has")
check(hardware.meets_minimum(machine(15.0, 8, 4), None) == [], "exactly the minimum (15 GB visible, 8 threads) passes")
check(hardware.meets_minimum(machine(14.9, 8, 4), None) != [], "just under 15 GB visible is refused")

print(f"{ok} checks passed")
