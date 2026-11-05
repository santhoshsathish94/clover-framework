"""Windows process tweaks for long CPU jobs on the laptop."""
import os


def disable_power_throttling():
    """Windows can run a windowless background process in efficiency mode (EcoQoS: efficiency cores, low clocks),
    which made detached benchmark processes several times slower. Opt this process out. Returns True on success."""
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    class PPTS(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    st = PPTS(1, 0x1, 0x0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED controlled, state off
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st)))
    except Exception:  # never stop a job over this optimisation
        return False
