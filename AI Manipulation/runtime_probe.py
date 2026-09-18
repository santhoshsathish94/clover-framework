"""
Clover Runtime Probe
Discovers what the current execution environment can actually provide.
"""

import json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

def check(name, fn):
    try:
        return {"capability": name, "available": True, "detail": fn()}
    except Exception as e:
        return {"capability": name, "available": False, "detail": f"{type(e).__name__}: {e}"}

def cpu():
    return {"logical_cpus": os.cpu_count(), "platform": platform.platform(),
            "python": sys.version.split()[0]}

def memory():
    p = Path("/proc/meminfo")
    if p.exists():
        vals = {}
        for line in p.read_text().splitlines()[:8]:
            k, v = line.split(":", 1)
            vals[k] = v.strip()
        return vals
    return "OS memory metrics unavailable"

def filesystem():
    root = Path(".").resolve()
    test = root / ".clover_runtime_probe"
    test.write_text(str(time.time()))
    value = test.read_text()
    test.unlink()
    return {"cwd": str(root), "write_read_delete": value}

def process():
    r = subprocess.run([sys.executable, "-c", "print('process-execution-ok')"],
                       capture_output=True, text=True, timeout=5)
    return {"returncode": r.returncode, "stdout": r.stdout.strip()}

def tools():
    return {"git": shutil.which("git") is not None,
            "curl": shutil.which("curl") is not None,
            "docker": shutil.which("docker") is not None}

results = [
    check("compute", cpu),
    check("memory_visibility", memory),
    check("persistent_filesystem", filesystem),
    check("child_process_execution", process),
    check("common_tool_binaries", tools),
]

print(json.dumps({
    "probe": "clover-runtime",
    "timestamp": time.time(),
    "results": results,
    "model_invocation": {
        "available_from_this_script": False,
        "reason": "No model-client credential/interface is assumed by the probe."
    },
    "continuation": {
        "available_inside_this_process": True,
        "automatic_future_model_turn": False,
        "reason": "A persistent supervisor can continue its own process, but a new model invocation requires an exposed model interface."
    }
}, indent=2))
