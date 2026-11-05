"""CSI in the Studio (csi_child.py): the package is the frozen framework copy byte for byte; missing parts are reported;
with everything installed (the laptop after setup) the child service starts, answers, and semantics reads a declined
tool through it. Usage: python tests/test_csi_child.py   (the live part runs where spaCy and data\\csi are installed)"""
import filecmp
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parent / "clover"
sys.path.insert(0, str(CLOVER))
import csi_child  # noqa: E402

ok = 0


def check(cond, what):
    global ok
    if not cond:
        print("FAIL", what)
        raise SystemExit(1)
    ok += 1
    print("PASS", what)


proto = HERE.parents[3] / "clover-framework" / "research" / "semantic-interface" / "prototype"
if proto.is_dir():
    names = sorted(p.name for p in (proto / "csi").glob("*.py"))
    same = [n for n in names if filecmp.cmp(proto / "csi" / n, CLOVER / "csi" / n, shallow=False)]
    check(same == names and filecmp.cmp(proto / "csi_service.py", CLOVER / "csi_service.py", shallow=False),
          f"the Studio's CSI is the frozen framework copy ({len(names)} files + service)")
else:
    print("SKIP byte comparison: clover-framework not beside this tree")

with tempfile.TemporaryDirectory() as tmp:
    need = csi_child.missing(tmp)
    check("morph-verb-v1.1/morphology.json" in need and csi_child.OEWN in need, f"an empty install reports what is missing: {need}")
    svc = csi_child.CSIService(tmp)
    svc.start()
    check(svc.state == "missing" and svc.url == "" and "setup" in svc.detail, "missing parts: not started, Clover works without")

base = Path(os.environ.get("CLOVER_BASE") or HERE.parent)
if csi_child.missing(base):
    print(f"SKIP live service: {base} lacks {csi_child.missing(base)}")
else:
    import semantics
    svc = csi_child.CSIService(base)
    t0 = time.time()
    svc.start()
    while svc.state == "starting" and time.time() - t0 < 300:
        time.sleep(1)
    check(svc.state == "ready", f"the child service starts ({svc.detail})")
    read = semantics.csi_reader(svc.url, timeout=30)
    a = read("Without searching the web, explain what photosynthesis is.")
    check(a is not None and a["readings"][0]["propositions"], "it answers with CSI's reading")
    check("web.search" in semantics.declined_caps("Without searching the web, explain what photosynthesis is.", a)
          or "web.search" in semantics.declined_caps("Don't search the web, just explain photosynthesis.",
                                                     read("Don't search the web, just explain photosynthesis.")),
          "a declined web search is read as declined")
    t1 = time.time()
    for _ in range(5):
        read("Search the web for heat pumps")
    print(f"     mean round trip {(time.time() - t1) / 5 * 1000:.0f} ms, start {svc.detail}")
    svc.stop()
    check(svc.proc is None and svc.state == "off", "stopped with the Studio")
print(f"{ok} checks passed")
