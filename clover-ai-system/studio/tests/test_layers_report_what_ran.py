"""Step 2 check: C2/C3 layer info says "not run" when the C2/C3 parts are missing, and still names the modules when they
ran (the real install's precomputed parts). Usage: python tests/test_layers_report_what_ran.py"""
import sys
import tempfile
from pathlib import Path

CLOVER = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(CLOVER))
import lfm3_layers  # noqa: E402
import omni_c2_c3  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


ok = True
kit = lfm3_layers.LayerKit(Path(tempfile.mkdtemp()))
c1_state, _ = kit.c1_infer("The answer is 42.")
_, c2 = kit.c2_plan_state("What is six times seven?", c1_state=c1_state, offered_tools=["run_python"])
text, c3 = kit.ncm_render("The answer is 42.")
ok &= check("without C2/C3 parts: C2 says not run, names no module",
            c2.get("ran") is False and "module" not in c2 and "latent_steps" not in c2 and c2.get("reason"))
ok &= check("without C2/C3 parts: the tools JEPA ranked are still reported", c2.get("planned_tools") == ["run_python"])
ok &= check("without C2/C3 parts: C3 says not run, names no decoder, text unchanged",
            c3.get("ran") is False and "decoder" not in c3 and "bridges" not in c3 and text == "The answer is 42.")

install = Path.home() / "CloverAI"
if omni_c2_c3.missing_c2_c3_files(install):
    print("SKIP with C2/C3 parts: no complete install at", install)
else:
    real = lfm3_layers.LayerKit(install)
    s1, _ = real.c1_infer("The answer is 42.")
    s2, r2 = real.c2_plan_state("What is six times seven?", c1_state=s1, offered_tools=["run_python"])
    _, r3 = real.c3_decode(s2)
    ok &= check("with C2/C3 parts: C2 reports the modules that ran", "HybridLink" in str(r2.get("module"))
                and r2.get("ran") is not False)
    ok &= check("with C2/C3 parts: C3 reports the decoder that ran", "NextConceptTransformer" in str(r3.get("decoder"))
                and r3.get("ran") is not False)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
