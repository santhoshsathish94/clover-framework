"""Growth lessons through the real memory graph, on a temporary store (the user's memory is not touched):
stage -> listed as waiting and not retrieved -> keep -> active memory; stage -> discard -> gone."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clover"))
import lfm3_layers  # noqa: E402
import clover_cycle as C  # noqa: E402

base = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    kit = lfm3_layers.LayerKit(base, memory_root=Path(tmp) / "memory")
    text = C.lesson_text("How many grams heavier is a liter of milk than water? Round to the nearest 10 grams.",
                         ["Arithmetic: you wrote 1000 x 0.03 = 3, but 1000 x 0.03 = 30."], passed=True)
    lid = kit.stage_lesson(text, conversation_id="test", kind="verified")
    waiting = kit.lessons()
    print("staged:", lid, "| waiting:", [w["id"] for w in waiting])
    assert [w["id"] for w in waiting] == [lid]
    assert all(m["id"] != lid for m in kit.memories()), "a staged lesson must not be active memory yet"
    assert kit.memory_context("how much heavier is milk than water")[0] is None or lid not in str(kit.memory_context("how much heavier is milk than water"))
    kept = kit.keep_lesson(lid)
    print("kept:", kept, "| waiting now:", len(kit.lessons()), "| in memory:", any(m["id"] == kept for m in kit.memories()))
    assert not kit.lessons() and any(m["id"] == kept for m in kit.memories())
    lid2 = kit.stage_lesson(C.lesson_text("Summarize in one sentence.", [], "you missed who scored"), "test", kind="user")
    assert [w["id"] for w in kit.lessons()] == [lid2]
    assert kit.discard_lesson(lid2) and not kit.lessons()
    print("discarded:", lid2, "| waiting now:", len(kit.lessons()))
print("lesson store: all checks passed")
