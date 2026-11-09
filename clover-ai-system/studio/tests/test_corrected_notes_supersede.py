"""A user's correction supersedes the note it corrects (kept, marked superseded, linked by "corrects"), the memory block
says which note replaced it, and deleting the correction restores the old note. Before 2026-11-09 "Correction: my
manager is now Arun; ..." was kept next to "My manager is Priya." as two current notes and the model answered Priya.
Usage: python tests/test_corrected_notes_supersede.py"""
import sys
import tempfile
from pathlib import Path

CLOVER = Path(__file__).resolve().parent.parent / "clover"
sys.path.insert(0, str(CLOVER))
import stand_ins  # noqa: E402

stand_ins.install(CLOVER)
import lfm3_layers  # noqa: E402


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return bool(cond)


def kit_with(*notes, detect=True):
    base = Path(tempfile.mkdtemp(prefix="corrected-notes-"))
    kit = lfm3_layers.LayerKit(base, memory_root=base / "growth" / "knowledge")
    if not detect:
        kit._corrected_note = lambda text: None
    ids = [kit.remember(n, conversation_id="test") for n in notes]
    return kit, ids


def status(kit, eid):
    e = kit.store().entries[eid]
    return e.status, e.superseded_by


ok = True
OLD, NEW = "My manager is Priya.", "Correction: my manager is now Arun; Priya moved to another team."
kit, (a, b) = kit_with(OLD, NEW)
ok &= check("an explicit correction supersedes the note it corrects", status(kit, a) == ("superseded", b)
            and status(kit, b)[0] == "active")
block, _ = kit.memory_context("Who is my manager now?")
note_lines = [line for line in (block or "").splitlines() if line.startswith("[M")]
ok &= check("the memory block gives the correction, not the replaced note as a current one",
            any("Arun" in line for line in note_lines)
            and not any("My manager is Priya." in line and "(superseded" not in line for line in note_lines))
kit0, (a0, b0) = kit_with(OLD, NEW, detect=False)
ok &= check("liveness: without correction detection both notes stay current and both are retrieved",
            status(kit0, a0)[0] == "active" and status(kit0, b0)[0] == "active"
            and sum(line.startswith("[M") for line in (kit0.memory_context("Who is my manager now?")[0] or "").splitlines()) == 2)

kit, (a, b) = kit_with(OLD, "My manager is now Arun.")
ok &= check("'My X is now Y' replaces 'My X is Z'", status(kit, a) == ("superseded", b))
kit, (a, b) = kit_with(OLD, "My manager Priya is now on leave.")
ok &= check("a new fact about the same person is not a correction", status(kit, a)[0] == "active")
kit, (a, b) = kit_with("I like coffee.", "I also like tea.")
ok &= check("an addition corrects nothing", status(kit, a)[0] == "active")
kit, (a, b) = kit_with("I like coffee.", "Update: I like tea now, not coffee.")
ok &= check("an update that shares the subject supersedes", status(kit, a) == ("superseded", b))
kit, (a, b) = kit_with("My laptop is named atlas-7 and runs Windows 11.", "My laptop is now slow.")
ok &= check("a new state of a thing does not replace a longer note about it", status(kit, a)[0] == "active")
kit, (a, b) = kit_with(OLD, "My sister moved to Pune.")
ok &= check("a change cue about something else supersedes nothing", status(kit, a)[0] == "active")
kit, (a, b) = kit_with("I'm vegetarian and allergic to peanuts.", "I'm not vegetarian any more.")
ok &= check("a partial change does not hide the rest of a note (the allergy stays current)", status(kit, a)[0] == "active")

kit, (a, b) = kit_with(OLD, NEW)
kit.forget(b)
ok &= check("deleting the correction restores the old note", status(kit, a) == ("active", None)
            and kit.store().entries[a].revisions[-1].action == "restored")
block, _ = kit.memory_context("Who is my manager?")
ok &= check("after the restore the old note is current again", any(
    "My manager is Priya." in line and "(superseded" not in line for line in (block or "").splitlines() if line.startswith("[M")))
kit, (a, b, c) = kit_with(OLD, NEW, "My laptop is named atlas-7.")
kit.forget(c)
ok &= check("deleting an unrelated note restores nothing", status(kit, a) == ("superseded", b))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
