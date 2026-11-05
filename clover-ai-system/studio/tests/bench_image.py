"""Make one picture with a preset, through the same code the Studio uses (clover/image_gen.py).
python tests\\bench_image.py <preset> <result.json> [seed] [size] [prompt index]
Writes the result (or the error) to result.json; the picture goes to logs\\image-bench\\<preset>. Run by bench_image.ps1,
which measures memory while this runs."""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "clover"))
import image_gen  # noqa: E402

PROMPTS = ["a red and green parrot sitting on a branch in a rainforest, detailed photo, soft daylight",
           "a cosy reading corner with a green armchair and a lamp, watercolour illustration"]


def main():
    preset, res = sys.argv[1], Path(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 512
    prompt = PROMPTS[int(sys.argv[5]) if len(sys.argv) > 5 else 0]
    out = BASE / "logs" / "image-bench" / preset
    try:
        r = image_gen.ImageGen(BASE, preset).generate(prompt, out, size=size, seed=seed, time_limit_s=1200)
        r.update(ok=True, preset=preset, prompt=prompt, path=str(out / r["file"]))
    except RuntimeError as exc:
        r = {"ok": False, "preset": preset, "error": str(exc)}
    res.write_text(json.dumps(r), encoding="utf-8")


if __name__ == "__main__":
    main()
