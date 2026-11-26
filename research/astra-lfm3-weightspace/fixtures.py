import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import random
import unittest
import urllib.request


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def mcq(identifier, modality, question, options, correct_index, split="regression", **kwargs):
    if len(options) != 4 or len(set(map(str, options))) != 4:
        raise ValueError("Four distinct choices are required")
    choices = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", options))
    return {
        "id": identifier, "modality": modality, "split": split,
        "prompt": f"{question}\n{choices}\nAnswer with only the correct letter.",
        "options": options, "gold_letter": "ABCD"[correct_index],
        "max_new_tokens": 24, **kwargs,
    }


def arithmetic_cases(split, seed):
    rng = random.Random(seed)
    cases = []
    for index in range(32):
        kind = index % 4
        first = rng.randint(27, 98)
        second = rng.randint(13, 89)
        if kind == 0:
            first *= 11
            question, value = f"What is {first} plus {second}?", first + second
        elif kind == 1:
            first *= 13
            question, value = f"What is {first} minus {second}?", first - second
        elif kind == 2:
            question, value = f"What is {first} times {second}?", first * second
        else:
            question, value = f"What is {first * second} divided by {second}?", first
        correct_index = (index // 4 + kind) % 4
        options = [value - 1, value + 1, value + second]
        options.insert(correct_index, value)
        case = mcq(f"{split}-{index:03d}", "text", question, options, correct_index, split,
                   family=["addition", "subtraction", "multiplication", "division"][kind],
                   computed_answer=value)
        cases.append(case)
    return cases


def unrelated_cases():
    entries = [
        ("Which is the capital of Japan?", ["Tokyo", "Rome", "Lima", "Cairo"], 0),
        ("Which chemical symbol denotes oxygen?", ["Au", "O", "Na", "Fe"], 1),
        ("At standard pressure, water freezes at which Celsius temperature?", ["100", "50", "0", "-100"], 2),
        ("Which is a prime number?", ["9", "21", "27", "29"], 3),
        ("What is the plural of child?", ["children", "childs", "childes", "childrens"], 0),
        ("Which word is the opposite of increase?", ["extend", "decrease", "enlarge", "raise"], 1),
        ("All tulips are plants. This object is a tulip. What must be true?", ["It is a tree", "It is blue", "It is a plant", "It is not a plant"], 2),
        ("What comes next: 2, 4, 8, 16?", ["18", "20", "24", "32"], 3),
        ("In Python, what is len([4, 5, 6])?", ["3", "2", "4", "15"], 0),
        ("In Python, what is [10, 20, 30][1]?", ["10", "20", "30", "1"], 1),
        ("Which JSON value represents a Boolean truth value?", ["\"true\"", "None", "true", "True"], 2),
        ("Which data structure normally uses first-in-first-out order?", ["stack", "set", "tree", "queue"], 3),
        ("Which is the largest: 0.5, 0.05, 0.005, 0.0005?", ["0.5", "0.05", "0.005", "0.0005"], 0),
        ("Which unit measures electrical resistance?", ["volt", "ohm", "ampere", "watt"], 1),
        ("Which planet is closest to the Sun?", ["Mars", "Earth", "Mercury", "Jupiter"], 2),
        ("Which organ primarily pumps blood through the body?", ["lung", "kidney", "liver", "heart"], 3),
    ]
    return [mcq(f"unrelated-{index:03d}", "text", question, options, correct,
                family="unrelated") for index, (question, options, correct) in enumerate(entries)]


def visual_cases(directory):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    colors = {"red": (220, 30, 40), "green": (20, 160, 70),
              "blue": (30, 80, 230), "yellow": (240, 210, 30)}
    color_names = list(colors)
    cases = []
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 60) if font_path.exists() else ImageFont.load_default(size=60)
    for index in range(12):
        image = Image.new("RGB", (256, 256), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        if index < 4:
            draw.rectangle((55, 55, 201, 201), fill=colors[color_names[index]])
            question, options, correct = "What color is the square?", color_names, index
        elif index < 8:
            count = index - 3
            for marker in range(count):
                position = 28 + marker * 56
                draw.ellipse((position, 103, position + 35, 138), fill=colors["blue"])
            question, options, correct = "How many blue circles are visible?", ["one", "two", "three", "four"], count - 1
        elif index < 10:
            left = index == 8
            draw.rectangle((25 if left else 161, 90, 95 if left else 231, 160), fill=colors["red"])
            draw.ellipse((161 if left else 25, 90, 231 if left else 95, 160), fill=colors["blue"])
            question, options, correct = "Where is the red square relative to the blue circle?", ["left", "right", "above", "below"], 0 if left else 1
        else:
            printed = "472" if index == 10 else "836"
            draw.text((59, 85), printed, font=font, fill=(0, 0, 0))
            question, options, correct = "Which number is printed in the image?", ["427", "863", "472", "836"], index - 8
        filename = f"image-{index:03d}.png"
        image.save(directory / filename)
        cases.append(mcq(f"vision-{index:03d}", "vision", question, options, correct, image=filename))
    for index in range(8):
        first = index // 2
        second = (first + 1) % 4
        if index % 2:
            first, second = second, first
        frames = []
        for frame_index in range(8):
            color = color_names[first if frame_index < 4 else second]
            image = Image.new("RGB", (256, 256), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((48, 48, 208, 208), fill=colors[color])
            frames.append(np.asarray(image))
        filename = f"video-{index:03d}.npy"
        np.save(directory / filename, np.stack(frames), allow_pickle=False)
        cases.append(mcq(f"video-{index:03d}", "video", "Which color does the square have first in the video?",
                         color_names, first, video=filename, reversed_pair=index // 2))
    return cases


def speech_cases(directory, root):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    repo_id = "hf-internal-testing/librispeech_asr_dummy"
    url = f"https://huggingface.co/api/datasets/{repo_id}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        info = json.load(response)
    dataset_dir = directory / "source-librispeech"
    dataset_dir.mkdir(exist_ok=False)
    (dataset_dir / "upstream.json").write_text(json.dumps(info, indent=2))
    candidates = sorted([entry for entry in info["siblings"] if entry["rfilename"].endswith(".parquet")],
                        key=lambda entry: entry["rfilename"])
    if not candidates:
        raise RuntimeError("No public speech parquet found")
    rows = []
    sources = []
    for entry in candidates:
        path = Path(hf_hub_download(
            repo_id, filename=entry["rfilename"], repo_type="dataset", revision=info["sha"],
            local_dir=dataset_dir, cache_dir=root / "cache" / "datasets", token=False,
        ))
        digest = sha256(path)
        if entry.get("lfs") and digest != entry["lfs"]["sha256"]:
            raise RuntimeError("Speech dataset digest mismatch")
        sources.append({"file": entry["rfilename"], "sha256": digest})
        rows.extend(pq.read_table(path).to_pylist())
    cases = []
    selected = []
    for row in rows:
        audio_bytes = row["audio"]["bytes"]
        samples, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if samples.ndim != 1 or rate != 16_000 or not 2 <= len(samples) / rate <= 10:
            continue
        index = len(selected)
        filename = f"speech-input-{index:03d}.wav"
        sf.write(directory / filename, samples, rate, subtype="FLOAT")
        selected.append({"source_id": str(row.get("id", row.get("file"))), "file": filename,
                         "duration_s": len(samples) / rate, "transcript": row["text"]})
        if index < 6:
            cases.append({
                "id": f"asr-{index:03d}", "modality": "audio_asr", "split": "regression",
                "system": "Perform ASR in English.", "audio": filename,
                "gold_text": row["text"], "max_new_tokens": 96,
            })
        if len(selected) == 7:
            break
    if len(selected) != 7:
        raise RuntimeError("Insufficient bounded-duration utterances for six cases and a separate pilot")
    for index, waveform in enumerate((np.zeros(32_000, dtype=np.float32),
                                      (0.2 * np.sin(2 * np.pi * 440 * np.arange(32_000) / 16_000)).astype(np.float32))):
        filename = f"nonspeech-{index:03d}.wav"
        sf.write(directory / filename, waveform, 16_000, subtype="FLOAT")
        cases.append(mcq(f"audio-control-{index:03d}", "audio", "What can be heard in this recording?",
                         ["silence", "a steady tone", "a spoken sentence", "applause"], index, audio=filename))
    for index, text in enumerate(("Good morning.", "The meeting starts at nine.",
                                  "Please close the window.", "There are three green cups.")):
        cases.append({
            "id": f"tts-{index:03d}", "modality": "speech", "split": "regression",
            "system": "Perform TTS in American English using a female voice.",
            "prompt": text, "generation_mode": "audio", "max_new_tokens": 96,
        })
    provenance = {"repo_id": repo_id, "revision": info["sha"], "files": sources,
                  "selection": "first 7 rows with 16 kHz mono audio lasting 2-10 seconds; seventh is pilot only",
                  "selected": selected, "dataset_card": info.get("cardData", {})}
    (directory / "speech-provenance.json").write_text(json.dumps(provenance, indent=2))
    return cases, selected[-1]


def build(root):
    root = Path(root)
    directory = root / "fixtures"
    if (directory / "cases.json").exists():
        raise FileExistsError("Refusing to replace study fixtures")
    cases = arithmetic_cases("discovery", 12001) + arithmetic_cases("validation", 24002) + unrelated_cases()
    cases += visual_cases(directory)
    audio_cases, pilot_audio = speech_cases(directory, root)
    cases += audio_cases
    if len(cases) != 112:
        raise RuntimeError(f"Expected 112 cases, got {len(cases)}")
    pilot_cases = [
        {"id": "pilot-vision", "modality": "vision", "image": "image-000.png", "prompt": "Describe the shape and its color.", "max_new_tokens": 24},
        {"id": "pilot-video", "modality": "video", "video": "video-000.npy", "prompt": "Describe how the square changes color.", "max_new_tokens": 32},
        {"id": "pilot-asr", "modality": "audio_asr", "audio": pilot_audio["file"], "system": "Perform ASR in English.", "max_new_tokens": 96},
        {"id": "pilot-speech", "modality": "speech", "prompt": "Hello.", "system": "Perform TTS in American English using a female voice.", "generation_mode": "audio", "max_new_tokens": 96},
    ]
    with (directory / "cases.json").open("x") as output:
        json.dump(cases, output, indent=2)
    with (directory / "pilot-cases.json").open("x") as output:
        json.dump(pilot_cases, output, indent=2)
    fixture_files = [path for path in sorted(directory.iterdir()) if path.is_file()]
    manifest = {"case_count": len(cases), "cases_sha256": sha256(directory / "cases.json"),
                "files": [{"file": path.name, "sha256": sha256(path)} for path in fixture_files],
                "counts": dict(Counter(case["modality"] for case in cases)),
                "qualification": "Cases/material frozen; runtime/scorer/protocol seal occurs after API pilot and before baseline."}
    with (directory / "MATERIALS.json").open("x") as output:
        json.dump(manifest, output, indent=2)
    print("FIXTURES_READY " + json.dumps(manifest["counts"]), flush=True)


class FixtureTests(unittest.TestCase):
    def test_arithmetic_balanced(self):
        for split, seed in (("discovery", 12001), ("validation", 24002)):
            cases = arithmetic_cases(split, seed)
            self.assertEqual(len(cases), 32)
            self.assertEqual(Counter(case["gold_letter"] for case in cases), {letter: 8 for letter in "ABCD"})
            for family in ("addition", "subtraction", "multiplication", "division"):
                self.assertEqual(Counter(case["gold_letter"] for case in cases if case["family"] == family),
                                 {letter: 2 for letter in "ABCD"})
            for case in cases:
                self.assertEqual(case["options"]["ABCD".index(case["gold_letter"])], case["computed_answer"])

    def test_no_discovery_validation_duplicates(self):
        discovery = {case["prompt"] for case in arithmetic_cases("discovery", 12001)}
        validation = {case["prompt"] for case in arithmetic_cases("validation", 24002)}
        self.assertFalse(discovery & validation)

    def test_unrelated_cases_balanced(self):
        cases = unrelated_cases()
        self.assertEqual(len(cases), 16)
        self.assertEqual(Counter(case["gold_letter"] for case in cases), {letter: 4 for letter in "ABCD"})

    def test_duplicate_options_refused(self):
        with self.assertRaises(ValueError):
            mcq("bad", "text", "Question", ["one", "one", "two", "three"], 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=["fixtures.py"], verbosity=2)
    elif args.root:
        build(args.root)
    else:
        parser.error("--root or --self-test required")