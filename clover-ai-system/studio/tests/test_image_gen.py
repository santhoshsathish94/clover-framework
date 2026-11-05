"""Offline checks for image_gen (no model): the arguments sd-cli gets, the saved file, failures and Stop.
A stand-in sd-cli (a Python script behind a .cmd) writes a PNG, fails or hangs.
Usage: python tests/test_image_gen.py"""
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
import image_gen  # noqa: E402

ok = 0


def check(cond, what):
    global ok
    if not cond:
        raise AssertionError(what)
    ok += 1
    print("PASS", what)


PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d49444154789c6360000000000200010be7f5a50000000049454e44ae426082")
with tempfile.TemporaryDirectory() as d:
    base = Path(d)
    sd = base / "tools" / "sd.cpp" / "vulkan"
    sd.mkdir(parents=True)
    fake = sd / "fake_sd.py"
    fake.write_text(
        "import sys, time, pathlib\n"
        "a = sys.argv[1:]\n"
        "pathlib.Path(sys.argv[0]).with_name('last_args.txt').write_text('\\n'.join(a), encoding='utf-8')\n"
        "p = a[a.index('-p') + 1]\n"
        "if 'FAIL' in p: print('ggml_vulkan: out of memory'); sys.exit(3)\n"
        "if 'HANG' in p: time.sleep(60)\n"
        f"pathlib.Path(a[a.index('-o') + 1]).write_bytes({PNG!r})\n", encoding="utf-8")
    real_run = image_gen.local_terminal.run
    image_gen.local_terminal.run = lambda args, **kw: real_run([sys.executable, str(fake)] + args[1:], **kw)
    (sd / "sd-cli.exe").write_bytes(b"")
    gen = image_gen.ImageGen(base, "sd15-lcm")
    check(not gen.available()[0] and "setup again" in gen.available()[1], "missing model files reported")
    for f in image_gen.presets(base)["sd15-lcm"]["files"]:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    check(gen.available()[0], "available once the files exist")
    media = base / "media"
    r = gen.generate("a parrot on a branch; rm -rf / & del *", media, size=500, seed=7)
    args = (sd / "last_args.txt").read_text(encoding="utf-8").splitlines()
    check(args[args.index("-p") + 1] == "a parrot on a branch; rm -rf / & del *<lora:lcm-lora-sdv1-5:1>",
          "prompt passed as one argument, unchanged, with the LCM suffix")
    check(args[args.index("-W") + 1] == "448" and args[args.index("-s") + 1] == "7", "size rounded to 64, seed passed")
    gen.generate("a big wallpaper", media, size=1024, seed=8)
    args = (sd / "last_args.txt").read_text(encoding="utf-8").splitlines()
    check(args[args.index("-W") + 1] == "768", "a 1024 request is capped at 768 (it needed 11 GB)")
    for f in media.glob("*.png"):
        if f.read_bytes() == PNG and f.name != r["file"]:
            f.unlink()
    check((media / r["file"]).read_bytes() == PNG and len(r["file"]) == 36, "picture saved under a content name")
    check(not list(media.glob(".gen-*")), "no temporary file left")
    try:
        gen.generate("FAIL please", media)
        check(False, "failure raises")
    except RuntimeError as exc:
        check("out of memory" in str(exc), "a failure says why")
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    t0 = time.time()
    try:
        gen.generate("HANG please", media, cancel=stop)
        check(False, "stop raises")
    except RuntimeError as exc:
        check("stopped by the user" in str(exc) and time.time() - t0 < 10, "Stop ends the generation")
    try:
        gen.generate("   ", media)
        check(False, "empty prompt raises")
    except RuntimeError:
        check(True, "an empty description is refused")
    check(len(list(media.iterdir())) == 1, "only the one picture is in the media folder")
    free = image_gen.free_mb()
    check(free is None or 0 < free < 1024 * 1024, "free memory is read")

    # the tool: too little memory frees the idle voice first, then refuses without starting sd-cli
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "clover"))
    import server  # noqa: E402

    class Args:
        pass

    class Eng:
        freed = 0

        def free_voice(self):
            Eng.freed += 1
            return True
    server.S.args, server.S.engine = Args(), Eng()
    server.S.args.base = str(base)
    server.S.settings = type("St", (), {"get": staticmethod(lambda: {"image_model": "sd15-lcm"})})()
    server.media_dir = lambda: media
    image_gen.free_mb = lambda: 1000
    made = []
    out = server.image_tool("s", None, made)["run"]({"prompt": "a cabin"})
    check(Eng.freed == 1 and "1.0 GB of memory free" in out and "Do not try again" in out and not made,
          "low memory: voice freed, then a plain refusal, sd-cli not started")
    image_gen.free_mb = lambda: 8000
    out = server.image_tool("s", None, made)["run"]({"prompt": "FAIL a cabin"})
    check("do not call generate_image again" in out, "a failed picture tells the model not to retry")
    schema = server.image_tool("s", None, made)["schema"]["function"]["parameters"]["properties"]
    check("size" not in schema, "the model cannot choose the picture size")

print(f"\n{ok} checks passed")
