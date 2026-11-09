import sys

sys.path.insert(0, "llama.cpp-dory/gguf-py")
import gguf  # noqa: E402

r = gguf.GGUFReader("dory-4b-bf16.gguf")
for k, f in r.fields.items():
    if k.startswith("dory.") or k in ("general.architecture", "general.file_type"):
        try:
            v = f.contents()
        except Exception:
            v = "?"
        if isinstance(v, list) and len(v) > 8:
            v = str(v[:8]) + f"... ({len(v)})"
        print(k, "=", v)
print("tensors", len(r.tensors))
