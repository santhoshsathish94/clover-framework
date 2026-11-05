"""Which tools the routing gives everyday messages (the real-world regression's batch 2).
Usage: python tests/route_probe.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clover"))
import lfm3_native as N  # noqa: E402

for m in ("Remember that I'm vegetarian and allergic to peanuts.", "Remember that my favourite colour is teal.",
          "Write a short, polite email to my landlord saying the kitchen tap is leaking.",
          "Convert 72 degrees Fahrenheit to Celsius.", "Suggest a quick dinner I can cook tonight."):
    print(f"{m!r:90} -> {N.asked_tools(m)}")
