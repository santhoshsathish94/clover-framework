import json, urllib.request


def ask(sid, msg):
    data = json.dumps({"session_id": sid, "prompt": msg, "remember": True}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8765/api/chat", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    for m in [
        "can you find relevant research on recent AI development",
        "find it",
    ]:
        r = ask("sess_web_agent_test", m)
        print("USER      :", m)
        print("TOOLS     :", [(t["tool"], t["input"], t["result"][:160] + "...") for t in r.get("tool_calls", [])])
        print("K3 EXPERTS:", r.get("k3_experts"))
        print("REPLY     :\n" + r["reply"], "\n" + "=" * 65)


if __name__ == "__main__":
    main()
