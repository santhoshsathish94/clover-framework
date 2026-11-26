#!/usr/bin/env python3
"""Tests for Collective Problem-Solving Learning in the Clover runtime:
`collective.py`, `collective_gateway.py`, `collective_evaluator.py`, `core_registry.py`, `core_sync.py`, and
`semantics.Runtime.collective`. Usage: python test_collective.py"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clover_cycle as C  # noqa: E402
import collective  # noqa: E402
import collective_evaluator  # noqa: E402
import collective_gateway  # noqa: E402
import core_registry  # noqa: E402
import core_sync  # noqa: E402
import personal  # noqa: E402
import semantics  # noqa: E402

try:
    from s23_on_device_clover import PHONE_OFFER, PHONE_TOOLS, phone_runtime  # noqa: E402
    import local_terminal  # noqa: E402
    _RISKY = local_terminal.risky
except ImportError:
    PHONE_TOOLS = {
        "web.search": ("web_search",), "web.page": ("web_search",), "python": ("python_interpreter",),
        "files.read": ("repo_read",), "files.list": ("repo_read",), "files.write": ("repo_write",),
        "shell": ("run_shell",), "status": ("run_shell",), "ssh": ("run_ssh",), "image": ("generate_image",),
        "memory.search": ("repo_search",), "remember": ("remember",),
    }
    PHONE_OFFER = ("web_search", "python_interpreter")
    _RISKY = lambda cmd: ("deletes files" if "del " in str(cmd).lower() or "rm " in str(cmd).lower() else None)

    def phone_runtime():
        root = os.environ.get("CLOVER_PPL_DIR", "").strip()
        c_root = os.environ.get("CLOVER_COLLECTIVE_DIR", "").strip()
        return semantics.Runtime(
            tools=PHONE_TOOLS,
            device="phone",
            risky=_RISKY,
            personal=personal.for_process(root, "phone", None),
            collective=core_sync.for_process(c_root, "phone", None),
        )

FAILED: list[str] = []
PASSED = [0]


def check(cond: bool, what: str) -> None:
    if cond:
        PASSED[0] += 1
        print("PASS", what)
    else:
        FAILED.append(what)
        print("FAIL", what)


class Chat:
    def __init__(self, rt, session):
        self.rt, self.session, self.history = rt, session, []

    def say(self, text):
        last = C.last_exchange(self.history) if self.history else None
        retry = C.retry_requested(text, last[0]["content"]) if last else None
        if self.rt.personal:
            self.rt.personal.before_turn(self.session, text, retry)
        f = semantics.interpret(text, self.rt, history=self.history)
        if retry:
            f.act = "retry"
        if self.rt.personal:
            self.rt.personal.after_turn(self.session, f, {"turn_n": len(self.history) // 2 + 1})
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": f"(answer to {text})"}]
        return f


# 1. Default off & opt-in consent checks
os.environ.pop("CLOVER_PPL_DIR", None)
os.environ.pop("CLOVER_COLLECTIVE_DIR", None)
os.environ.pop("CLOVER_COLLECTIVE", None)
check(phone_runtime().collective is None, "no collective dir known yet: collective is None")
plain = semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=_RISKY)
prompts = [
    "Corroborate the claim about heat pumps.",
    "Search the web for heat pumps",
    "Delete the old report files in C:\\tmp\\x",
    "What time is it?",
    "Explain photosynthesis.",
    "thanks",
]
check(all(semantics.interpret(p, plain) == semantics.interpret(p, phone_runtime()) for p in prompts),
      "without a Collective store every frame is unchanged")

with tempfile.TemporaryDirectory() as tmp0:
    os.environ["CLOVER_COLLECTIVE_DIR"] = str(Path(tmp0) / "coll-store")
    coll0 = phone_runtime().collective
    check(coll0 is not None and not coll0.consent_contribute and not coll0.consent_receive,
          "collective store defaults to opt-in (both consent_contribute and consent_receive are False)")
    os.environ["CLOVER_COLLECTIVE"] = "off"
    check(core_sync.for_process(str(Path(tmp0) / "other"), "phone", None) is None,
          "CLOVER_COLLECTIVE=off disables collective learning for the process")
    os.environ.pop("CLOVER_COLLECTIVE", None)
    os.environ.pop("CLOVER_COLLECTIVE_DIR", None)

# 2. Local candidate extraction, privacy boundary & symbolic unlearning ripple resistance
with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    # Create 3 independent Clover installations that all learn "corroborate <topic>" -> web.search + python
    gw = collective_gateway.CollectiveGateway()
    extractors = []
    persons = []

    for idx in range(3):
        p_store = tmp_path / f"person_{idx}"
        pers = personal.Personal(p_store, device="phone", filter_repo=[])
        rt_i = semantics.Runtime(tools=PHONE_TOOLS, device="phone", risky=_RISKY, personal=pers)
        chat = Chat(rt_i, f"sess-{idx}")

        # Embed private PII / secret tokens inside the raw training prompts
        chat.say(f"Corroborate the claim about fusion (ref secret_project_{idx}_acmecorp).")
        chat.say("No, search the web and use Python to check.")
        chat.say(f"Corroborate the figures on solar panels for alice{idx}@acmecorp.internal.")
        chat.say("No, look it up online and use Python to compute.")

        # Also add a personal preference and a statement (which must NEVER be exported)
        chat.say("Summarize Hamlet.")
        chat.say("Shorter, please.")
        chat.say("When I say ping, I mean search the web.")

        ext_optout = collective.LocalCollectiveExtractor(f"inst_{idx}", consent_contribute=False)
        check(ext_optout.export_candidates(pers) == [], f"inst_{idx}: opt-out exports 0 candidates")

        ext_optin = collective.LocalCollectiveExtractor(f"inst_{idx}", consent_contribute=True)
        preview = ext_optin.preview_candidates(pers)
        exported = ext_optin.export_candidates(pers)
        check(preview == exported and len(exported) == 1, f"inst_{idx}: preview == export and 1 candidate extracted")
        pkg = exported[0]
        check("tok=corroborate" in pkg["trigger"] and not any("secret" in f or "acmecorp" in f for f in pkg["trigger"]),
              f"inst_{idx}: trigger keeps 'tok=corroborate' and strips all private tokens ({pkg['trigger']})")
        check(pkg["category"] == "problem-solving" and "python" in pkg["target"] and "web.search" in pkg["target"],
              f"inst_{idx}: classified as multi-capability problem-solving strategy ({pkg['target']})")

        # Test Symbolic Unlearning Ripple Effect: delete the second experience and confirm the first experience's
        # private token (secret_project_X_acmecorp) is STILL never unmasked in the scrubbed learner!
        second_id = [e["id"] for e in pers.store.all_experiences() if "solar" in e["text"]][0]
        pers.retention([second_id], "stop")
        after_stop_leaks = [
            f for h in ext_optin._build_scrubbed_learner(pers.store.eligible()).hyps for f in h.trigger
            if "secret" in f or "acmecorp" in f
        ]
        check(after_stop_leaks == [], f"inst_{idx}: zero symbolic unlearning ripple when 2nd experience is stopped")
        pers.retention([second_id], "keep")

        for p in ext_optin.export_candidates(pers):
            gw.ingest(p, consent_verified=True)
        extractors.append(ext_optin)
        persons.append(pers)

    # 3. Sybil, personal habit, conflicting habit, and Byzantine poison tests at Gateway & Evaluator
    # (a) Sybil: 1 user submits 10 copies of a personal habit ("dig into" -> web.search)
    syb_ext = collective.LocalCollectiveExtractor("inst_sybil", consent_contribute=True)
    syb_pkg = {
        "package_id": "cand_sybil_dig",
        "schema_version": "1.0.0",
        "installation_id": syb_ext.pseudo_id,
        "category": "semantics",
        "trigger": ["ctx=phone", "tok=dig"],
        "target": ["web.page", "web.search"],
        "local_support_count": 5,
        "local_contra_count": 0,
        "local_precision": 1.0,
        "cross_context_generalised": False,
    }
    for _ in range(10):
        gw.ingest(syb_pkg, consent_verified=True)

    # (b) Conflicting habit: 3 users say "scrub" -> files.write, 3 users say "scrub" -> shell
    for k in range(3):
        gw.ingest({
            "package_id": f"cand_scrub_a_{k}", "schema_version": "1.0.0", "installation_id": f"inst_conf_a_{k}",
            "category": "semantics", "trigger": ["ctx=phone", "tok=scrub"], "target": ["files.write"],
            "local_support_count": 3, "local_contra_count": 0, "local_precision": 1.0, "cross_context_generalised": False,
        })
        gw.ingest({
            "package_id": f"cand_scrub_b_{k}", "schema_version": "1.0.0", "installation_id": f"inst_conf_b_{k}",
            "category": "semantics", "trigger": ["ctx=phone", "tok=scrub"], "target": ["shell"],
            "local_support_count": 3, "local_contra_count": 0, "local_precision": 1.0, "cross_context_generalised": False,
        })

    # (c) Byzantine poison: 3 colluding nodes try to hijack "check" -> files.write (regresses ordinary "Check the grammar...")
    for k in range(3):
        gw.ingest({
            "package_id": f"cand_byz_{k}", "schema_version": "1.0.0", "installation_id": f"inst_byz_{k}",
            "category": "semantics", "trigger": ["ctx=phone", "tok=check"], "target": ["files.write"],
            "local_support_count": 6, "local_contra_count": 0, "local_precision": 1.0, "cross_context_generalised": False,
        })

    # (d) Package with PII in trigger is rejected by Gateway
    ok_pii = gw.ingest({
        "package_id": "cand_pii", "schema_version": "1.0.0", "installation_id": "inst_bad_pii",
        "category": "semantics", "trigger": ["tok=alice@acmecorp.com"], "target": ["web.search"],
        "local_support_count": 3, "local_contra_count": 0, "local_precision": 1.0, "cross_context_generalised": False,
    })
    check(not ok_pii, "gateway rejects package with PII token in trigger")

    agg = gw.aggregate_candidates()
    syb_agg = next(c for c in agg if "tok=dig" in c["trigger"])
    check(syb_agg["independent_installations"] == 1 and syb_agg["total_submissions"] == 10,
          "gateway collapses 10 Sybil submissions from 1 installation to independent_installations == 1")

    # 4. Quarantine Evaluation (G1..G5)
    val_suite = {
        "strategy_probes": [
            {"context": "phone", "text": "Could you corroborate the claim about batteries?",
             "target": ["python", "web.page", "web.search"]},
            {"context": "phone", "text": "Please corroborate the figures on wind turbines.",
             "target": ["python", "web.page", "web.search"]},
            {"context": "phone", "text": "Corroborate the claim about ocean temperatures.",
             "target": ["python", "web.page", "web.search"]},
        ],
        "regression_suite": collective_evaluator.DEFAULT_SEMANTICS_REGRESSION_SUITE,
    }
    ev = collective_evaluator.CollectiveEvaluator(validation_suite=val_suite, min_installations=3, min_agreement=0.75)
    eval_res = ev.evaluate_all(agg)
    check(eval_res["promoted_count"] == 1 and "tok=corroborate" in eval_res["promoted"][0]["trigger"],
          "evaluator promotes only 'corroborate' and rejects Sybil ('dig'), conflicting ('scrub'), and Byzantine ('check')")

    # 5. CoreRegistry Git publication, Ed25519 public-key verification, and tamper / forged-key rejection
    reg_dir = tmp_path / "clover-collective"
    # Verify that CoreRegistry refuses to sign without a private signing key
    reg_unsigned = core_registry.CoreRegistry(reg_dir)
    no_key_refused = False
    try:
        reg_unsigned.publish_release(eval_res, note="Should fail without private key")
    except RuntimeError:
        no_key_refused = True
    check(no_key_refused, "CoreRegistry refuses to publish a release when no Ed25519 private signing key is provided")

    # Generate an ephemeral Ed25519 keypair for this test release
    priv_hex, pub_hex = core_registry.generate_ed25519_keypair()
    reg = core_registry.CoreRegistry(reg_dir, signing_key=priv_hex)
    manifest = reg.publish_release(eval_res, note="Release v1.0.1")
    check(manifest["rules_count"] == 1 and manifest["signature_alg"] == "Ed25519"
          and manifest["public_key"] == pub_hex and (reg_dir / "releases" / "manifest.json").is_file(),
          "CoreRegistry writes Ed25519-signed releases/manifest.json and commits to Git")
    check(reg.find_strings_anywhere_in_repo(["secret_project", "acmecorp", "alice0", priv_hex]) == [],
          "0 private tokens or private signing keys anywhere in clover-collective Git objects or working tree")

    # Verify that an attacker with a different Ed25519 keypair is rejected by CoreSync
    attacker_priv, _attacker_pub = core_registry.generate_ed25519_keypair()
    forged_rejected = False
    try:
        core_sync.CoreSync(verify_key=core_registry.ed25519_public_key(bytes.fromhex(attacker_priv)).hex()).verify_and_load(reg_dir)
    except core_sync.CoreSyncError:
        forged_rejected = True
    check(forged_rejected, "CoreSync rejects manifest when verified against a different Ed25519 public key")

    # 6. Recipient installation: CoreSync + semantics.Runtime.collective + Personal > Collective > General priority
    recip_coll_dir = tmp_path / "recip_collective"
    recip_pers_dir = tmp_path / "recip_personal"
    recip_coll = core_sync.Collective(recip_coll_dir, device="phone", verify_key=pub_hex)
    recip_pers = personal.Personal(recip_pers_dir, device="phone", filter_repo=[])
    rt_recip = semantics.Runtime(
        tools=PHONE_TOOLS, device="phone", risky=_RISKY,
        blocked=getattr(phone_runtime(), "blocked", None),
        personal=recip_pers, collective=recip_coll
    )

    # Before opting in to receive: collective rule is not applied
    f0 = semantics.interpret("Corroborate the claim about batteries.", rt_recip)
    check(f0.caps == [] and not f0.collective, "before opting in to receive: collective rule is not applied")

    # Opt in to receive and sync from registry
    recip_coll.set_consent(receive=True)
    sync_info = recip_coll.sync_from_registry(reg_dir)
    check(sync_info["synced"] and sync_info["rules_count"] == 1, "opted-in recipient verifies and loads 1 collective rule")

    f1 = semantics.interpret("Corroborate the claim about batteries.", rt_recip)
    check(set(f1.asked) == {"web_search", "python_interpreter"} and f1.collective.get("rule_id") is not None,
          "collective rule activates web_search + python_interpreter on an unseen problem")
    check("from shared core" in f1.understood() and "collective" in f1.layer() and "Collective problem-solving" in f1.guidance,
          "frame shows 'from shared core' and attaches problem-solving strategy guidance")
    check(set(semantics.route(f1, "ask", True, offer=PHONE_OFFER)[0]) == {"web_search", "python_interpreter"},
          "When I ask (default mode): collective rule's tools are routed")

    # Declined capability is respected even when a collective rule matches
    f_dec = semantics.interpret("Without Python, corroborate the claim about batteries.", rt_recip)
    check(f_dec.asked == ["web_search"] and "python" in f_dec.declined,
          "a capability the message declines ('Without Python') is never offered by a collective rule")

    # Personal habit overrides collective rule for that individual (Personal > Collective > General)
    r_chat = Chat(rt_recip, "recip-chat")
    r_chat.say("Corroborate the local README.md file.")
    r_chat.say("No, just read the file.")
    r_chat.say("Corroborate the local config.json file.")
    r_chat.say("No, read the file.")
    f_pers = r_chat.say("Corroborate the local notes.txt file.")
    check(f_pers.asked == ["repo_read"] and f_pers.personal.get("hypothesis") is not None and not f_pers.collective,
          "Personal > Collective: once this user validates a personal habit, their personal reading wins")

    # Safety rules (vault & risky commands) are never bypassed
    f_vault = semantics.interpret("Corroborate vault/github.json", rt_recip)
    check(f_vault.discernment == "refuse" and not f_vault.collective, "vault stays off-limits with collective on")

    # Consent withdrawal demotes rule when quorum drops below 3
    gw.remove_installation("inst_0")
    eval_after = ev.evaluate_all(gw.aggregate_candidates())
    m2 = reg.publish_release(eval_after, note="After inst_0 withdrew consent")
    check(m2["rules_count"] == 0, "withdrawing 1 of 3 contributors drops quorum to 2 < 3 and demotes the rule in v1.0.2")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print(f"{PASSED[0]} checks passed")
