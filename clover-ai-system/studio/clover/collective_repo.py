"""The `clover-collective` repository's automation (`collective_repo.py`), run by GitHub Actions in that repository.

Collective learning has no server. The shared core is the public GitHub repository `clover-collective`:
- every Clover app downloads its signed release (`releases/manifest.json`, then `releases/history/v<version>.json`)
  and checks the Ed25519 signature against the public key pinned in the app (`core_sync.py`), with no sign-in;
- a contribution is a public GitHub issue opened with the contributor's GitHub sign-in
  (`collective_gateway.request_body`). GitHub says who opened it, so no other identity check is needed;
- this script checks every open request, keeps the accepted packages per GitHub account in `contributions/`,
  aggregates them, runs the held-out evaluator, signs a new release when the promoted rules change, pushes, and then
  replies on each request and closes it. Issues that are not Clover requests are left alone.

Usage, in the repository's workflow: python engine/collective_repo.py run --root .
Environment: GITHUB_REPOSITORY, GITHUB_TOKEN, CLOVER_CPSL_PRIVATE_KEY (the release signing seed, a repository secret),
CLOVER_COLLECTIVE_MIN_INSTALLATIONS (default 3), CLOVER_COLLECTIVE_MIN_ACCOUNT_DAYS (default 30).
Only counts and versions are printed: the workflow's log is public.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import collective_evaluator
import collective_gateway
import core_registry
from collective import STRATEGY_GUIDANCE

OEWN_URL = "https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025.xml.gz"
OEWN_SHA256 = "9ca6d1dcb75f822fdd66617f7d9da48142ace38dd544d6ad5e2feca1674ad3fe"
CONTRIB_DIR = "contributions"
SEED_FILE = "seed/rules.json"
MAX_PER_CONTRIBUTOR = 200


class GitHubAPI:
    """The few GitHub REST calls the automation makes, with the workflow's own token."""

    def __init__(self, repo: str, token: str, opener=urllib.request.urlopen):
        self.repo, self.token, self.opener = repo, token, opener

    def _call(self, method: str, path: str, body: dict | None = None):
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "clover-collective-automation"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(f"https://api.github.com{path}", data=data, headers=headers, method=method)
        with self.opener(req, timeout=30) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else {}

    def open_issues(self) -> list[dict]:
        out, page = [], 1
        while True:
            items = self._call("GET", f"/repos/{self.repo}/issues?state=open&sort=created&direction=asc"
                                      f"&per_page=100&page={page}")
            out += [i for i in items if "pull_request" not in i]
            if len(items) < 100:
                return out
            page += 1

    def account_created(self, user_id: int) -> str:
        return str(self._call("GET", f"/user/{int(user_id)}").get("created_at") or "")

    def reply_and_close(self, number: int, text: str, reason: str):
        self._call("POST", f"/repos/{self.repo}/issues/{int(number)}/comments", {"body": text})
        self._call("PATCH", f"/repos/{self.repo}/issues/{int(number)}", {"state": "closed", "state_reason": reason})


def ensure_vocabulary(cache_dir: Path) -> Path:
    """The closed vocabulary the apps check with (Open English WordNet 2025, CC BY 4.0), pinned by SHA-256. Without
    it, packages with ordinary words would be refused, so a failed download stops the run."""
    if os.environ.get("CSI_DICT") and Path(os.environ["CSI_DICT"]).is_file():
        return Path(os.environ["CSI_DICT"])
    dest = Path(cache_dir) / "english-wordnet-2025.xml.gz"
    if not dest.is_file() or hashlib.sha256(dest.read_bytes()).hexdigest() != OEWN_SHA256:
        with urllib.request.urlopen(OEWN_URL, timeout=120) as resp:
            data = resp.read()
        if hashlib.sha256(data).hexdigest() != OEWN_SHA256:
            raise RuntimeError("Open English WordNet 2025 did not match its pinned SHA-256")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    os.environ["CSI_DICT"] = str(dest)
    return dest


def _save_json(path: Path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def load_contributions(root) -> dict[str, dict]:
    folder = Path(root) / CONTRIB_DIR
    out = {}
    for p in sorted(folder.glob("github-*.json")) if folder.is_dir() else ():
        doc = json.loads(p.read_text(encoding="utf-8"))
        out[doc["contributor"]] = doc
    return out


def apply_request(root, req: dict, author: dict, gw, *, now: float) -> dict:
    """One checked request from one GitHub account -> {"changed", "accepted", "rejected", "removed"}."""
    cid = f"github-{int(author['id'])}"
    path = Path(root) / CONTRIB_DIR / f"{cid}.json"
    if req["action"] == "withdraw":
        removed = 0
        if path.is_file():
            removed = len(json.loads(path.read_text(encoding="utf-8")).get("packages") or [])
            path.unlink()
        return {"changed": removed > 0, "accepted": [], "rejected": [], "removed": removed}
    doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"contributor": cid, "packages": []}
    have = {(tuple(p["trigger"]), tuple(p["target"])): p for p in doc["packages"]}
    accepted, rejected = [], []
    for raw in req["packages"]:
        if not isinstance(raw, dict):
            rejected.append((None, "not_a_dict"))
            continue
        if collective_gateway.FORBIDDEN_ENVELOPE_FIELDS & set(raw):
            rejected.append((raw.get("package_id"), "forbidden_personal_field"))
            continue
        pkg = dict(raw, installation_id=cid)  # the GitHub account, never the device's own id
        ok, why = gw.validate_package(pkg)
        if not ok:
            rejected.append((pkg.get("package_id"), why))
            continue
        key = (tuple(sorted(pkg["trigger"])), tuple(pkg["target"]))
        if key not in have and len(have) >= MAX_PER_CONTRIBUTOR:
            rejected.append((pkg.get("package_id"), "contributor_limit"))
            continue
        have[key] = dict(pkg, trigger=list(key[0]), target=list(key[1]))
        accepted.append(pkg.get("package_id"))
    packages = [have[k] for k in sorted(have)]
    changed = packages != doc["packages"]
    if changed:
        _save_json(path, {"contributor": cid, "login": str(author.get("login") or ""), "packages": packages,
                          "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))})
    return {"changed": changed, "accepted": accepted, "rejected": rejected, "removed": 0}


def _seed_reports(root, taken: set) -> list[dict]:
    """Rules carried over from an earlier shared core (`seed/rules.json`), kept until contributions promote a rule
    for the same trigger."""
    p = Path(root) / SEED_FILE
    if not p.is_file():
        return []
    out = []
    for r in json.loads(p.read_text(encoding="utf-8")).get("rules") or []:
        trig = sorted(r["trigger"])
        if tuple(trig) in taken:
            continue
        out.append({
            "candidate_id": "agg_" + str(r["rule_id"])[len("core_"):], "category": r["category"],
            "trigger": trig, "target": list(r["target"]),
            "strategy": STRATEGY_GUIDANCE.get(tuple(sorted(r["target"])), ""),
            "independent_installations": int(r["independent_installations"]), "supporting_installations": [],
            "total_submissions": 0, "cross_user_agreement": 1.0,
            "gates": {"G1_quorum": True, "G2_consistency": True, "G3_generalisation": True, "G4_cross_domain": True,
                      "G5_zero_regression": True},
            "metrics": {"matched_heldout_probes": 0, "heldout_accuracy": float(r["heldout_accuracy"]),
                        "baseline_b0_accuracy": 0.0, "heldout_lift": float(r["heldout_lift"]), "matched_domains": [],
                        "regressions": int(r["regressions"]), "personal_hijacks": 0},
            "promoted": True, "status": "promoted", "rejection_reasons": [], "seed": True,
        })
    return out


def _rule_keys(rules) -> list:
    return sorted((r["rule_id"], r["category"], tuple(r["trigger"]), tuple(r["target"]), r.get("strategy", ""),
                   int(r["independent_installations"]), float(r["heldout_accuracy"]), float(r["heldout_lift"]),
                   int(r["regressions"])) for r in rules)


def build_release(root, *, signing_key, min_installations: int, note: str) -> dict:
    """Aggregate every contributor's packages, evaluate them, add the seed rules, and sign a new release only when
    the promoted rules differ from the current release."""
    gw = collective_gateway.CollectiveGateway()
    for doc in load_contributions(root).values():
        for p in doc.get("packages") or []:
            gw.ingest(dict(p, installation_id=doc["contributor"]))
    ev = collective_evaluator.CollectiveEvaluator(min_installations=min_installations, min_agreement=0.75)
    res = ev.evaluate_all(gw.aggregate_candidates())
    seeds = _seed_reports(root, {tuple(sorted(r["trigger"])) for r in res["promoted"]})
    promoted = res["promoted"] + seeds
    summary = dict(res, promoted=promoted, promoted_count=len(promoted), all_reports=res["all_reports"] + seeds,
                   total_evaluated=res["total_evaluated"] + len(seeds))
    wanted = [{"rule_id": f"core_{r['candidate_id'][4:]}", "category": r["category"], "trigger": r["trigger"],
               "target": r["target"], "strategy": r.get("strategy", ""),
               "independent_installations": r["independent_installations"],
               "heldout_accuracy": r["metrics"]["heldout_accuracy"], "heldout_lift": r["metrics"]["heldout_lift"],
               "regressions": r["metrics"]["regressions"]} for r in promoted]
    mpath = Path(root) / "releases" / "manifest.json"
    current = json.loads(mpath.read_text(encoding="utf-8")) if mpath.is_file() else None
    if current is not None and _rule_keys(current.get("rules") or []) == _rule_keys(wanted):
        return {"published": False, "version": current.get("version"), "rules_count": len(current.get("rules") or [])}
    if not signing_key:
        raise RuntimeError("the release signing key (repository secret CLOVER_CPSL_PRIVATE_KEY) is not set")
    manifest = core_registry.CoreRegistry(root, signing_key=signing_key).publish_release(summary, note=note)
    return {"published": True, "version": manifest["version"], "rules_count": manifest["rules_count"]}


def _git(root, *args) -> str:
    r = subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {r.stderr.strip()[:300]}")
    return r.stdout


def _reply(login: str, res: dict, release: dict, min_installations: int) -> str:
    core = (f"Shared core: new release {release['version']} with {release['rules_count']} rules."
            if release["published"] else
            f"Shared core: unchanged at release {release['version']} with {release['rules_count']} rules.")
    if "removed" in res and res.get("withdraw"):
        return (f"Done, @{login}. Removed {res['removed']} of your patterns from future releases. Earlier issues stay "
                f"public on GitHub.\n\n{core}")
    lines = [f"Thanks, @{login}. The collective learning automation checked this request.", "",
             f"- Accepted: {len(res['accepted'])}"]
    if res["rejected"]:
        lines.append(f"- Not accepted: {len(res['rejected'])}")
        lines += [f"  - `{pid}`: {why}" for pid, why in res["rejected"][:20]]
    lines += ["", core, f"A pattern joins the shared core once at least {min_installations} different GitHub accounts "
                        "send it and it passes the held-out checks."]
    return "\n".join(lines)


def process(root, api, *, signing_key, now: float | None = None, min_installations: int = 3,
            min_account_days: int = 30, push: bool = True) -> dict:
    """Check every open request, publish a release if the promoted rules changed, commit and push, then reply on each
    request and close it. A request that cannot be checked right now stays open for the next run."""
    root = Path(root)
    now = time.time() if now is None else now
    gw = collective_gateway.CollectiveGateway()
    handled = []  # (issue number, reply text or result, close reason, login)
    for issue in api.open_issues():
        req = collective_gateway.parse_request(issue.get("body"))
        if req is None:
            continue
        num, user = int(issue["number"]), issue.get("user") or {}
        login = str(user.get("login") or "")
        if "error" in req:
            handled.append((num, f"@{login} This looks like a Clover collective learning request, but {req['error']}.",
                            "not_planned", login))
            continue
        if user.get("type") != "User" or not isinstance(user.get("id"), int):
            handled.append((num, "Only personal GitHub accounts can send collective learning requests.",
                            "not_planned", login))
            continue
        if req["action"] == "contribute" and min_account_days > 0:
            try:
                created = calendar.timegm(time.strptime(api.account_created(user["id"]), "%Y-%m-%dT%H:%M:%SZ"))
            except (OSError, ValueError, urllib.error.URLError) as exc:
                print(f"#{num}: the account check failed ({type(exc).__name__}); left open", file=sys.stderr)
                continue
            if (now - created) / 86400 < min_account_days:
                handled.append((num, f"Thanks, @{login}. Contributions are accepted from GitHub accounts at least "
                                     f"{min_account_days} days old, so new accounts cannot outvote real ones. Please "
                                     "send it again later.", "not_planned", login))
                continue
        res = apply_request(root, req, user, gw, now=now)
        res["withdraw"] = req["action"] == "withdraw"
        reason = "completed" if res["withdraw"] or res["accepted"] else "not_planned"
        handled.append((num, res, reason, login))
    numbers = ", ".join(f"#{h[0]}" for h in handled)
    release = build_release(root, signing_key=signing_key, min_installations=min_installations,
                            note=f"Requests {numbers}" if numbers else "Scheduled check")
    committed = False
    if _git(root, "status", "--porcelain").strip():
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", f"Collective learning requests {numbers}" if numbers else "Collective learning")
        committed = True
    if push and (committed or release["published"]):
        _git(root, "push", "-q", "origin", "HEAD")
    for num, res, reason, login in handled:
        text = res if isinstance(res, str) else _reply(login, res, release, min_installations)
        api.reply_and_close(num, text, reason)
    return {"handled": len(handled), "release": release, "committed": committed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The clover-collective repository's automation (GitHub Actions).")
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args(argv)
    ensure_vocabulary(Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()))  # never inside the repository
    api = GitHubAPI(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
    out = process(a.root, api, signing_key=os.environ.get("CLOVER_CPSL_PRIVATE_KEY", "").strip() or None,
                  min_installations=int(os.environ.get("CLOVER_COLLECTIVE_MIN_INSTALLATIONS") or 3),
                  min_account_days=int(os.environ.get("CLOVER_COLLECTIVE_MIN_ACCOUNT_DAYS") or 30),
                  push=not a.no_push)
    print(json.dumps({"requests_handled": out["handled"], "committed": out["committed"], **out["release"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
