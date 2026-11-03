"""Personal Process Learning 0.1 - experiment runner (k3).

  python run_ppl.py general --out DIR            train the shared general interpreter from population data
  python run_ppl.py dev --out DIR                tune on development persons (dev habit pool)
  python run_ppl.py test --out DIR --params P    the pre-registered test on fresh persons (test habit pool)

CSI is imported from CSI_PROTO (default /opt/clover-k3/meaning-core/csi/proto) with the frozen environment."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("CSI_PROTO", "/opt/clover-k3/meaning-core/csi/proto"))

from ppl import evaluate as E  # noqa: E402
from ppl import features as F  # noqa: E402
from ppl import general as G  # noqa: E402
from ppl import personas as P  # noqa: E402
from ppl.store import PersonalStore  # noqa: E402

DEV_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006]
GRID = dict(C=(0.0, 0.3, 0.5), K=(1, 2, 3), P=(0.7, 0.9), apply_provisional=(True, False),
            tau=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
PROBES = ["Please forward the draft to Raj.", "Could you summarize the proposal?", "The tests on utils.py are red.",
          "Move the standup to 3pm.", "I don't need sources.", "Tidy up the budget memo, please.",
          "Can you ptal api.ts?", "the proposal -> PDF", "She did not send the invoice email.", "Ship it."]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p).encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def csi_integrity():
    """Hashes of the CSI code and frozen artefacts, and of CSI's readings of fixed probe sentences."""
    proto = Path(os.environ.get("CSI_PROTO", "/opt/clover-k3/meaning-core/csi/proto"))
    code = sorted(str(p) for p in (proto / "csi").rglob("*.py"))
    frozen = []
    for var in ("CSI_MORPH",):
        d = os.environ.get(var)
        if d and Path(d).exists():
            frozen += sorted(str(p) for p in Path(d).rglob("*") if p.is_file())
    from csi import analyze
    readings = hashlib.sha256(json.dumps([analyze(s) for s in PROBES], sort_keys=True, default=str).encode()).hexdigest()
    return {"code": sha_files(code), "frozen": sha_files(frozen) if frozen else None, "readings": readings,
            "n_code": len(code), "n_frozen": len(frozen)}


def _read_chunk(args):
    texts, path = args
    sys.path.insert(0, os.environ.get("CSI_PROTO", "/opt/clover-k3/meaning-core/csi/proto"))
    r = F.CSIReader(path)
    for t in texts:
        r.read(t)
    return path


def stage_general(out, workers):
    rows = G.population_messages()
    texts = sorted({t for t, _, _ in rows})
    log(f"population: {len(rows)} messages, {len(texts)} distinct")
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    chunks = [(texts[k::workers], str(cache / f"general-{k}.jsonl")) for k in range(workers)]
    with mp.get_context("spawn").Pool(workers) as pool:
        for p in pool.imap_unordered(_read_chunk, chunks):
            log(f"read {p}")
    merged = cache / "general.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for _, p in chunks:
            f.write(Path(p).read_text(encoding="utf-8"))
    reader = F.CSIReader(str(merged))
    model = G.GeneralInterpreter.train(rows, reader)
    model.save(str(out / "general.npz"))
    # held-out check on a fresh population sample (different seed): general accuracy
    test = G.population_messages(seed=99, per_phrase=4)
    rtest = F.CSIReader(str(cache / "general-heldout.jsonl"))
    ok = sum(model.scores(F.features(t, c, rtest.read(t))[0])[0][0] == lab for t, c, lab in test)
    info = {"rows": len(rows), "labels": len(model.labels), "vocab": len(model.vocab), "digest": model.digest(),
            "heldout_acc": ok / len(test), "heldout_n": len(test)}
    (out / "general.json").write_text(json.dumps(info, indent=1))
    log(f"general interpreter: {info}")


def _person_job(args):
    kind, spec, out, params_sets, extra = args
    sys.path.insert(0, os.environ.get("CSI_PROTO", "/opt/clover-k3/meaning-core/csi/proto"))
    general = G.GeneralInterpreter.load(extra["general"])
    reader = F.CSIReader(str(Path(out) / "cache" / f"{spec['pool']}-{spec['seed']}.jsonl"))
    results = {}
    for name, arm, params in params_sets:
        store = None
        if kind == "test" and arm == "PPL":
            store = PersonalStore(Path(out) / "stores", f"p{spec['seed']}", extra["filter_repo"])
        asst, train_log, recs = E.run(spec, arm, general, reader, params, store)
        res = {"records": recs, "train": train_log}
        if store is not None:
            store.save_knowledge(asst.learner.knowledge(), "knowledge after training")
            res["knowledge"] = asst.learner.knowledge()
            tr, ev = E.person_items(spec)
            res["retention"] = E.retention_test(asst, train_log, spec, ev, general, reader, params, store,
                                                str(Path(out) / "copies" / f"p{spec['seed']}"))
        elif asst.learner is not None and kind == "test":
            res["knowledge"] = asst.learner.knowledge()
        results[name] = res
    return spec["seed"], results


def configs_dev():
    out = []
    for C in GRID["C"]:
        base = {"C": C}
        out.append((f"B0|C={C}", "B0", base))
        out.append((f"B1|C={C}", "B1", base))
        for tau in GRID["tau"]:
            out.append((f"B2|C={C}|tau={tau}", "B2", dict(base, tau=tau)))
        for K in GRID["K"]:
            for Pv in GRID["P"]:
                for prov in GRID["apply_provisional"]:
                    lp = {"K": K, "P": Pv, "apply_provisional": prov}
                    out.append((f"PPL|C={C}|K={K}|P={Pv}|prov={prov}", "PPL", dict(base, learner=lp)))
    return out


def stage_dev(out, workers, general_path):
    specs = [P.sample_person(s, "dev") for s in DEV_SEEDS]
    cfgs = configs_dev()
    log(f"dev: {len(specs)} persons x {len(cfgs)} configurations")
    jobs = [("dev", spec, str(out), cfgs, {"general": general_path}) for spec in specs]
    allres = {}
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        for seed, res in pool.imap_unordered(_person_job, jobs):
            allres[seed] = res
            log(f"dev person {seed} done")
    table = {}
    for name, _, _ in cfgs:
        recs = [r for s in DEV_SEEDS for r in allres[s][name]["records"]]
        table[name] = E.summarise(recs)

    def acc(n):
        return table[n]["all"]["acc"]

    def best(prefix, ok=lambda n: True):
        names = [n for n, _, _ in cfgs if n.split("|")[0] == prefix and ok(n)]
        return max(names, key=lambda n: (round(acc(n), 6), -names.index(n)))

    choice = {"B0": best("B0"), "B1": best("B1"), "B2": best("B2"),
              "PPL": best("PPL", lambda n: (table[n]["control"]["overgen"] or 0) <= 0.10
                          and (table[n]["oneoff_probe"]["overgen"] or 0) <= 0.10)}
    params = {}
    for arm, name in choice.items():
        params[arm] = dict(next(p for n, _, p in cfgs if n == name))
    for abl in ("PPL-surface", "PPL-noval", "PPL-noctx"):
        params[abl] = dict(params["PPL"])
    # ablations on dev with the chosen PPL parameters, for information
    abl_cfgs = [(a, a, params[a]) for a in ("PPL-surface", "PPL-noval", "PPL-noctx")]
    jobs = [("dev", spec, str(out), abl_cfgs, {"general": general_path}) for spec in specs]
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        for seed, res in pool.imap_unordered(_person_job, jobs):
            for k, v in res.items():
                allres[seed][k] = v
    for a in ("PPL-surface", "PPL-noval", "PPL-noctx"):
        table[a] = E.summarise([r for s in DEV_SEEDS for r in allres[s][a]["records"]])
    (out / "dev_table.json").write_text(json.dumps(table, indent=1))
    (out / "params.json").write_text(json.dumps({"choice": choice, "params": params}, indent=1))
    with open(out / "dev_records.jsonl", "w", encoding="utf-8") as f:
        for s in DEV_SEEDS:
            for name, res in allres[s].items():
                for r in res["records"]:
                    f.write(json.dumps(dict(r, person=s, config=name), ensure_ascii=False) + "\n")
    log("chosen: " + json.dumps(choice))
    ppl_names = [n for n, _, _ in cfgs if n.startswith("PPL|")]
    for name in list(choice.values()) + ["PPL-surface", "PPL-noval", "PPL-noctx"] + ppl_names:
        t = table[name]
        log(f"{name:45s} all {t['all']['acc']:.3f} habit {t['habit']['acc']:.3f} control {t['control']['acc']:.3f} "
            f"overgen {t['control']['overgen']:.3f} oneoff {t['oneoff_probe']['acc']:.3f}/{t['oneoff_probe']['overgen']:.3f} "
            f"generic {t['generic']['acc']:.3f} routine {t['routine']['acc']:.3f} "
            f"open {t['open_query']['acc']:.3f} pres {t['presentation']['adherence']:.3f}")


def stage_test(out, workers, general_path, params_path, seeds, filter_repo, pool_name="test"):
    params = json.loads(Path(params_path).read_text())["params"]
    specs = [P.sample_person(s, pool_name) for s in seeds]
    general = G.GeneralInterpreter.load(general_path)
    before = {"general": general.digest(), "csi": csi_integrity()}
    arms = ["B0", "B1", "B2", "PPL", "PPL-surface", "PPL-noval", "PPL-noctx"]
    sens = [(f"B2|tau={t}", "B2", dict(params["B2"], tau=t)) for t in GRID["tau"]]
    cfgs = [(a, a, params[a]) for a in arms] + sens
    jobs = [("test", spec, str(out), cfgs, {"general": general_path, "filter_repo": filter_repo}) for spec in specs]
    allres = {}
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        for seed, res in pool.imap_unordered(_person_job, jobs):
            allres[seed] = res
            log(f"test person {seed} done")
    log("leakage run (all persons interleaved in one process)")
    reader = F.CSIReader(str(out / "cache" / "leakage.jsonl"),
                         preload=[str(out / "cache" / f"{pool_name}-{s}.jsonl") for s in seeds])
    inter = E.interleaved_eval(specs, general, reader, params["PPL"])
    leak = {}
    for spec in specs:
        alone = [r["got"] for r in allres[spec["seed"]]["PPL"]["records"]]
        leak[spec["seed"]] = sum(a != b for a, b in zip(alone, inter[spec["seed"]]))
    after = {"general": G.GeneralInterpreter.load(general_path).digest(), "csi": csi_integrity()}
    result = {"seeds": seeds, "params": params, "before": before, "after": after, "leakage": leak,
              "summary": {}, "per_person": {}, "retention": {}}
    for a in arms:
        recs = [r for s in seeds for r in allres[s][a]["records"]]
        result["summary"][a] = E.summarise(recs)
        result["per_person"][a] = {k: E.per_person({s: allres[s][a]["records"] for s in seeds}, k)
                                   for k in ("habit", "all", "control", "generic")}
    for s in seeds:
        result["retention"][s] = allres[s]["PPL"]["retention"]
    # sensitivity (not a criterion): conversation memory at every threshold, including those best on this test
    result["sensitivity_B2"] = {}
    for name, _, _ in sens:
        sm = E.summarise([r for s in seeds for r in allres[s][name]["records"]])
        result["sensitivity_B2"][name] = {"all": sm["all"]["acc"], "habit": sm["habit"]["acc"],
                                          "oneoff_overapplied": sm["oneoff_probe"]["overgen"],
                                          "control_overapplied": sm["control"]["overgen"]}
    with open(out / "test_records.jsonl", "w", encoding="utf-8") as f:
        for s in seeds:
            for a in arms:
                for r in allres[s][a]["records"]:
                    f.write(json.dumps(dict(r, person=s, arm=a), ensure_ascii=False) + "\n")
    with open(out / "test_knowledge.jsonl", "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps({"person": s, "spec": P.sample_person(s, pool_name),
                                "knowledge": allres[s]["PPL"].get("knowledge")}, ensure_ascii=False) + "\n")
    result["criteria"] = criteria(result, seeds)
    (out / "test_result.json").write_text(json.dumps(result, indent=1, default=str))
    for k, v in result["criteria"].items():
        log(f"{k}: {v}")


def criteria(res, seeds):
    S, pp = res["summary"], res["per_person"]

    def diff(a, b, kind):
        return [pp[a][kind][s] - pp[b][kind][s] for s in seeds]

    hab = {a: S[a]["habit"]["acc"] for a in S}
    allv = {a: S[a]["all"]["acc"] for a in S}
    c = {}
    p1 = E.sign_flip(diff("PPL", "B0", "habit"))
    p2 = E.sign_flip(diff("PPL", "B2", "habit"))
    c["L1 habits: PPL >= B0+20 and >= B2+10, p<0.01"] = {
        "PPL": hab["PPL"], "B0": hab["B0"], "B2": hab["B2"], "p_vs_B0": p1, "p_vs_B2": p2,
        "pass": hab["PPL"] >= hab["B0"] + 0.20 and hab["PPL"] >= hab["B2"] + 0.10 and p1 < 0.01 and p2 < 0.01}
    c["L2 all fresh items: PPL >= B0+10 and >= B2+5"] = {
        "PPL": allv["PPL"], "B0": allv["B0"], "B2": allv["B2"],
        "pass": allv["PPL"] >= allv["B0"] + 0.10 and allv["PPL"] >= allv["B2"] + 0.05}
    og = {a: S[a]["control"]["overgen"] for a in S}
    oo = {a: S[a]["oneoff_probe"]["overgen"] for a in S}
    c["L3 over-application <= 10% and <= B2, on controls and on one-off probes"] = {
        "controls": {"PPL": og["PPL"], "B2": og["B2"]}, "oneoff_probes": {"PPL": oo["PPL"], "B2": oo["B2"]},
        "pass": og["PPL"] <= 0.10 and og["PPL"] <= og["B2"] + 1e-9 and oo["PPL"] <= 0.10
        and oo["PPL"] <= oo["B2"] + 1e-9}
    gen = {a: S[a]["generic"]["acc"] for a in S}
    c["L4 generic: PPL >= B0-2"] = {"PPL": gen["PPL"], "B0": gen["B0"], "pass": gen["PPL"] >= gen["B0"] - 0.02}
    pres = {a: S[a]["presentation"]["adherence"] for a in S}
    c["L5 presentation: PPL >= B0+25 and >= B1+10"] = {
        "PPL": pres["PPL"], "B0": pres["B0"], "B1": pres["B1"],
        "pass": pres["PPL"] >= pres["B0"] + 0.25 and pres["PPL"] >= pres["B1"] + 0.10}
    c["L6 routines >= 80% and open-item F1 >= 0.8"] = {
        "routine": S["PPL"]["routine"]["acc"], "open_f1": S["PPL"]["open_f1"],
        "pass": S["PPL"]["routine"]["acc"] >= 0.8 and S["PPL"]["open_f1"] >= 0.8}
    c["L7 no leakage; CSI and general interpreter unchanged"] = {
        "leakage_diffs": sum(res["leakage"].values()), "general_same": res["before"]["general"] == res["after"]["general"],
        "csi_same": res["before"]["csi"] == res["after"]["csi"],
        "pass": sum(res["leakage"].values()) == 0 and res["before"]["general"] == res["after"]["general"]
        and res["before"]["csi"] == res["after"]["csi"]}
    ret = res["retention"]
    sd = [s for s in seeds if not ret[s]["vacuous_delete"]]
    ss = [s for s in seeds if not ret[s]["vacuous_stop"]]
    d_rev = E.mean([ret[s]["delete"]["reverted_to_general"] for s in sd])
    s_rev = E.mean([ret[s]["stop"]["reverted_to_general"] for s in ss])
    c["L8 retention: deletion leaves nothing, behaviour reverts; stop keeps records, behaviour reverts"] = {
        "persons_with_dependent_knowledge": {"delete": len(sd), "stop": len(ss), "of": len(seeds)},
        "delete_references_left": sum(ret[s]["delete"]["references_left"] for s in seeds),
        "delete_found_in_store": sum(ret[s]["delete"]["found_in_store"] for s in seeds),
        "delete_found_in_copy": sum(ret[s]["delete"]["found_in_copy"] for s in seeds),
        "delete_reverted": d_rev, "stop_reverted": s_rev,
        "stop_records_kept": E.mean([ret[s]["stop"]["records_kept"] for s in seeds]),
        "stop_references_left": sum(ret[s]["stop"]["references_left"] for s in seeds),
        "rebuild_identical": all(ret[s]["rebuild_identical"] for s in seeds),
        "pass": len(sd) >= 0.9 * len(seeds) and len(ss) >= 0.9 * len(seeds)
        and sum(ret[s]["delete"]["references_left"] + ret[s]["delete"]["found_in_store"]
                + ret[s]["stop"]["references_left"] for s in seeds) == 0
        and d_rev >= 0.95 and s_rev >= 0.95 and E.mean([ret[s]["stop"]["records_kept"] for s in seeds]) == 1.0
        and all(ret[s]["rebuild_identical"] for s in seeds)}
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("general", "dev", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--general", default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--filter-repo", default=None)
    ap.add_argument("--pool", default="test", choices=("test", "dev"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "general":
        stage_general(out, a.workers)
    elif a.stage == "dev":
        stage_dev(out, a.workers, a.general)
    else:
        seeds = [int(x) for x in a.seeds.split(",") if x]
        stage_test(out, a.workers, a.general, a.params, seeds, a.filter_repo.split(), a.pool)


if __name__ == "__main__":
    main()
