#!/usr/bin/env python3
"""Cycle 1 GPU Evaluation: McCulloch & Pitts (1943) Inhibitory Veto + Minsky & Papert (1969)
Order-k 'One-in-a-Box' Conjunctive Predicates on Clover's Ontology GraphRAG & Memory Graph.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import torch
import k3_ontology_graph as kog
import k3_memory_graph as kmg


def _get_code_provenance() -> dict:
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(_SCRIPT_DIR), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_sha = "standalone_checkout"
    return {
        "script_path": str(Path(__file__).resolve()),
        "git_head_sha": git_sha,
        "k3_ontology_graph_file": str(Path(kog.__file__).resolve()),
        "k3_ontology_graph_sha256_12": hashlib.sha256(Path(kog.__file__).read_bytes()).hexdigest()[:12],
        "k3_memory_graph_file": str(Path(kmg.__file__).resolve()),
        "k3_memory_graph_sha256_12": hashlib.sha256(Path(kmg.__file__).read_bytes()).hexdigest()[:12],
    }


ONTOLOGY_STRESS_QUERIES = [
    {
        "id": "ONTO-S1-OIL-SPILL-BUOYANCY",
        "kind": "one_in_a_box_and_veto",
        "query": "A puddle of water on a garage floor is covered by a layer of oil with density 800 kg/m^3. Will a plastic bead of density 920 kg/m^3 float at the oil-water interface by Archimedes' principle?",
        "unwanted_ref": "STEM_CONCEPT:STEM-THIN-FILM-INTERFERENCE",
    },
    {
        "id": "ONTO-S2-TFT-TRANSISTOR",
        "kind": "one_in_a_box_and_veto",
        "query": "In an amorphous silicon thin-film transistor (thin film MOSFET), how does gate oxide capacitance affect threshold voltage and drain current?",
        "unwanted_ref": "STEM_CONCEPT:STEM-THIN-FILM-INTERFERENCE",
    },
    {
        "id": "ONTO-S3-GENERIC-COUNCIL-BUDGET",
        "kind": "lexical_one_in_a_box",
        "query": "On Tuesday the district council voted to approve million pounds for a temporary street parking project lasting several weeks starting in March.",
        "unwanted_ref": "DOC_RECORD:DOC-HARBOURSIDE-CYCLE",
    },
    {
        "id": "ONTO-S4-GENERIC-DB-FAILOVER",
        "kind": "lexical_one_in_a_box",
        "query": "When a primary database volume fills and automated failover promotes a replica in another region, how are delayed and failed transactions rolled back?",
        "unwanted_ref": "DOC_RECORD:DOC-PAYMENTS-OUTAGE-209",
    },
    {
        "id": "ONTO-S5-GENERIC-POSTGRES-CLUSTER",
        "kind": "lexical_one_in_a_box",
        "query": "In a distributed storage cluster with synchronous replication across availability zones, how do read replicas maintain low latency on batch records per transaction?",
        "unwanted_ref": "DOC_RECORD:DOC-AURORA-STORAGE-V42",
    },
    {
        "id": "ONTO-S6-ALGEBRA-POLYNOMIAL-DIVISORS",
        "kind": "inhibitory_veto",
        "query": "Find the common divisor polynomial of x^3 - 1 and x^2 - 1 over the ring of polynomials Q[x] using polynomial long division.",
        "unwanted_ref": "STEM_CONCEPT:STEM-COMMON-DIVISORS",
    },
    {
        "id": "ONTO-S7-LOGISTIC-REGRESSION-ML",
        "kind": "one_in_a_box_and_veto",
        "query": "When training a logistic classification model on a population of 10,000 hospital patients, how does L2 regularization affect cross-entropy loss?",
        "unwanted_ref": "STEM_CONCEPT:STEM-LOGISTIC-CARRYING-CAPACITY",
    },
    {
        "id": "ONTO-S8-CIRCLE-MIRROR-OPTICS",
        "kind": "lexical_one_in_a_box",
        "query": "When a laser beam in the plane is reflected from a curved mirror surface of radius r, calculate the focal point coordinates.",
        "unwanted_ref": "STEM_CONCEPT:STEM-CIRCLE-POINT-REFLECTION",
    },
]

MEMORY_STRESS_QUERIES = [
    {
        "id": "MEM-S1-SWIMMING-POOL-THERMODYNAMICS",
        "kind": "inhibitory_veto_and_box",
        "query": "In an outdoor heated swimming pool, the water temperature stays flat across different pump capacity settings because heat loss to the night air is the bottleneck. What should we suspect about the pool cover?",
    },
    {
        "id": "MEM-S2-CELL-BIOLOGY-ENDOCYTOSIS",
        "kind": "inhibitory_veto",
        "query": "In eukaryotic cell biology, once receptor-mediated endocytosis completes, everything happens inside the intracellular endosome and lysosome compartments rather than outside in the extracellular matrix. How does pH drop?",
    },
    {
        "id": "MEM-S3-BASEBALL-BATTING-ZERO-HITS",
        "kind": "inhibitory_veto_and_box",
        "query": "A rookie baseball batter got 0 hits across 64 repeated curveball pitches in batting practice because his swing timing drifts early. Does this prove his bat weight is useless?",
    },
    {
        "id": "MEM-S4-FORKLIFT-PALLET-WAREHOUSE",
        "kind": "inhibitory_veto_and_box",
        "query": "In a warehouse timing study, a standalone forklift moving random pallets in an isolated aisle was 2x faster than in situ on the real loading dock. How many pallets per hour can it move?",
    },
    {
        "id": "MEM-S5-HIGHWAY-TRAFFIC-1HOP-EXIT",
        "kind": "inhibitory_veto_and_box",
        "query": "On a commuter highway, a 1-hop exit ramp with a 50-car limit floods the local street and blocks the 2-hop arterial route to the hospital. How should traffic lights be timed?",
    },
    {
        "id": "MEM-S6-CSS-SHORTHAND-SHADOWING",
        "kind": "inhibitory_veto_and_box",
        "query": "In a woodworking shop, a shorter 1-char stencil label on a shelf is shadowed by a longer 2-char box label when sorting jars by branch. How should the jars be arranged?",
    },
    {
        "id": "MEM-S7-GAMING-GPU-MONITOR-NIBBLE",
        "kind": "inhibitory_veto_and_box",
        "query": "When streaming a 4-bit retro pixel art game on an RTX 3090 GPU vs a laptop CPU, does HDMI display cable bandwidth limit the monitor refresh rate?",
    },
    {
        "id": "MEM-S8-BOOK-EDITION-CORRECTION",
        "kind": "inhibitory_veto",
        "query": "In the 1920 novel's second printing, an earlier interpretation of the detective's alibi was superseded by a correction in chapter 12. Who was the culprit?",
    },
]

ONTOLOGY_POSITIVE_QUERIES = [
    ("Which of the following could be the complete set of common divisors of two integers?", "STEM_CONCEPT:STEM-COMMON-DIVISORS"),
    ("What is the most effective long-term means of controlling a mosquito population using logistic carrying capacity?", "STEM_CONCEPT:STEM-LOGISTIC-CARRYING-CAPACITY"),
    ("A photon detector has a quantum efficiency of 0.1. If 100 photons are sent into the detector, what is the rms deviation?", "STEM_CONCEPT:STEM-PHOTON-BINOMIAL-DETECTOR"),
    ("A thin layer of oil (index of refraction 1.50, thickness 500 nm) floats on a puddle of water (index of refraction 1.33). What is the most strongly reflected wavelength?", "STEM_CONCEPT:STEM-THIN-FILM-INTERFERENCE"),
    ("The circle x^2 + y^2 = 16 is reflected in the point (3, 2) to give x^2 + y^2 + bx + cy + d = 0. What is the ordered triple (b, c, d)?", "STEM_CONCEPT:STEM-CIRCLE-POINT-REFLECTION"),
    ("It costs $90 to plant a lemon tree. Each year it grows 7 lemons which sell for $1.50 each, and it costs $3 a year to water and feed. How many years before he starts earning money?", "STEM_CONCEPT:STEM-PROFIT-BREAKEVEN"),
    ("How many temporary parking spaces did the Harbourside Council allocate on Dock Street for the Mill Road cycle lane?", "DOC_RECORD:DOC-HARBOURSIDE-CYCLE"),
    ("In the Region A payments database outage at 02:14 UTC before Region B failover at 02:31 UTC, how many transactions failed permanently?", "DOC_RECORD:DOC-PAYMENTS-OUTAGE-209"),
    ("What database version and maximum batch size of 4,096 records does Aurora Distributed Storage v4.2 use?", "DOC_RECORD:DOC-AURORA-STORAGE-V42"),
    ("Explain CVE-2024-6387 and its CWE and CAPEC attack patterns.", "CVE-2024-6387"),
]

MEMORY_POSITIVE_QUERIES = [
    ("In a 48-stage packet-processing pipeline, a global shared LRU cache stays flat at 19.4% across capacities 512 to 4096. What should we suspect about the shared pool?", "MEM-CONCEPT-FLAT-RESPONSE-DIAGNOSIS-001"),
    ("We tested a recurrent layer activation memoization cache by repeating the same token 64 times in one sequence and got 0 hits. Does this prove the cache can never hit?", "MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001"),
    ("In our security graph database, sorting paths by 1-hop length(p.edges) first caused 150 AFFECTS edges to drown out all 2-hop CWE->CAPEC root-cause paths. How should we fix traversal?", "MEM-OBS-ONTORAG-FANOUT-FLOODING-001"),
    ("Can we conclude from a CPU 4-bit quantisation run (-2.4% vs -18.3% LB_Q4) that 4-bit is a universal dead end on our 8 x RTX 3090 GPU?", "MEM-CONCEPT-HARDWARE-SCOPED-VERDICTS-001"),
    ("What did our recorded memory say about 'everything happens inside' and how was the initial interpretation corrected regarding the inside and outside boundary?", "MEM-CORR-INSIDE-BOUNDARY-002"),
    ("In our hand-written Python lexer, checking a shorter 1-char prefix '+' before a longer 2-char prefix '+=' in an if/elif chain shadowed the '+=' branch. What principle applies?", "MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001"),
]


def run_cycle1_eval() -> dict:
    device = "cuda:7" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(7) if torch.cuda.is_available() else "CPU"
    retriever = kog.CloverOntologyRetriever()
    mem_store = kmg.ensure_default_memory_tree()

    onto_stress_fp = sum(1 for item in ONTOLOGY_STRESS_QUERIES if len(retriever.link_entities(item["query"])) > 0)
    mem_stress_fp = sum(1 for item in MEMORY_STRESS_QUERIES if kmg.retrieve_relevant_memory_package(mem_store, item["query"]) is not None)

    onto_pos_hits = sum(
        1 for q, exp_id in ONTOLOGY_POSITIVE_QUERIES
        if exp_id in {eid for item in retriever.retrieve(q) for eid in item.entities}
    )
    mem_pos_hits = sum(
        1 for q, exp_id in MEMORY_POSITIVE_QUERIES
        if (pkg := kmg.retrieve_relevant_memory_package(mem_store, q)) is not None
        and exp_id in {it.entry.id for it in pkg.items}
    )

    return {
        "code_provenance": _get_code_provenance(),
        "device": device,
        "gpu_name": gpu_name,
        "onto_stress_fp": f"{onto_stress_fp}/{len(ONTOLOGY_STRESS_QUERIES)}",
        "mem_stress_fp": f"{mem_stress_fp}/{len(MEMORY_STRESS_QUERIES)}",
        "total_stress_fp": f"{onto_stress_fp + mem_stress_fp}/{len(ONTOLOGY_STRESS_QUERIES) + len(MEMORY_STRESS_QUERIES)}",
        "onto_pos_recall": f"{onto_pos_hits}/{len(ONTOLOGY_POSITIVE_QUERIES)}",
        "mem_pos_recall": f"{mem_pos_hits}/{len(MEMORY_POSITIVE_QUERIES)}",
    }


if __name__ == "__main__":
    print(json.dumps(run_cycle1_eval(), indent=2))
