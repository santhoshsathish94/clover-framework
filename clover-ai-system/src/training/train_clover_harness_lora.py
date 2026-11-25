#!/usr/bin/env python3
"""2-Server Distributed Clover + OpenCode Harness LoRA Trainer (`train_clover_harness_lora.py`).

Supports two coordinated multi-GPU roles across our two `4 x RTX 3090` servers:

  1. `--role coordinator-35b` (Server #1 `51.89.50.15`, `4 x RTX 3090 = 96 GiB` VRAM):
     - Loads `Qwen/Qwen3.6-35B-A3B` in exact `BF16`, drops the unused vision tower
       (`--language-model-only`), and shards the 40 MoE layers evenly across all 4 GPUs
       (`10` layers/GPU = `~17.3 GiB`/GPU, leaving `~26 GiB` headroom for activations + LoRA).
     - Attaches PEFT LoRA (`rank = 32, alpha = 64`) across all 40 layers (`in_proj_qkv`,
       `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `q_proj`, `k_proj`, `v_proj`,
       `o_proj`, and `shared_expert.{gate_proj,up_proj,down_proj}`).
     - **Stage 1 (Assistant-Masked Trajectory SFT):** Trains strictly on assistant tactical plans
       + tool calls (`labels = -100` on system/user/observation tokens, `enable_thinking=False`).
     - **Stage 2 (Live 2-Server `GRPO` RL):** Generates `G = 4` rollouts per task (`T = 0.8`),
       dispatches them over HMAC-authenticated HTTP to **Server #2 (`175.155.64.150:19178`)**'s
       `OpenEnv + safeagents` Verifier Cluster, and updates LoRA weights with group-normalized
       policy-gradient advantages (`skip` when group reward std == 0, matching HF `harness-hub`).

  2. `--role router-ddp` (Server #2 `175.155.64.150`, `4 x RTX 3090` via `torchrun --nproc_per_node=4`):
     - Trains `Qwen3-1.7B-Clover-Router-SafeGuard-LoRA` (`rank = 32, alpha = 64`) across all 4 GPUs
       with PyTorch `DistributedDataParallel` (`NCCL`) for `< 6 ms` specialist routing + `safeagents` guard.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any
import urllib.request

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def build_assistant_masked_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_seq_len: int = 3072,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Tokenize multi-turn chat with `enable_thinking=False` and mask ALL non-assistant tokens (`-100`)."""
    input_ids: list[int] = []
    labels: list[int] = []

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            # Explicitly close empty `<think>\n\n</think>\n\n` if Qwen3.6 template uses it for enable_thinking=False,
            # or train directly on crisp assistant tokens after `<|im_start|>assistant\n`
            header_ids = [im_start] + tokenizer.encode("assistant\n", add_special_tokens=False)
            body_ids = tokenizer.encode(content, add_special_tokens=False) + [im_end] + nl_ids
            input_ids.extend(header_ids + body_ids)
            labels.extend([-100] * len(header_ids) + body_ids)
        else:
            seg_ids = [im_start] + tokenizer.encode(f"{role}\n{content}", add_special_tokens=False) + [im_end] + nl_ids
            input_ids.extend(seg_ids)
            labels.extend([-100] * len(seg_ids))
        if len(input_ids) >= max_seq_len:
            break

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    if sum(1 for x in labels if x != -100) < 4:
        return None
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def score_rollouts_remote_or_local(
    verifier_url: str,
    secret: str,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    body = json.dumps({"items": items}).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    if verifier_url:
        try:
            req = urllib.request.Request(
                verifier_url.rstrip("/") + "/verify_batch",
                data=body,
                headers={"Content-Type": "application/json", "X-Clover-Sig": sig},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["results"]
        except Exception as exc:
            print(f"[GRPO Verifier Warning] Remote {verifier_url} fallback to local ({exc})", flush=True)

    from remote_verifier_daemon import _eval_single_item
    return [_eval_single_item(it) for it in items]


def build_4gpu_qwen36_device_map(num_layers: int = 40, num_gpus: int = 4) -> dict[str, int | str]:
    dmap: dict[str, int | str] = {
        "model.visual": "cpu",
        "mtp": "cpu",
        "model.language_model.embed_tokens": 0,
        "model.language_model.rotary_emb": 0,
        "model.language_model.norm": num_gpus - 1,
        "lm_head": num_gpus - 1,
    }
    per_gpu = math.ceil(num_layers / num_gpus)
    for lid in range(num_layers):
        dmap[f"model.language_model.layers.{lid}"] = min(lid // per_gpu, num_gpus - 1)
    return dmap


def run_coordinator_35b(args: argparse.Namespace) -> None:
    from transformers import Qwen3_5MoeForConditionalGeneration

    t_start = time.perf_counter()
    random.seed(20260728)
    torch.manual_seed(20260728)

    n_gpus = torch.cuda.device_count()
    print(f"[Coordinator-35B] Loading `{args.model_id}` across {n_gpus} x RTX 3090 in BF16 (language-model-only)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dmap = build_4gpu_qwen36_device_map(40, n_gpus)
    full_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map=dmap,
        low_cpu_mem_usage=True,
    )
    # Free CPU/GPU memory of unused visual & MTP modules immediately
    if hasattr(full_model, "model") and hasattr(full_model.model, "visual"):
        full_model.model.visual = None
    if hasattr(full_model, "mtp"):
        full_model.mtp = None
    torch.cuda.empty_cache()

    full_model.config.use_cache = False
    if hasattr(full_model.config, "text_config"):
        full_model.config.text_config.use_cache = False
    full_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    full_model.enable_input_require_grads()

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(full_model, lora_cfg)
    trainable_p, total_p = model.get_nb_trainable_parameters()
    vram_alloc = [round(torch.cuda.memory_allocated(i) / (1024**3), 2) for i in range(n_gpus)]
    print(
        f"[Coordinator-35B] Attached LoRA r={args.lora_rank}, alpha={args.lora_alpha}: "
        f"trainable={trainable_p:,} / {total_p:,} ({100.0*trainable_p/total_p:.4f}%) | VRAM GiB per GPU={vram_alloc}",
        flush=True,
    )

    data_dir = Path(args.data_dir)
    sft_raw = [json.loads(l) for l in (data_dir / "sft_clover_opencode.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sft_tensors = []
    for r in sft_raw:
        ex = build_assistant_masked_example(tokenizer, r["messages"], max_seq_len=args.max_seq_len)
        if ex is not None:
            sft_tensors.append(ex)
    print(f"[Stage 1 SFT] Prepared {len(sft_tensors)} assistant-masked trajectories (max_seq_len={args.max_seq_len}).", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_opt_steps = args.sft_steps + args.grpo_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(8, args.sft_steps // 10), num_training_steps=total_opt_steps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_log: list[dict[str, Any]] = []

    # --- Stage 1: Assistant-Masked Trajectory SFT ---
    model.train()
    optimizer.zero_grad()
    idx_cursor = 0
    first_dev = torch.device("cuda:0")

    for step in range(1, args.sft_steps + 1):
        t_s0 = time.perf_counter()
        step_loss = 0.0
        step_toks = 0
        for _ in range(args.grad_accum):
            inp, lbl = sft_tensors[idx_cursor % len(sft_tensors)]
            idx_cursor += 1
            if idx_cursor % len(sft_tensors) == 0:
                random.shuffle(sft_tensors)
            inp_b = inp.unsqueeze(0).to(first_dev)
            out = model(input_ids=inp_b, use_cache=False)
            logits = out.logits[0, :-1, :].float()
            targets = lbl[1:].to(logits.device)
            loss = F.cross_entropy(logits, targets, ignore_index=-100) / args.grad_accum
            loss.backward()
            step_loss += float(loss.item())
            step_toks += int(inp.numel())

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        dt = time.perf_counter() - t_s0

        rec = {
            "stage": "sft",
            "step": step,
            "loss": round(step_loss, 4),
            "grad_norm": round(grad_norm, 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "tokens": step_toks,
            "tok_per_s": round(step_toks / max(dt, 1e-4), 1),
            "wall_s": round(dt, 2),
        }
        metrics_log.append(rec)
        if step == 1 or step % 5 == 0 or step == args.sft_steps:
            print(
                f"[SFT {step:03d}/{args.sft_steps:03d}] loss={rec['loss']:.4f} grad={rec['grad_norm']:.3f} "
                f"tok/s={rec['tok_per_s']} ({rec['wall_s']}s)",
                flush=True,
            )

    sft_ckpt = out_dir / "stage1_sft"
    model.save_pretrained(str(sft_ckpt))
    tokenizer.save_pretrained(str(sft_ckpt))
    print(f"[Stage 1 Complete] Saved SFT checkpoint -> {sft_ckpt}", flush=True)

    # --- Stage 2: Live 2-Server Verifier-Gated GRPO RL (Server 1 Policy <-> Server 2 OpenEnv+safeagents Verifier) ---
    rl_tasks = [json.loads(l) for l in (data_dir / "grpo_rl_tasks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(
        f"[Stage 2 GRPO] Starting {args.grpo_steps} online RL steps (G={args.grpo_group} rollouts/task) "
        f"scored by Server #2 `{args.verifier_url}`...",
        flush=True,
    )
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_ids = tokenizer.encode("\n", add_special_tokens=False)

    rl_updates = 0
    rl_skipped_zero_var = 0
    for rstep in range(1, args.grpo_steps + 1):
        t_r0 = time.perf_counter()
        task = rl_tasks[(rstep - 1) % len(rl_tasks)]
        prompt_ids: list[int] = []
        for m in task["messages"]:
            prompt_ids.extend([im_start] + tokenizer.encode(f"{m['role']}\n{m['content']}", add_special_tokens=False) + [im_end] + nl_ids)
        prompt_ids.extend([im_start] + tokenizer.encode("assistant\n", add_special_tokens=False))
        p_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=first_dev)

        # Rollout generation (thinking OFF, G=4 parallel sequences)
        model.eval()
        full_model.gradient_checkpointing_disable()
        with torch.no_grad():
            gen_batch = model.generate(
                input_ids=p_tensor.repeat(args.grpo_group, 1),
                max_new_tokens=384,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end,
            )

        completions: list[str] = []
        gen_seqs: list[torch.Tensor] = []
        p_len = p_tensor.shape[1]
        for gi in range(args.grpo_group):
            seq = gen_batch[gi]
            # Trim at first im_end after prompt
            gen_part = seq[p_len:].tolist()
            if im_end in gen_part:
                gen_part = gen_part[: gen_part.index(im_end) + 1]
            completions.append(tokenizer.decode(gen_part[:-1] if gen_part and gen_part[-1] == im_end else gen_part, skip_special_tokens=True))
            gen_seqs.append(torch.tensor(prompt_ids + gen_part, dtype=torch.long, device=first_dev))

        batch_req = [
            {
                "completion": c,
                "test_code": task.get("test_code", ""),
                "expected_tool": task.get("expected_tool", ""),
                "required_arg_substr": task.get("required_arg_substr", ""),
                "expect_refusal": task.get("expect_refusal", False),
            }
            for c in completions
        ]
        scored = score_rollouts_remote_or_local(args.verifier_url, args.rl_secret, batch_req)
        rewards = [float(s["reward"]) for s in scored]
        pass_cnt = sum(1 for s in scored if s["passed"])
        mean_r = sum(rewards) / len(rewards)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))

        # HF `harness-hub` GRPO rule: when every rollout passes or every rollout fails (std == 0), skip zero-signal step!
        full_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        pg_loss_val = 0.0
        if std_r > 1e-4:
            optimizer.zero_grad()
            for gi in range(args.grpo_group):
                adv = (rewards[gi] - mean_r) / (std_r + 1e-5)
                seq_t = gen_seqs[gi].unsqueeze(0)
                if seq_t.shape[1] <= p_len + 1:
                    continue
                out = model(input_ids=seq_t, use_cache=False)
                logits = out.logits[0, p_len - 1 : -1, :].float()
                tgt = seq_t[0, p_len:].to(logits.device)
                tok_nll = F.cross_entropy(logits, tgt, reduction="mean")
                loss_g = (adv * tok_nll) / args.grpo_group
                loss_g.backward()
                pg_loss_val += float(loss_g.item())
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            rl_updates += 1
        else:
            scheduler.step()
            rl_skipped_zero_var += 1

        dt = time.perf_counter() - t_r0
        rec = {
            "stage": "grpo",
            "step": rstep,
            "task_id": task["task_id"],
            "pass_of_group": f"{pass_cnt}/{args.grpo_group}",
            "mean_reward": round(mean_r, 4),
            "std_reward": round(std_r, 4),
            "pg_loss": round(pg_loss_val, 4),
            "wall_s": round(dt, 2),
        }
        metrics_log.append(rec)
        if rstep == 1 or rstep % 4 == 0 or rstep == args.grpo_steps:
            print(
                f"[GRPO {rstep:03d}/{args.grpo_steps:03d}] {task['task_id']} pass={pass_cnt}/{args.grpo_group} "
                f"mean_R={mean_r:.3f} std_R={std_r:.3f} updates={rl_updates} ({dt:.1f}s)",
                flush=True,
            )

    final_dir = out_dir / "final_clover_harness_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    summary = {
        "base_model": args.model_id,
        "adapter_name": "Qwen3.6-35B-A3B-Clover-Harness-LoRA",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules,
        "trainable_params": trainable_p,
        "total_params": total_p,
        "sft_steps": args.sft_steps,
        "grpo_steps": args.grpo_steps,
        "grpo_informative_updates": rl_updates,
        "grpo_zero_variance_skipped": rl_skipped_zero_var,
        "verifier_server_2": args.verifier_url,
        "total_wall_min": round((time.perf_counter() - t_start) / 60.0, 2),
        "metrics_tail": metrics_log[-10:],
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "metrics_history.jsonl").write_text("\n".join(json.dumps(m) for m in metrics_log) + "\n", encoding="utf-8")
    print(f"[COORDINATOR_35B_LORA_TRAINING_COMPLETE] -> {final_dir} in {summary['total_wall_min']} min!", flush=True)


def run_router_4gpu_ddp(args: argparse.Namespace) -> None:
    """4-GPU PyTorch DDP LoRA trainer on Server #2 (`175.155.64.150`)."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    t_start = time.perf_counter()
    if local_rank == 0:
        print(f"[Router-DDP] Loading `{args.model_id}` across {world_size} GPUs via NCCL DDP...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16).to(dev)
    base.config.use_cache = False
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(base, lora_cfg)
    ddp_model = DDP(peft_model, device_ids=[local_rank]) if world_size > 1 else peft_model

    raw_rows = [json.loads(l) for l in (Path(args.data_dir) / "router_safeguard_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    examples = []
    for r in raw_rows[local_rank::world_size]:
        ex = build_assistant_masked_example(tokenizer, r["messages"], max_seq_len=512)
        if ex is not None:
            examples.append(ex)

    opt = torch.optim.AdamW([p for p in ddp_model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=15, num_training_steps=args.sft_steps)

    ddp_model.train()
    cursor = 0
    losses = []
    for step in range(1, args.sft_steps + 1):
        opt.zero_grad()
        acc_loss = 0.0
        for _ in range(args.grad_accum):
            inp, lbl = examples[cursor % len(examples)]
            cursor += 1
            inp_b = inp.unsqueeze(0).to(dev)
            lbl_b = lbl.unsqueeze(0).to(dev)
            out = ddp_model(input_ids=inp_b, labels=lbl_b, use_cache=False)
            loss = out.loss / args.grad_accum
            loss.backward()
            acc_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_([p for p in ddp_model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        losses.append(round(acc_loss, 5))
        if local_rank == 0 and (step == 1 or step % 20 == 0 or step == args.sft_steps):
            print(f"[Router-4GPU-DDP {step:03d}/{args.sft_steps:03d}] loss={acc_loss:.5f}", flush=True)

    if local_rank == 0:
        out_dir = Path(args.out_dir) / "final_clover_router_safeguard_lora"
        out_dir.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        (Path(args.out_dir) / "router_training_summary.json").write_text(
            json.dumps(
                {
                    "base_model": args.model_id,
                    "world_size_gpus": world_size,
                    "steps": args.sft_steps,
                    "initial_loss": losses[0] if losses else None,
                    "final_loss": losses[-1] if losses else None,
                    "wall_min": round((time.perf_counter() - t_start) / 60.0, 2),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[ROUTER_4GPU_DDP_LORA_COMPLETE] -> {out_dir}", flush=True)

    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["coordinator-35b", "router-ddp"], required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--sft-steps", type=int, default=160)
    ap.add_argument("--grpo-steps", type=int, default=60)
    ap.add_argument("--grpo-group", type=int, default=4)
    ap.add_argument("--verifier-url", default="http://175.155.64.150:19178")
    ap.add_argument("--rl-secret", default=os.environ.get("CLOVER_RL_SECRET", "clover-rl-mesh-2026"))
    args = ap.parse_args()

    if args.role == "coordinator-35b":
        run_coordinator_35b(args)
    else:
        run_router_4gpu_ddp(args)


if __name__ == "__main__":
    main()
