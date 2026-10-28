"""Cycle 51: Direct Chat Mode (`k3_fast_chat.py`) — Plain Prompt In -> Output Out (Zero 93-Layer Verification).
Runs the unified K3-Native pipeline directly from a user prompt (or file of prompts):
  1. `HybridAgent-8B-Agentic` (`cuda:1`) runs a fast `<latent>` reasoning span (`R(h_t)`) and emits 1-2 factual/narrative
     window plans (`Window 1:` and `Window 2:`), plus extracts continuous `R(h)` (`[B, 48, 4096]`).
  2. `e6_k3_native` (`11`-pass NCM generator) + `K3PlanCopyDecoder` (`cuda:0`) generates the response directly in
     Clover Kimi K3's native `163,840`-token vocabulary (`libk3tok.so`) in a single batched pass (~125-355 ms) —
     with ZERO 93-layer Kimi K3 verification and ZERO DSpark verifier.
"""
import argparse, collections, json, os, re, sys, time
from clover_paths import TOOLS_DIR, setup_clover_sys_path, find_hybrid_ncm_ckpt_dir, resolve_device, cuda_sync

ROOT = setup_clover_sys_path()
if os.path.isdir(ROOT):
    os.chdir(ROOT)

import torch
import torch.nn.functional as F

from k3_tokenizer import K3Tokenizer
from train_k3_native_all import load_k3_native_models, V_K3, D_VQ, D_HA, L_WMT, P_RH, P_BPE, P_COMBO, K3_EOS, K3_PAD
from train_e7_clean_chat import K3LexicalPointerDecoder
from train_combo_prefix import ComboBridge, pack_combo_prefix
from eval_both_tracks import jacc, pick_tokens, generate_from_zpref
from hybrid_agent.model import load_hybrid_agent
from hybrid_agent.constants import ENTRY_ANCHOR, EXIT_ANCHOR, THINK_OPEN, THINK_CLOSE, special_token_id
from train_latent_bridge import encode_batch_rh


CLOVER_DEFAULT_CONTEXT_PATH = os.environ.get(
    "CLOVER_DEFAULT_CONTEXT_PATH",
    os.path.join(TOOLS_DIR, "CLOVER_DEFAULT_CONTEXT.md")
    if os.path.exists(os.path.join(TOOLS_DIR, "CLOVER_DEFAULT_CONTEXT.md"))
    else "/home/usergpu/merge_lf/CLOVER_DEFAULT_CONTEXT.md",
)


def _load_memory_tree_rules() -> dict:
    base_rules = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_tree", "_rules")
    for fname in ("dataset-memory-retrieval-rules.json", "memory_retrieval_rules.json"):
        rules_fp = os.path.join(base_rules, fname)
        if os.path.exists(rules_fp):
            try:
                with open(rules_fp, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


_MEMORY_TREE_RULES = _load_memory_tree_rules()
CLOVER_AGENTS_MD_KERNEL = str(_MEMORY_TREE_RULES.get("clover_agents_md_kernel") or "")
_PLANNER_META_PREFIXES = tuple(str(x) for x in (_MEMORY_TREE_RULES.get("planner_meta_prefixes") or ()))


def load_default_clover_context(path: str = CLOVER_DEFAULT_CONTEXT_PATH) -> str:
    """Loads `CLOVER_DEFAULT_CONTEXT.md` from disk so default Clover operating rules and architecture facts
    are read from the context file rather than hardcoded in Python."""
    for candidate in (path, os.path.join(os.path.dirname(__file__), "CLOVER_DEFAULT_CONTEXT.md")):
        if candidate and os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                pass
    return CLOVER_AGENTS_MD_KERNEL


CHAT_SYSTEM_PROMPT = (
    "You are Clover, a polite, kind, and helpful assistant built on Clover (Kimi K3 93-Layer 2.78T MoE + HybridAgent + Next-Concept Modeling + Ontology GraphRAG + Generative Compilation + Collective Memory Tree) inside the Clover Framework. "
    + CLOVER_AGENTS_MD_KERNEL
    + str(_MEMORY_TREE_RULES.get("chat_system_prompt_body") or "")
)

CHAT_PLAN_PROMPT = (
    "Write your direct response to the user starting with `W1:` (and `W2:`, `W3:`, `W4:` on new lines if the answer has multiple parts or a timetable/schedule). "
    "Do NOT write meta-commentary about what the user asked; write ONLY the actual reply to the user.\n\n"
    "User message:\n{prompt}"
)


def is_short_greeting_or_ack(user_prompt: str) -> bool:
    p = re.sub(r"[^a-z\s]", " ", user_prompt.lower())
    # Strip 'clover' when used as a vocative ('Hi clover', 'Hello clover how are you', 'Thanks clover')
    p_no_name = " ".join(w for w in p.split() if w != "clover").strip()
    if p_no_name in (
        "hi", "hello", "hey", "hi there", "hello there", "hey there",
        "good morning", "good afternoon", "good evening",
        "how are you", "how are you doing", "how are you doing today",
        "hi how are you", "hi how are you doing", "hi how are you doing today",
        "hello how are you", "hello how are you doing", "hello how are you doing today",
        "hey how are you", "hey how are you doing", "hey how are you doing today",
        "thanks", "thank you", "thank you so much", "ty", "thanku",
        "ok", "okay", "got it", "bye", "goodbye",
    ):
        return True
    return False


def is_conversational_meta_or_confirmation(user_prompt: str) -> bool:
    """True for short conversational confirmations or capability check-ins ('really', 'are you sure',
    'can you really help me', 'is it working', 'you there') that should be answered in 1-2 natural sentences."""
    p = re.sub(r"[^a-z0-9\s]", " ", user_prompt.lower())
    p_clean = " ".join(w for w in p.split() if w != "clover").strip()
    if p_clean in (
        "really", "oh really", "for real", "seriously", "are you sure", "you sure",
        "can you really", "can you really help", "can you really help me",
        "can you help", "can you help me", "are you there", "you there",
        "is this working", "is it working", "does this work", "test", "testing",
        "nice", "cool", "awesome", "great", "good", "wow",
    ):
        return True
    return False


def is_structured_or_multi_part_prompt(user_prompt: str) -> bool:
    if is_short_greeting_or_ack(user_prompt) or is_conversational_meta_or_confirmation(user_prompt):
        return False
    p = user_prompt.lower().strip()
    # Short single-fact lookup or simple arithmetic questions ('what is the capital of...', 'what is 25 * 4?') are NOT multi-part
    if len(p.split()) <= 9 and re.search(r"\b\d+\s*[\+\-\*/]\s*\d+\b", p) and "step" not in p:
        return False
    if len(p.split()) <= 9 and p.startswith(("what is the ", "what is our ", "what is my ", "who is the ", "when is the ", "where is the ")) and not any(
        w in p for w in (
            "and", "difference", "compare", "trends", "steps", "how",
            "code", "function", "program", "algorithm", "script", "prime", "binary", "sort", "fibonacci",
            "schedule", "timetable", "plan", "list", "guide", "design", "architecture", "clover", "kimi", "ncm", "hybrid",
            "sum", "product", "divisor", "gcd", "lcm", "molar", "distance", "speed", "time", "balance", "interest",
        )
    ):
        return False
    # Quantitative word problems, multiple-choice questions, code specs, or Sudoku puzzles are always multi-step
    if ("(a)" in p and "(b)" in p) or "def " in p or "```" in p or "sudoku" in p:
        return True
    if len(re.findall(r"\b\d+(?:\.\d+)?\b", p)) >= 2 and any(
        qw in p for qw in ("how many", "how much", "how long", "how far", "what is", "find the", "round to", "total", "left", "profit", "cost", "earn", "spend", "take", "takes")
    ):
        return True
    return any(
        kw in p
        for kw in (
            "time table", "timetable", "schedule", "study plan", "plan", "routine",
            "step", "steps", "list", "guide", "how can i", "how do i", "how should i", "how to ",
            "can you ", "could you ", "please ", "prepare", "explain", "compare", " vs ", " versus ",
            "find", "research", "recent", "search", "papers", "arxiv", "developments",
            "what is", "what are", "how does", "how do", "why ", "better than", "difference",
            "trends", "describe", "tell me", "overview", "analyze", "summary", "summarize",
            "detail", "details", "link", "links", "example", "examples", "point ", "gpu", "gpus", "chip", "chips",
            "launch", "launched", "code", "write", "implement", "design", "architecture", "calculate", "compound",
            "function", "program", "algorithm", "prime", "binary search", "fibonacci", "factorial", "quicksort",
            "http://", "https://", "refer", "read ", "agents.md", "ncm", "next-concept", "next concept",
            "hybridagent", "hybrid agent", "what is clover", "clover framework", "clover kimi", "clover k3",
            "clover-k3", "framework", "cut off", "didn't explain", "did not explain", "name", "names", "idea", "ideas",
            "suggest", "recommend", "type", "types", "kind", "kinds", "way", "ways", "reason", "reasons",
            "tip", "tips", "benefit", "benefits", "feature", "features", "google", "baby",
            "how many", "how much", "how long", "how far", "solve",
        )
    )


def parse_and_split_plans(raw_text: str, user_prompt: str, k3_tok, max_bpe: int = 32, max_windows: int = 24):
    """Parses a single-line or multi-line response (`W1: ... W2: ... W3: ...` or bulleted/numbered lines) from HybridAgent
    into `<= max_bpe` Kimi-K3-token windows without dropping lines that lack a `W2:` prefix. Preserves line-break boundaries
    via a leading `\n` marker on windows that start a new paragraph/schedule block."""
    wants_summary = ("summary" in user_prompt.lower()) or ("summarize" in user_prompt.lower())
    # If </think> is present, keep the text after the last non-empty </think> block
    for tc in (THINK_CLOSE, "</think>"):
        if tc in raw_text:
            parts = [p.strip() for p in raw_text.split(tc) if p.strip() and p.strip() != "W1:"]
            if parts:
                raw_text = parts[-1]
    raw_text = raw_text.replace("<answer>", "").replace("</answer>", "")
    # Remove any raw <search>...</search> or <python>...</python> tags
    raw_text = re.sub(r"<search>.*?</search>", "", raw_text, flags=re.IGNORECASE | re.DOTALL)
    raw_text = re.sub(r"<python>.*?</python>", "", raw_text, flags=re.IGNORECASE | re.DOTALL)
    # Truncate immediately before any leaked <think> tag or trailing post-answer self-review
    for stop_marker in (
        THINK_OPEN, "<think>", "<|im_end|>", "<|endoftext|>",
        "\nYou asked for ", "\nThe user asked for ", "\nI followed the instructions",
        "\nThe answer is concise", "\nNo other information or steps",
        "\nI've been given the information", "\nThis is a straightforward factual",
        "\nThe response should be", "\nThis result has been verified using Python",
        "\nThe calculation was verified using Python", "\nFor more detailed information, you can refer",
        "\nMy response will be structured", "\nI've provided the exact", "\nI've also included the verification",
        "\nThese are the key pieces of information extracted", "\nPython calculation confirms",
        "\nThis is a simple, friendly response", "\nThis response is concise", "\nThis response is direct",
        "\nThis response is polite", "\nI'm maintaining a polite", "\nI also included a question to encourage",
        "\nYour message is straightforward, so a single", "\nA single `W1:` line is sufficient",
        "\nwhich aligns with the character description",
        "\nThis sentence is concise", "\nThis sentence captures", "\nThis rewritten sentence",
    ):
        if stop_marker in raw_text:
            raw_text = raw_text.split(stop_marker)[0]
    raw_text = re.sub(
        r"\s*(?:This (?:is a simple, friendly|sentence (?:is concise|captures|accurately|combines)|response is (?:concise|direct|polite|clear|helpful|simple))[\s\S]*$|"
        r"I'm maintaining a polite and warm tone[\s\S]*$|"
        r"I also included a question to encourage[\s\S]*$|"
        r"Your message is straightforward,\s*so a single[\s\S]*$|"
        r"It's also polite and helpful,\s*which aligns with[\s\S]*$)",
        "",
        raw_text,
        flags=re.IGNORECASE,
    )
    # Replace <latent> / </latent> anchors first
    raw_text = raw_text.replace(ENTRY_ANCHOR, "\n").replace(EXIT_ANCHOR, "\n")
    raw_text = re.sub(r"</?latent>", "\n", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"<\|[^>]+?\|>", "", raw_text)

    # Extract fenced code blocks (` ```python ... ``` `) or unclosed trailing fenced code blocks (` ```python ... `)
    # BEFORE stripping `W1:`..`W99:` tags so 4-space indentation inside code blocks is 100% preserved!
    code_blocks = []

    def _save_code_block(lang: str, body: str) -> str:
        cleaned_lines = [re.sub(r"^\s*W\d+:\s?", "", ln) for ln in body.strip("\r\n").splitlines()]
        clean_body = "\n".join(cleaned_lines).strip("\r\n")
        if not clean_body:
            return ""
        idx_cb = len(code_blocks)
        lang_tag = (lang or "python").strip() or "python"
        code_blocks.append(f"```{lang_tag}\n{clean_body}\n```")
        return f"\n__CODE_BLOCK_{idx_cb}__\n"

    raw_text = re.sub(
        r"```([a-zA-Z0-9_+-]*)\s*\n([\s\S]*?)```",
        lambda m: _save_code_block(m.group(1), m.group(2)),
        raw_text,
    )
    if "```" in raw_text:
        raw_text = re.sub(
            r"```([a-zA-Z0-9_+-]*)\s*\n([\s\S]+)$",
            lambda m: _save_code_block(m.group(1), m.group(2)),
            raw_text,
        )

    raw_text = re.sub(r"(?:^|\s)\bW\d+:\s*", "\n", raw_text).strip()
    raw_text = re.sub(r"(?:^|\n)\s*W\d+\s*$", "", raw_text, flags=re.IGNORECASE).strip()
    # Join lines where an equation ends with `=` and the number is on the next line
    raw_text = re.sub(r"=\s*\n+\s*(?=[$\d\-])", "= ", raw_text)
    wants_one_sent = any(
        bs in user_prompt.lower()
        for bs in ("one sentence", "1 sentence", "one clear sentence", "one clear, grammatical sentence", "single sentence")
    )

    # Also detect unfenced Python function/class blocks (`def foo(...):` followed by indented lines or docstrings)
    if not code_blocks and re.search(r"(?:^|\n)\s*def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(", raw_text):
        lines_all = raw_text.splitlines()
        rebuilt_lines = []
        cur_code = []
        in_code = False
        for orig_ln in lines_all:
            s_ln = orig_ln.strip()
            s_clean = re.sub(r"^W\d+:\s*", "", s_ln, flags=re.IGNORECASE).strip()
            is_code_start = bool(re.match(r"^(?:def|class|import|from)\s+[a-zA-Z_]", s_clean))
            is_code_cont = in_code and (
                orig_ln.startswith(("  ", "\t"))
                or s_clean.startswith(('"""', "'''", "#", "return", "if ", "elif ", "else:", "for ", "while ", "try:", "except", "print("))
                or bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=", s_clean))
                or not s_clean
            )
            if is_code_start or is_code_cont:
                if not in_code:
                    in_code = True
                if s_clean:
                    cur_code.append(orig_ln if orig_ln.startswith(("  ", "\t")) else s_clean)
            else:
                if in_code and cur_code:
                    rebuilt_lines.append(_save_code_block("python", "\n".join(cur_code)))
                    cur_code = []
                    in_code = False
                rebuilt_lines.append(orig_ln)
        if in_code and cur_code:
            rebuilt_lines.append(_save_code_block("python", "\n".join(cur_code)))
        raw_text = "\n".join(rebuilt_lines)

    raw_lines = []
    seen_norm_lines = set()
    for ln in raw_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m_cb = re.match(r"^__CODE_BLOCK_(\d+)__$", ln)
        if m_cb:
            raw_lines.append(ln)
            continue
        ln = re.sub(r"^W\d+:\s*", "", ln, flags=re.IGNORECASE).strip()
        # Require whitespace after numbered list markers (`1. `) so leading decimals like `3.14` or `5.5` are never stripped
        ln = re.sub(r"^(?:[-*•]+\s*|\d+[.)]\s+)", "", ln).strip()
        ln = re.sub(r"\*\*", "", ln).strip()
        ln = re.sub(r":(\d+[.)]\s)", r": \1", ln)
        # Convert markdown links `[Label](https://...)` into `Label (https://...)` (or just `https://...` if Label == URL)
        ln = re.sub(
            r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
            lambda m: m.group(2) if m.group(1).strip() == m.group(2).strip() else f"{m.group(1)} ({m.group(2)})",
            ln,
        )
        # Strip trailing meta-notes after URLs like '(This is a general link to recent...)'
        ln = re.sub(r"\s*\(This is a general (?:link|list).*$", "", ln, flags=re.IGNORECASE).strip()
        # Convert bracketed domain paths like `[arxiv.org/list/cs.AI/recent]` into full `https://...` URLs
        ln = re.sub(
            r"\[(?:https?://)?((?:[a-z0-9\-]+\.)+(?:org|com|gov|edu|io|ai|net)/[^\s\]]+)\]",
            r"https://\1",
            ln,
            flags=re.IGNORECASE,
        )
        # Filter third-person planner meta-reasoning BEFORE any capitalization or normalization
        raw_low_pre = ln.lower()
        if (
            raw_low_pre.startswith((
                "the user ", "since the user", "because the user", "if the user", "when the user",
                "what the user", "given the context of the previous", "given the previous conversation",
                "for example, they might", "for example, the user",
                "they might want to", "they are ready to", "it's likely they", "it is likely they",
                "i should ask", "i need to clarify what", "to proceed, i need to",
                "to provide a helpful response, i need to", "since the previous conversation",
                "example prompt:", "this approach keeps the conversation",
                "it seems like you're asking for a reason", "could you please clarify what you're looking for",
                "are you asking why something happens", "once i understand the context, i can provide",
            ))
            or re.search(
                r"\b(?:the user's|the user is|the user was|the user has|the user needs|the user wants|the user asks|the user asked|the user didn't|confirmed their understanding|what they want to achieve|ensures the user gets|ensures you gets|likely they are ready|ask for more specific guidance|likely refers to their|that they understand how|i should highlight|i should focus on|i should mention|i should provide|i will formulate|i'll formulate)\b",
                raw_low_pre,
            )
        ):
            continue
        # Strip inline Example prompt: "..." or outer quote wrappers if the planner emitted a quoted response line
        m_ex_prompt = re.match(r"^(?:example\s+(?:prompt|response|reply)\s*:\s*)?[\"“](.+?)[\"”]?\s*$", ln, flags=re.IGNORECASE)
        if m_ex_prompt and not ln.endswith(":"):
            ln = m_ex_prompt.group(1).strip().strip('"“”')
        ln = re.sub(
            r"(?:Now I will write|The user is asking|Okay,\s*so\s*the\s*user|I should structure|No need for further|I will end the interaction|Identify the scope:|This approach ensures|This approach keeps|This response provides|This summary captures|I've kept the information concise|No further analysis|No other information or steps|The answer is concise|This is a simple, friendly response|This response is concise|This response is direct|This response is polite|I'm maintaining a polite|I also included a question to encourage|Your message is straightforward, so a single|which aligns with the character description|It also provides a simple example to illustrate).*$",
            "",
            ln,
            flags=re.IGNORECASE,
        ).strip()
        ln = re.sub(r"^From the previous (?:turn|analysis)[^,]*,\s*(?:I know that\s*|the key points were:\s*)?", "", ln, flags=re.IGNORECASE).strip()
        ln = re.sub(r"^(?:Based on|Looking at|According to|From) the (?:search results|provided reference|reference information|provided references|reference facts|information provided|information given|context provided)(?:\s+provided)?[,:\s]*", "", ln, flags=re.IGNORECASE).strip()
        ln = re.sub(r"\s*based on the provided (?:reference(?:\s+facts|s)?|information):", ":", ln, flags=re.IGNORECASE).strip()
        ln = re.sub(r"\s*based on the provided references:", ":", ln, flags=re.IGNORECASE).strip()
        ln = re.sub(r"\s*mentioned in the reference:", ":", ln, flags=re.IGNORECASE).strip()
        ln = re.sub(r"\s*\(from the provided references\)", "", ln, flags=re.IGNORECASE).strip()
        ln = re.sub(r"Here's what I found from the search results:", "Here is a breakdown:", ln, flags=re.IGNORECASE).strip()
        if not wants_summary:
            ln = re.sub(r"\s*;?\s*Summary:.*$", "", ln, flags=re.IGNORECASE).strip()
        # Replace semicolons with periods so `has_semi=False` in `_dec_k3_aligned` and 1-to-1 pointer alignment is used
        ln = re.sub(r";\s*", ". ", ln).strip()
        if not ln or re.match(r"^W\d+:?$", ln, flags=re.IGNORECASE):
            continue
        low = ln.lower()
        if (_PLANNER_META_PREFIXES and low.startswith(_PLANNER_META_PREFIXES)) or low in ("first,", "alright,", "okay,", "however,", "so,", "but"):
            continue

        # Check if this valid line is a mid-sentence continuation of `raw_lines[-1]` (e.g. previous line ended without
        # sentence punctuation `.!?:)]"'` or this line starts with a lowercase continuation word) BEFORE capitalizing `ln[0]`!
        starts_lowercase_cont = bool(ln and ln[0].islower() and not ln.startswith(("http", "def ", "class ", "import ", "from ", "return ", "__CODE_BLOCK_")))
        prev_unfinished = bool(
            raw_lines
            and not raw_lines[-1].startswith("__CODE_BLOCK_")
            and not raw_lines[-1].endswith((".", "!", "?", ":", ")", "]", '"', "'", "`"))
        )
        if raw_lines and (prev_unfinished or (starts_lowercase_cont and not re.match(r"^[a-z0-9_\-]+\s*:", ln))):
            raw_lines[-1] = (raw_lines[-1].rstrip() + " " + ln.lstrip()).strip()
            continue
        if starts_lowercase_cont:
            ln = ln[0].upper() + ln[1:]
        norm_key = re.sub(r"[^a-z0-9]+", " ", low).strip()
        if len(norm_key) > 12 and norm_key in seen_norm_lines:
            continue
        seen_norm_lines.add(norm_key)
        if wants_one_sent and not ln.startswith("__CODE_BLOCK_"):
            first_sent = re.split(r"(?<=[.!?])\s+(?=[A-Z])", ln)[0].strip()
            if first_sent.endswith((".", "!", "?")) and len(first_sent) >= 12:
                raw_lines.append(first_sent)
                break
        raw_lines.append(ln)
        if len(raw_lines) >= 2 and any(
            ending in low
            for ending in (
                "please let me know!",
                "let me know if you need",
                "let me know if you have any",
                "feel free to ask!",
                "hope this helps!",
            )
        ):
            break

    # Drop any trailing dangling header (`:`), comma, unclosed parenthesis, dangling conjunction, or unclosed mid-sentence cut-off line
    while len(raw_lines) >= 1 and not raw_lines[-1].startswith("__CODE_BLOCK_") and (
        raw_lines[-1].endswith((":", ","))
        or bool(re.match(r"^\d+[.)]?$", raw_lines[-1].strip()))
        or (raw_lines[-1].count("(") > raw_lines[-1].count(")"))
        or re.search(
            r"\b(?:and|or|to|for|of|in|with|by|from|that|which|since|because|however|so|but|the|a|an|assume|assuming|clarify|whether|if|when|while|as)$",
            raw_lines[-1].lower(),
        )
        or (
            len(raw_lines) >= 2
            and not raw_lines[-1].endswith((".", "!", "?", ")", '"', "'", "`"))
            and not re.search(r"https?://\S+$", raw_lines[-1])
        )
    ):
        if len(raw_lines) == 1:
            # Never leave a single truncated fragment without sentence-ending punctuation
            if not raw_lines[0].endswith((".", "!", "?", ")", '"', "'", "`")) or bool(re.match(r"^\d+[.)]?$", raw_lines[0].strip())):
                raw_lines.pop()
            break
        raw_lines.pop()

    if not raw_lines:
        if "rewrite these notes into one clear" in user_prompt.lower() and "\n" in user_prompt:
            notes_part = user_prompt.split("\n", 1)[1].strip()
            cleaned_notes = re.sub(r";\s*", " ", notes_part).strip()
            if cleaned_notes:
                if not cleaned_notes.endswith((".", "!", "?")):
                    cleaned_notes += "."
                raw_lines = [cleaned_notes[0].upper() + cleaned_notes[1:]]
        if not raw_lines:
            if is_short_greeting_or_ack(user_prompt) or is_conversational_meta_or_confirmation(user_prompt):
                return ["Hello! How can I help you today?"]
            return ["Unable to process this request right now."]

    plans = []
    for line_idx, ln in enumerate(raw_lines):
        m_cb = re.match(r"^__CODE_BLOCK_(\d+)__$", ln)
        if m_cb:
            cb_idx = int(m_cb.group(1))
            if 0 <= cb_idx < len(code_blocks):
                cb_str = code_blocks[cb_idx]
                plans.append(("\n" + cb_str) if line_idx > 0 else cb_str)
                if len(plans) >= max_windows:
                    break
            continue
        # Split line into single-sentence / compact windows of <= max_bpe Kimi K3 tokens so K3LexicalPointerDecoder has 100% 1-to-1 positional alignment
        clauses = [c.strip() for c in re.split(r"(?<=[.!?;])\s+", ln) if c.strip()]
        if len(clauses) <= 1 and len(k3_tok.encode(ln)) <= max_bpe:
            line_wins = [ln]
        else:
            line_wins = []
            cur = ""
            for cl in clauses:
                cand = (cur + " " + cl).strip() if cur else cl
                if len(k3_tok.encode(cand)) <= min(26, max_bpe):
                    cur = cand
                else:
                    if cur:
                        line_wins.append(cur)
                    if len(k3_tok.encode(cl)) <= max_bpe:
                        cur = cl
                    else:
                        words = cl.split()
                        w_cur = ""
                        for w in words:
                            w_cand = (w_cur + " " + w).strip() if w_cur else w
                            if len(k3_tok.encode(w_cand)) <= max_bpe:
                                w_cur = w_cand
                            else:
                                if w_cur:
                                    line_wins.append(w_cur)
                                w_cur = w
                        cur = w_cur
            if cur:
                line_wins.append(cur)

        line_wins = [w for w in line_wins if len(w) >= 2]
        if not line_wins:
            continue
        # Never add a partial multi-window line if the whole line will not fit in `max_windows`
        if plans and len(plans) + len(line_wins) > max_windows:
            break
        for w_idx, w_str in enumerate(line_wins):
            # Prefix "\n" if this window starts a new line (line_idx > 0 and w_idx == 0)
            if line_idx > 0 and w_idx == 0:
                plans.append("\n" + w_str)
            else:
                plans.append(w_str)
            if len(plans) >= max_windows:
                break
        if len(plans) >= max_windows:
            break

    if plans:
        return plans
    if is_short_greeting_or_ack(user_prompt) or is_conversational_meta_or_confirmation(user_prompt):
        return ["Hello! How can I help you today?"]
    return ["Unable to process this request right now."]


def format_readable_response(raw_joined: str) -> str:
    """Formats a rendered response into clean, readable paragraphs, bullet lists, and fenced code blocks:
    - Preserves fenced code blocks (` ```python ... ``` `) with exact newlines and indentation.
    - Lines with 'Heading/Topic: Explanation' become bullet items ('• Heading/Topic: Explanation').
    - Long single-block paragraphs (4+ sentences) are split into 2-sentence readable paragraphs separated by blank lines.
    """
    saved_fenced = []

    def _stash_fenced(m):
        idx_f = len(saved_fenced)
        saved_fenced.append(m.group(0).strip())
        return f"\n__FENCED_CODE_{idx_f}__\n"

    working = re.sub(r"```[a-zA-Z0-9_+-]*\n[\s\S]*?```", _stash_fenced, raw_joined)
    lines = [ln.strip() for ln in working.splitlines() if ln.strip()]
    if not lines:
        return raw_joined.strip()

    formatted_blocks = []
    for idx, ln in enumerate(lines):
        m_fc = re.match(r"^__FENCED_CODE_(\d+)__$", ln)
        if m_fc:
            fc_i = int(m_fc.group(1))
            if 0 <= fc_i < len(saved_fenced):
                formatted_blocks.append(saved_fenced[fc_i])
            continue
        ln_clean = re.sub(r"^[•\-*]\s*", "", ln).strip()
        # Check if this line is a section header ending with ':' (e.g. 'Morning (6:00 AM - 8:00 AM):')
        if ln_clean.endswith(":") and len(ln_clean.split()) <= 10 and idx > 0:
            formatted_blocks.append(ln_clean)
            continue

        # Check if this line is a 'Title / Time-slot: Explanation' item
        m_item = re.match(r"^([^:]{2,95}):\s+(.+)$", ln_clean)
        if m_item and (idx > 0 or len(lines) > 1):
            label, body = m_item.group(1).strip(), m_item.group(2).strip()
            if len(label.split()) <= 16 and not label.lower().startswith(("http", "https", "here is", "here's", "note")):
                formatted_blocks.append(f"• {label}: {body}")
                continue

        # Ordinary prose paragraph: if it has >= 3 sentences, split into 2-sentence paragraphs for readability
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", ln_clean) if s.strip()]
        if len(sents) >= 3:
            for s_i in range(0, len(sents), 2):
                formatted_blocks.append(" ".join(sents[s_i : s_i + 2]))
        else:
            formatted_blocks.append(ln_clean)

    # Join consecutive bullet items with '\n' and paragraphs/code blocks with '\n\n'
    out_str = ""
    for i, blk in enumerate(formatted_blocks):
        if i == 0:
            out_str = blk
        else:
            prev_is_bullet = formatted_blocks[i - 1].startswith("• ")
            cur_is_bullet = blk.startswith("• ")
            if prev_is_bullet and cur_is_bullet:
                out_str += "\n" + blk
            else:
                out_str += "\n\n" + blk
    return out_str


def join_rendered_windows(txts, plans):
    """Joins rendered window strings `txts` using `\n` whenever the corresponding entry in `plans` starts a new line
    or begins with a schedule/block header (`Block`, `Session`, `Morning`, `Afternoon`, `Evening`, `Tip:`), then
    formats into clean paragraphs and bullet lists."""
    out = []
    for i, t in enumerate(txts):
        t_clean = t.strip()
        if not t_clean:
            continue
        if i == 0:
            out.append(t_clean)
        else:
            p_orig = plans[i] if i < len(plans) else ""
            is_newline = (
                p_orig.startswith("\n")
                or any(
                    t_clean.startswith(pfx)
                    for pfx in ("Block ", "Session ", "Step ", "Morning", "Afternoon", "Evening", "Night", "Tip:", "Note:")
                )
                or bool(re.match(r"^(?:\d{1,2}:\d{2}|\d+\.|[A-Z][A-Za-z0-9\s()\-]{2,55}:\s)", t_clean))
            )
            out.append(("\n" if is_newline else " ") + t_clean)
    return format_readable_response("".join(out))

GRAMMAR_CONNECTOR_WORDS = [
    "the", "The", "a", "A", "an", "An", "in", "In", "on", "On", "at", "At", "to", "To", "for", "For",
    "of", "by", "with", "from", "as", "into", "over", "after", "before", "during", "between", "through",
    "and", "or", "but", "so", "while", "when", "where", "which", "that", "who", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "it", "its", "this", "these",
    "they", "them", "their", "we", "our", "you", "your", "I", "my", "me",
    ".", ",", ";", ":", "!", "?", "'s", "'", "-", "(", ")", "/", "//", "://", "_", "=", "&", "#", "[", "]",
    "https", "http", "www", "com", "org", "io", "ai",
]


class K3FastChatEngine:
    def __init__(self, out_dir: str | None = None, dev_ncm="cuda:0", dev_ha="cuda:1"):
        t0 = time.perf_counter()
        if out_dir is None or not os.path.isdir(out_dir):
            out_dir = find_hybrid_ncm_ckpt_dir()
        self.dev = torch.device(resolve_device(dev_ncm))
        self.dev_ha = torch.device(resolve_device(dev_ha))
        self.k3_tok = K3Tokenizer()

        self.ha_model, self.ha_tok = load_hybrid_agent(
            "togethercomputer/HybridAgent-8B-Agentic", dtype=torch.bfloat16, device=self.dev_ha, require_hybridlink=True
        )
        self.ha_tok.padding_side = "right"
        self.lat_close_id = special_token_id(self.ha_tok, EXIT_ANCHOR)

        vq_ckpt = os.path.join(out_dir, "checkpoint-concept-vector-quantizer-adapted.pt")
        if not os.path.exists(vq_ckpt):
            vq_ckpt = os.path.join(out_dir, "k3_vqvae_adapted.pt")
        ncm_ckpt = os.path.join(out_dir, "checkpoint-next-concept-generator-base.pt")
        if not os.path.exists(ncm_ckpt):
            ncm_ckpt = os.path.join(out_dir, "e6_last.pt")
        _, self.tk, self.vqvae, self.net, src = load_k3_native_models(self.dev, vq_ckpt, ncm_ckpt)

        direct_ckpt = os.path.join(out_dir, "checkpoint-next-concept-direct-decoder.pt")
        if not os.path.exists(direct_ckpt):
            direct_ckpt = os.path.join(out_dir, "e8_k3_direct.pt")
        e8_sd = torch.load(direct_ckpt, map_location="cpu", weights_only=True)
        self.vqvae.load_state_dict(e8_sd["vqvae"])
        self.net.load_state_dict(e8_sd["net"])
        self.vqvae.eval(); self.net.eval()

        self.cbridge = ComboBridge(D_HA, D_VQ).to(self.dev)
        self.cbridge.load_state_dict(src["combo_bridge"])
        self.cbridge.eval()

        self.copy_dec = K3LexicalPointerDecoder(d=D_VQ, n_layer=3, n_head=6).to(self.dev)
        self.copy_dec.load_state_dict(e8_sd["ptr_dec"])
        self.copy_dec.eval()
        self.restricted_mask = e8_sd["restricted_mask"].to(self.dev)

        # Build grammatical connector token set in Kimi K3's 163,840 vocabulary
        self.grammar_ids = {K3_EOS}
        self.repeatable_grammar_ids = {K3_EOS}
        for gw in GRAMMAR_CONNECTOR_WORDS:
            for variant in (gw, " " + gw):
                for tid in self.k3_tok.encode(variant):
                    if 0 <= tid < 163584:
                        self.grammar_ids.add(tid)
                        if gw.lower() in (
                            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "by", "with", "from",
                            "as", "is", "are", "was", "were", ".", ",", ";", ":", "-", "/", "//", "://", "_", "=", "&", "(", ")", "[", "]",
                        ):
                            self.repeatable_grammar_ids.add(tid)

        levels = list(self.vqvae.vq.levels)
        self.sched = [1.0 - 0.6 * (k / (len(levels) - 1)) for k in range(len(levels))]
        self.setup_s = round(time.perf_counter() - t0, 2)
        self.history = []  # List of {"user": str, "reply": str, "plans": list[str]} for multi-turn context memory

        # Warmup GPU kernels
        self._warmup()

    def clear_history(self):
        self.history.clear()

    def _warmup(self):
        p_warm = ["The Harbourside Council voted 7 to 2 on Tuesday; approved 2.4 million pounds"]
        r_w, m_w = encode_batch_rh(self.ha_model, self.ha_tok, p_warm, P_RH, self.dev_ha)
        zr_w = self.cbridge(r_w.to(self.dev)).float()
        mw = (m_w.to(self.dev) != 0)
        pk_w = torch.tensor([self.k3_tok.encode(p_warm[0])[:P_BPE] + [K3_PAD] * (P_BPE - len(self.k3_tok.encode(p_warm[0])[:P_BPE]))], dtype=torch.long, device=self.dev)
        zb_w = self.vqvae.encoder.wte(pk_w)
        zp_w, pm_w = pack_combo_prefix(zr_w, mw, zb_w, (pk_w == K3_PAD))
        with torch.no_grad():
            g_w = generate_from_zpref(self.net, self.tk, zp_w, pm_w, self.sched)
            zh_w, _ = self.vqvae.vq.idx_to_zhats(g_w, return_zlevels=True)
            _ = self.copy_dec(self.vqvae.decoder, zh_w[-1], zr_w, mw, zb_w, pk_w)
        cuda_sync(self.dev, self.dev_ha)

    def _dec_k3(self, r):
        r = list(r)
        if K3_EOS in r:
            r = r[:r.index(K3_EOS)]
        clean_ids = []
        for t in r:
            ti = int(t)
            if 0 <= ti < 163584 and (not clean_ids or ti != clean_ids[-1]):
                clean_ids.append(ti)
        return self.k3_tok.decode(clean_ids), clean_ids

    @torch.no_grad()
    def plan_with_latent(self, user_prompt, history=None, min_lat=8, max_lat=14, max_new=260):
        backbone = self.ha_model.backbone
        embed_w = backbone.get_input_embeddings().weight
        msgs = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        if history:
            for h in history[-6:]:
                msgs.append({"role": "user", "content": h["user"]})
                msgs.append({"role": "assistant", "content": "W1: " + h["reply"].replace("\n", "\nW2: ", 1)})
        msgs.append({"role": "user", "content": CHAT_PLAN_PROMPT.format(prompt=user_prompt.strip())})

        base_chat = self.ha_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if base_chat.rstrip("\n").endswith(THINK_OPEN):
            prefix_str = base_chat + "Let me think carefully and reply directly to the user." + ENTRY_ANCHOR
        else:
            prefix_str = base_chat + THINK_OPEN + "\nLet me think carefully and reply directly to the user." + ENTRY_ANCHOR

        inp_ids = self.ha_tok(prefix_str, return_tensors="pt").input_ids.to(self.dev_ha)
        out = backbone(input_ids=inp_ids, use_cache=True, output_hidden_states=True, return_dict=True)
        past_kv = out.past_key_values
        last_h = out.hidden_states[-1][:, -1, :]
        next_logits = out.logits[:, -1, :]

        lat_steps = 0
        exit_reason = "max_steps"
        for step in range(max_lat):
            r_h = self.ha_model.run_hybridlink(last_h, output_dtype=embed_w.dtype)
            arg_id = int(next_logits.argmax(dim=-1).item())
            if step >= min_lat and arg_id == self.lat_close_id:
                exit_reason = "anchor"
                break
            lat_steps += 1
            out = backbone(inputs_embeds=r_h.unsqueeze(1), past_key_values=past_kv, use_cache=True, output_hidden_states=True, return_dict=True)
            past_kv = out.past_key_values
            last_h = out.hidden_states[-1][:, -1, :]
            next_logits = out.logits[:, -1, :]

        bridge_str = EXIT_ANCHOR + " Now I will write my direct response to the user." + THINK_CLOSE + "\nW1:"
        bridge_ids = self.ha_tok(bridge_str, add_special_tokens=False, return_tensors="pt").input_ids.to(self.dev_ha)
        out = backbone(input_ids=bridge_ids, past_key_values=past_kv, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

        gen_ids = []
        eos_ids = {self.ha_tok.eos_token_id}
        is_struct = is_structured_or_multi_part_prompt(user_prompt)
        for _ in range(max_new):
            nxt = int(next_logits.argmax(dim=-1).item())
            if nxt in eos_ids:
                break
            gen_ids.append(nxt)
            txt_so_far = self.ha_tok.decode(gen_ids, skip_special_tokens=False)
            if THINK_CLOSE in txt_so_far or THINK_OPEN in txt_so_far or "</think>" in txt_so_far:
                break
            if txt_so_far.endswith("\n"):
                lines = [ln.strip() for ln in txt_so_far.splitlines() if ln.strip()]
                if lines:
                    last_ln = lines[-1].lower()
                    introduces_list = last_ln.endswith(":") or any(
                        kw in last_ln for kw in ("here's", "here is", "sample timetable", "sample study", "schedule", "below", "following")
                    )
                    if not is_struct and not introduces_list:
                        if len(lines) == 1 and lines[0].endswith((".", "!", "?")):
                            break
                        if len(lines) >= 2 and len(lines[1]) > 15:
                            break
                    elif len(lines) >= 10 and not last_ln.endswith(":"):
                        break
            out = backbone(input_ids=torch.tensor([[nxt]], device=self.dev_ha), past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]

        raw = "W1: " + self.ha_tok.decode(gen_ids, skip_special_tokens=False).strip()
        plans = parse_and_split_plans(raw, user_prompt, self.k3_tok, max_bpe=P_BPE - 4, max_windows=12)
        return plans, lat_steps, exit_reason

    def _build_window_vocab_mask(self, plan_str):
        """Builds a strict boolean mask `[V_K3]` that masks out (`True`) any token NOT in `plan_str`
        and NOT in `self.grammar_ids`, preventing VQ-VAE from ever hallucinating memorized words
        (like 'France', 'Paris', 'Bristol', 'lithium') that are not in the current window's plan!"""
        allowed = set(self.grammar_ids)
        allowed.update(self.k3_tok.encode(plan_str))
        for w in re.findall(r"[A-Za-z0-9.,:!?'£$€%/_\-+=&#()\[\]]+", plan_str):
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.lower(), " " + w.lower()):
                allowed.update(self.k3_tok.encode(v))
        mask = torch.ones(V_K3, dtype=torch.bool, device=self.dev)
        t_allow = torch.tensor([x for x in allowed if 0 <= x < V_K3], dtype=torch.long, device=self.dev)
        mask[t_allow] = False
        return mask

    def _dec_k3_aligned(self, row_logits, plan_ids):
        """Decodes a 64-position logit tensor `[T, V_K3]` while:
        1. Enforcing 2-gram subword continuation for multi-subword words/times/decimals/URLs in `plan_ids`.
        2. Capping each content token's emission count to its occurrence count in `plan_ids` (preventing
           duplicate words like `Paris. Paris.` or `45 beds and 45 beds`).
        3. Stopping cleanly at `K3_EOS` once all multiplicity counts of the window's planned content tokens are covered.
        """
        cont_map = {}
        valid_p = [int(x) for x in plan_ids if 0 <= int(x) < 163584]
        for a, b in zip(valid_p[:-1], valid_p[1:]):
            bs = self.k3_tok.decode([b], skip_special_tokens=False)
            if bs and (not bs.startswith(" ")) and (bs[0].isalnum() or bs[0] in (":", ".", "'", "/", "-", "_", "=", "?", "&", "#")):
                cont_map.setdefault(a, set()).add(b)

        max_counts = collections.Counter(valid_p)
        content_set = {tid for tid in valid_p if tid not in self.repeatable_grammar_ids}
        emitted_counts = collections.Counter()
        has_semi = (26 in valid_p)

        T = row_logits.size(0)
        max_len = min(T, int(len(valid_p) * 1.35) + 4) if has_semi else min(T, len(valid_p))
        clean_ids = []
        prev_tid = -1
        prev_word = ""
        for t in range(max_len):
            lg = row_logits[t].clone()
            if prev_tid >= 0:
                # Only penalize immediate exact token repetition if the plan itself doesn't repeat that token
                if emitted_counts[prev_tid] >= max_counts.get(prev_tid, 1):
                    lg[prev_tid] -= 30.0
                if prev_tid in cont_map:
                    for nxt_b in cont_map[prev_tid]:
                        if emitted_counts[nxt_b] < max_counts.get(nxt_b, 1):
                            lg[nxt_b] += 12.0
            # Penalize content tokens that have already been emitted as many times as they appear in `plan_ids`
            for tid, cnt in emitted_counts.items():
                if tid not in self.repeatable_grammar_ids and cnt >= max_counts.get(tid, 1):
                    lg[tid] -= 30.0

            # Require ALL multiplicity occurrences of every content token in `valid_p` before boosting K3_EOS
            all_covered = bool(content_set) and all(
                emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set
            )
            if all_covered and t >= len(valid_p) - 1:
                lg[K3_EOS] += 18.0

            tid = int(lg.argmax().item())
            if tid == K3_EOS:
                break
            if 0 <= tid < 163584:
                ws = self.k3_tok.decode([tid], skip_special_tokens=False).strip().lower()
                if ws and ws == prev_word and len(ws) >= 2 and emitted_counts[tid] >= max_counts.get(tid, 1):
                    lg[tid] -= 30.0
                    tid = int(lg.argmax().item())
                    if tid == K3_EOS:
                        break
                    ws = self.k3_tok.decode([tid], skip_special_tokens=False).strip().lower()
                clean_ids.append(tid)
                emitted_counts[tid] += 1
                prev_tid = tid
                if ws:
                    prev_word = ws
                # Only stop on sentence-ending punctuation after ALL occurrences of all content tokens (e.g. both '5's in '5.5') have been emitted!
                if bool(content_set) and all(emitted_counts[c_id] >= max_counts.get(c_id, 1) for c_id in content_set):
                    raw_piece = self.k3_tok.decode([tid], skip_special_tokens=False).rstrip()
                    if raw_piece.endswith((".", "!", "?")) and t >= len(valid_p) - 1:
                        break
        return self.k3_tok.decode(clean_ids), clean_ids

    @torch.no_grad()
    def render_ncm(self, plans, prompt_text="", cands=1):
        clean_plans = [p.lstrip("\n").strip() for p in plans]
        rh_p, msk_p = encode_batch_rh(self.ha_model, self.ha_tok, clean_plans, P_RH, self.dev_ha)
        z_rh = self.cbridge(rh_p.to(self.dev)).float()
        msk_r = (msk_p.to(self.dev) != 0)
        p_kpad = torch.tensor(
            [self.k3_tok.encode(p)[:P_BPE] + [K3_PAD] * (P_BPE - len(self.k3_tok.encode(p)[:P_BPE])) for p in clean_plans],
            dtype=torch.long, device=self.dev
        )
        z_bpe = self.vqvae.encoder.wte(p_kpad)
        z_pref, pmask = pack_combo_prefix(z_rh, msk_r, z_bpe, (p_kpad == K3_PAD))

        W = len(clean_plans)
        win_masks = [self._build_window_vocab_mask(p) for p in clean_plans]

        if cands <= 1:
            torch.manual_seed(42)
            g1 = generate_from_zpref(self.net, self.tk, z_pref, pmask, self.sched)
            zh1, _ = self.vqvae.vq.idx_to_zhats(g1, return_zlevels=True)
            lg1 = self.copy_dec(self.vqvae.decoder, zh1[-1], z_rh, msk_r, z_bpe, p_kpad)
            txts, ids = [], []
            for w_i in range(W):
                lg1[w_i, :, win_masks[w_i]] -= 80.0
                ts, is_ = self._dec_k3_aligned(lg1[w_i], p_kpad[w_i].tolist())
                txts.append(ts.strip()); ids.extend(is_)
            return join_rendered_windows(txts, plans), ids

        # Batched MBR over `cands` candidates (all `cands * W` rows in a single 11-pass call!)
        z_pref_k = z_pref.repeat(cands, 1, 1)
        pmask_k = pmask.repeat(cands, 1)
        z_rh_k = z_rh.repeat(cands, 1, 1)
        msk_r_k = msk_r.repeat(cands, 1)
        z_bpe_k = z_bpe.repeat(cands, 1, 1)
        p_kpad_k = p_kpad.repeat(cands, 1)
        torch.manual_seed(42)
        gk = generate_from_zpref(self.net, self.tk, z_pref_k, pmask_k, self.sched)
        zhk, _ = self.vqvae.vq.idx_to_zhats(gk, return_zlevels=True)
        lgk = self.copy_dec(self.vqvae.decoder, zhk[-1], z_rh_k, msk_r_k, z_bpe_k, p_kpad_k)

        cands_t = [[] for _ in range(W)]
        cands_i = [[] for _ in range(W)]
        for k in range(cands):
            for w_i in range(W):
                row_idx = k * W + w_i
                lgk[row_idx, :, win_masks[w_i]] -= 80.0
                ts, is_ = self._dec_k3_aligned(lgk[row_idx], p_kpad[w_i].tolist())
                cands_t[w_i].append(ts.strip()); cands_i[w_i].append(is_)
        out_txts, out_ids = [], []
        for w_i in range(W):
            sc = [sum(jacc(cands_t[w_i][k], cands_t[w_i][j]) for j in range(cands) if j != k) for k in range(cands)]
            bk = max(range(cands), key=lambda k: sc[k])
            out_txts.append(cands_t[w_i][bk]); out_ids.extend(cands_i[w_i][bk])
        return join_rendered_windows(out_txts, plans), out_ids

    def chat_reply(self, user_prompt, cands=8, remember=False):
        cuda_sync(self.dev, self.dev_ha)
        t0 = time.perf_counter()
        active_hist = self.history if remember else None
        plans, lat_steps, exit_reason = self.plan_with_latent(user_prompt, history=active_hist)
        cuda_sync(self.dev_ha)
        t_plan_ms = (time.perf_counter() - t0) * 1000.0

        hist_ctx = ""
        if active_hist:
            hist_ctx = " ".join(f"{h['user']} {h['reply']}" for h in active_hist[-6:])
        full_prompt_ctx = (hist_ctx + " " + user_prompt).strip()

        t1 = time.perf_counter()
        text_1cand, ids_1cand = self.render_ncm(plans, prompt_text=full_prompt_ctx, cands=1)
        cuda_sync(self.dev, self.dev_ha)
        t_ncm_1cand_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        text_mbr, ids_mbr = self.render_ncm(plans, prompt_text=full_prompt_ctx, cands=cands)
        cuda_sync(self.dev, self.dev_ha)
        t_ncm_mbr_ms = (time.perf_counter() - t2) * 1000.0

        if remember:
            self.history.append({
                "user": user_prompt.strip(),
                "reply": text_mbr,
                "plans": plans,
            })

        return {
            "prompt": user_prompt.strip(),
            "turn": len(self.history) if remember else 1,
            "latent_steps": lat_steps,
            "exit_reason": exit_reason,
            "plans": plans,
            "plan_lat_ms": round(t_plan_ms, 1),
            "ncm_1cand_ms": round(t_ncm_1cand_ms, 1),
            "ncm_mbr8_ms": round(t_ncm_mbr_ms, 1),
            "total_1cand_ms": round(t_plan_ms + t_ncm_1cand_ms, 1),
            "total_mbr8_ms": round(t_plan_ms + t_ncm_mbr_ms, 1),
            "n_k3_tokens_1cand": len(ids_1cand),
            "n_k3_tokens_mbr8": len(ids_mbr),
            "reply_1cand": text_1cand,
            "reply_mbr8": text_mbr,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None, help="Single user prompt string to run in chat mode")
    ap.add_argument("--interactive", action="store_true", help="Run interactive multi-turn REPL chat mode with context memory")
    ap.add_argument("--test-multiturn", action="store_true", help="Run a 4-turn conversation test verifying multi-turn context memory")
    ap.add_argument("--out", default="/home/usergpu/merge_lf/out/e6_k3_native/chat_mode_results.json")
    a = ap.parse_args()

    eng = K3FastChatEngine()
    print(f"K3FastChatEngine ready (one-time setup {eng.setup_s}s; 93-layer verification OFF; DSpark OFF; Multi-turn memory ON)", flush=True)

    if a.test_multiturn:
        eng.clear_history()
        turns = [
            "the captial of france is",
            "What famous iron tower is located there, and who designed it?",
            "Now imagine St Mary's Hospital in Bristol opened a new 18 million pound children's wing on Friday led by Dr Elena Vance, adding 45 beds and 4 operating theatres. Summarize what happened.",
            "Who led that hospital project, how many beds and operating theatres did it add, and in which city was it opened?",
        ]
        mt_res = []
        for idx, u_turn in enumerate(turns, 1):
            r = eng.chat_reply(u_turn, cands=8, remember=True)
            mt_res.append(r)
            print(f"\n================ [Turn {idx}] ================", flush=True)
            print(f"USER   : {r['prompt']}", flush=True)
            print(f"PLAN   : {' || '.join(r['plans'])} ({r['plan_lat_ms']} ms)", flush=True)
            print(f"TIMING : 1-cand = {r['total_1cand_ms']/1000:.2f}s (NCM {r['ncm_1cand_ms']} ms) | MBR8 = {r['total_mbr8_ms']/1000:.2f}s (NCM {r['ncm_mbr8_ms']} ms)", flush=True)
            print(f"REPLY  : {r['reply_mbr8']}", flush=True)
        json.dump(mt_res, open("/home/usergpu/merge_lf/out/e6_k3_native/multiturn_chat_proof.json", "w"), indent=2)
        return

    if a.interactive:
        print("\nInteractive Multi-Turn K3FastChatEngine ready! Type any prompt (or 'reset' to clear memory, 'exit' to stop).\n", flush=True)
        while True:
            try:
                user_in = input(f"[Turn {len(eng.history)+1}] PROMPT> ").strip()
            except EOFError:
                break
            if not user_in or user_in.lower() in ("exit", "quit", "q"):
                break
            if user_in.lower() in ("reset", "clear"):
                eng.clear_history()
                print("Conversation memory cleared.\n", flush=True)
                continue
            r = eng.chat_reply(user_in, cands=8, remember=True)
            print(f"LATENT: {r['latent_steps']} steps ({r['exit_reason']}) | PLAN ({r['plan_lat_ms']} ms): {' || '.join(r['plans'])}", flush=True)
            print(f"TIMING: 1-cand NCM = {r['ncm_1cand_ms']} ms (Total {r['total_1cand_ms']/1000:.2f}s) | MBR8 NCM = {r['ncm_mbr8_ms']} ms (Total {r['total_mbr8_ms']/1000:.2f}s)", flush=True)
            print(f"OUTPUT (1-cand, {r['n_k3_tokens_1cand']} K3 tokens): {r['reply_1cand']}", flush=True)
            print(f"OUTPUT (MBR8,   {r['n_k3_tokens_mbr8']} K3 tokens): {r['reply_mbr8']}\n", flush=True)
        return

    if a.prompt:
        prompts = [("custom", a.prompt)]
    else:
        prompts = [
            ("p1", open("/home/usergpu/k3ncm/p1.txt", encoding="utf-8").read()),
            ("p2", open("/home/usergpu/k3ncm/p2.txt", encoding="utf-8").read()),
            ("p3", open("/home/usergpu/k3ncm/p3.txt", encoding="utf-8").read()),
            ("p4", open("/home/usergpu/k3ncm/p4.txt", encoding="utf-8").read()),
            ("p5", open("/home/usergpu/k3ncm/p5.txt", encoding="utf-8").read()),
            ("p6_new_hospital", "Write a news report of about 90 words using these facts: St Mary's Hospital opened a new 18 million pound children's wing on Friday in Bristol. Dr Elena Vance led the 3-year project, which adds 45 beds and 4 operating theatres. Give a one-sentence Summary at the end."),
            ("p7_new_battery", "Explain in about 90 words how a lithium-ion battery works during charging and discharging, and why it degrades over time. End with a one-sentence Summary."),
        ]

    results = {}
    for name, pr in prompts:
        r = eng.chat_reply(pr, cands=8)
        results[name] = r
        print(f"\n================ [{name}] ================", flush=True)
        print(f"PROMPT: {r['prompt']}", flush=True)
        print(f"LATENT: {r['latent_steps']} steps ({r['exit_reason']}) | PLAN ({r['plan_lat_ms']} ms): {' || '.join(r['plans'])}", flush=True)
        print(f"TIMING: 1-cand NCM = {r['ncm_1cand_ms']} ms (Total {r['total_1cand_ms']} ms = {r['total_1cand_ms']/1000:.2f}s) | MBR8 NCM = {r['ncm_mbr8_ms']} ms (Total {r['total_mbr8_ms']} ms = {r['total_mbr8_ms']/1000:.2f}s)", flush=True)
        print(f"OUTPUT (MBR8, {r['n_k3_tokens_mbr8']} Kimi K3 tokens):\n{r['reply_mbr8']}", flush=True)

    json.dump(results, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nSaved {len(results)} chat mode runs to {a.out}", flush=True)


if __name__ == "__main__":
    main()
