"""The Clover cycle after an answer: it starts when the Outcome is not right, and the human says so.

Context -> Direction -> Execution complete as one normal answer. The Outcome is what the user sees: when they say it
is not right ("try again", "that's wrong", "not what I expected", the Try again button, or the same question asked
again), Clover returns to Context with what that answer and the user's words now show:
- evidence the model cannot silently redefine (AGENTS.md): the arithmetic it wrote is recomputed, the form the
  question asks for is checked, code goes through the compiler layer, and a one-value answer (a number or an option
  letter) is solved twice more, independently, and a number once more by a Python program Clover runs (counted only
  when the program calculates, not when it writes down a value and prints it);
- Context it did not have: when facts can help (not for code, creative writing, material given in the question or
  self-contained arithmetic), Clover searches the web and reads the best pages (web_search.research), unless
  Clover's own records already answer the question. Only results that match the question are used, and they reach
  one of the independent solutions and the rewrite, not every solution: a misleading page moves one vote at most
  (one page made two solutions overturn a right answer, measured);
- then answers again from that evidence: concrete defects are fixed; for a one-value answer the majority of the
  independent solutions decides: another value replaces the answer only when at least two solutions give it and more
  back it than back the earlier answer (asking the model to reconsider did not change its mind in 4 of 4 measured
  cases); if they back the earlier answer, Clover says so and asks what the user expected; open answers are rewritten
  with the user's words as Direction.
Growth: what the correction taught is staged as a lesson the user keeps or discards (memory_tree/staging).
No answer key is used anywhere. Always-on checking was measured (ladder steps 8-9) and set aside by the user.
"""
import ast
import json
import math
import operator
import re

FIX = ("The user says your answer above is not right ({said}). Clover checked it and found:\n{findings}\nReturn to "
       "Context: re-read the question, find the cause of each problem and fix it. Reply with the complete corrected "
       "answer, in the form the question asks for, without comments about the check.")
REDO = ("The user says your answer above is not right or not what they expected ({said}). Return to Context: re-read "
        "the question and their words, work out what was missed or misread, and write a better answer. Reply with the "
        "complete new answer only.")
# no details given: a creative answer gets another version; any other answer is checked again, not changed for its own
# sake (told to "try a clearly different approach", a model can trade a right answer for a wrong one)
ANOTHER = ("The user wants a different answer than the one above ({said}). Write a clearly different version that still "
           "does what the question asks. Reply with the new answer only.")
RECHECK = ("The user says your answer above is not right ({said}) but did not say what is wrong. Return to Context: "
           "re-read the question, check each fact, step and number in your answer, and correct whatever is wrong. If "
           "you find nothing wrong, start with one sentence saying what you checked, then give the complete answer "
           "again (all of its code, if it has code). Reply with the complete answer only.")
_CREATIVE = re.compile(r"\b(?:write|compose|draft|create|make\s+up|come\s+up\s+with|suggest|generate|brainstorm|describe|"
                       r"rewrite|rephrase|paraphrase|invent|imagine|poem|haiku|story|joke|slogan|tagline|lyrics|caption|"
                       r"names?\s+for|title\s+for)\b", re.I)
# code that is "not right" has a bug to find, not a style to change
_CODE = re.compile(r"\b(?:code|function|program|script|class|method|query|regex|sql|python|javascript|typescript|"
                   r"java|rust|c\+\+|html|css|bash|powershell|algorithm|def|return)\b|```", re.I)
# the question brings its own material: a web search adds nothing
_GIVEN = re.compile(r"\b(?:summari[sz]e|translate|rewrite|rephrase|paraphrase|reconstruct|proofread|sudoku|"
                    r"(?:this|the following|the)\s+(?:passage|text|article|paragraph|sentence)\s+(?:below|above|:))", re.I)
_FORM = re.compile(r"\s*(?:Solve step by step and end your response with #### <number>\.?|"
                   r"State the correct option letter[^\n]*|Reply with (?:just|only) the number\.?)", re.I)
# numbers a word problem gives ("one" is left out: it usually means "a"), and quantities only the world can give
_NUMWORD = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|"
                      r"hundred|thousand|half|twice|double|triple|dozen)\b", re.I)
_FACT = re.compile(r"\b(?:typical(?:ly)?|speed\s+of\s+(?:sound|light)|(?:boiling|melting|freezing)\s+point|"
                   r"(?:from|to)\s+the\s+(?:sun|earth|moon)|population\s+of|capital\s+of|"
                   r"(?:mass|height|diameter|radius|age|area)\s+of\s+(?:the\s+)?[A-Z])", re.I)
# The user says the last answer is not right: at the start of the message (after a short lead-in such as "no," or "I
# think"); a bare word only when it stands alone. A false match would answer a new question with a re-check of the old
# one, so matching is narrow; the Try again button is the exact signal.
_END = r"\s*(?:[.,!?;:]|$)"
_RETRY_ASK = re.compile(
    r"^\W*(?:(?:no|nope|hmm+|um+|uh+|oh|sorry|but|well|ok(?:ay)?|actually|wait|still|hey|clover|i\s+think|"
    r"i\s+believe|i\s+feel|i'?m\s+(?:pretty\s+|quite\s+|fairly\s+)?sure)\W+){0,4}(?:"
    r"(?:please\s+)?(?:try\s+(?:it\s+|that\s+|this\s+)?again|retry|do\s+it\s+again|one\s+more\s+time)\b|"
    r"redo(?:\s+(?:it|that|this)\b|" + _END + r")|"
    r"(?:that'?s|that\s+is|this\s+is|it'?s|it\s+is|your\s+answer\s+is|the\s+answer\s+is|you'?re|you\s+are)\s+"
    r"(?:wrong|incorrect|not\s+(?:quite\s+)?(?:right|correct))\b|"
    r"(?:wrong(?:\s+answer)?|incorrect|nope|not\s+(?:it|right|correct))" + _END + r"|"
    r"(?:that'?s\s+|this\s+is\s+|it'?s\s+)?not\s+what\s+i\s+(?:expected|wanted|asked(?:\s+for)?|meant)\b|"
    r"i\s+don'?t\s+think\s+(?:that'?s|this\s+is|it'?s|that\s+is|it\s+is)\s+(?:right|correct)\b|"
    r"are\s+you\s+sure\b|"
    r"(?:check\s+(?:it\s+|that\s+|this\s+)?again|double[\s-]?check|re-?check)"
    r"(?:\s+(?:it|that|this|again|the\s+answer|your\s+(?:answer|work|math|maths|calculation|steps)))?"
    r"(?:\s+please)?" + _END + r")", re.I)
_FILLER = {"a", "an", "the", "please", "pls", "again", "so", "ok", "okay", "hey", "clover", "now", "just", "i", "me",
           "tell", "can", "could", "would", "you", "ask", "asking", "question", "same", "once", "more"}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT = r"(?:\s+per\s+[A-Za-z]+|\s+(?!x\b|and\b|of\b|is\b|are\b|to\b|or\b|than\b)[A-Za-z][A-Za-z\-]*(?:/[A-Za-z]+)?|/[A-Za-z]+){0,2}"
_TERM = rf"\(?\s*-?(?:{_NUM}){_UNIT}\s*\)?"
_ARITH = re.compile(rf"(?<![\w.)\]%])({_TERM}(?:\s*(?:[+\-\u00d7*/\u00f7^]|x(?=\s*\(?\s*\d))\s*{_TERM})+)\s*(=|\u2248)\s*(-?(?:{_NUM}))(?![\w^(]|\.\d|\s*[\^*/\u00d7\u00f7%])")
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|e\.g|i\.e|No|Inc|Ltd|Co|U\.S|U\.K|approx|Fig)\.$|\b[A-Z]\.$")


def normalize(text):
    """Plain arithmetic from the way the model writes it (Unicode spaces and dashes, LaTeX, $ signs, bold marks)."""
    t = (text or "").replace("\u202f", " ").replace("\u00a0", " ").replace("\u2009", " ").replace("\u2212", "-")
    t = re.sub(r"(?<=[\d\s])\u2013(?=[\s\d])", "-", t)
    t = re.sub(r"\\text\{[^{}]*\}|\\mathrm\{[^{}]*\}", " ", t)
    t = re.sub(r"\\[,;! ]", " ", t)
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = t.replace("\\times", "\u00d7").replace("\\cdot", "\u00d7").replace("\\div", "\u00f7")
    t = re.sub(r"\\[()\[\]]|\\\$|\$|\*\*", " ", t)
    return re.sub(r"[ \t]+", " ", t)


def _num(s):
    return float(s.replace(",", "").replace("\u2212", "-"))


def _eval(expr):
    """Value of + - * / ^ over plain numbers, or None (anything else, huge powers, division by zero)."""
    e = re.sub(r"(?<=[\d)])\s+(?!x\b|per\b)[A-Za-z][A-Za-z\-]*(?:/[A-Za-z]+)?|/[A-Za-z]+|\s+per\s+[A-Za-z]+", "", expr)
    e = e.replace("\u00d7", "*").replace("\u00f7", "/").replace("^", "**")
    e = re.sub(r"(?<=\d)\s*x\s*(?=[\d(])", "*", e)
    e = re.sub(r"(?<=\d),(?=\d{3})", "", e).strip()
    try:
        tree = ast.parse(e, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = ev(n.operand)
            return None if v is None else -v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if a is None or b is None:
                return None
            if isinstance(n.op, ast.Pow):
                return a ** b if abs(a) <= 1e6 and 0 <= b <= 64 and float(b).is_integer() else None
            if isinstance(n.op, ast.Div) and b == 0:
                return None
            fn = _BINOPS.get(type(n.op))
            return fn(a, b) if fn else None
        return None
    v = ev(tree)
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def arithmetic_findings(answer):
    """'a op b = c' statements in the answer whose c is wrong (allowing the rounding the answer shows)."""
    out = []
    for m in _ARITH.finditer(normalize(answer)):
        expr, sign, rhs = m.group(1), m.group(2), m.group(3)
        true = _eval(expr)
        if true is None:
            continue
        shown = _num(rhs)
        decimals = len(rhs.split(".")[1]) if "." in rhs else 0
        diff = abs(true - shown)
        if sign == "\u2248":
            bad = diff > max(0.5 * 10 ** -decimals, abs(true) * 0.01)
        elif float(true).is_integer():
            bad = diff > 1e-9 * max(1.0, abs(true))
        else:  # a rounded result is fine only if it is the value rounded to the decimals shown
            bad = diff >= 0.5 * 10 ** -decimals - 1e-12
            whole = all("." not in n for n in re.findall(_NUM, expr))
            if bad and whole and re.search(r"[/\u00f7]", expr) and shown == math.floor(true):
                bad = False  # whole-number division in a counting step (100 / 125 = 0 trailing zeros)
        if bad:
            good = f"{true:,.6g}" if not float(true).is_integer() else f"{int(true):,}"
            out.append(f"Arithmetic: you wrote {m.group(0).strip()}, but {expr.strip()} = {good}.")
    return out[:4]


def _sentences(text):
    t = re.sub(r"```[\s\S]*?```", " ", text)
    t = re.sub(r"[*_#>`]+", "", t).strip()
    parts, cur = [], ""
    for piece in re.split(r"(?<=[.!?])\s+", t):
        cur = (cur + " " + piece).strip()
        if not _ABBREV.search(cur):
            parts.append(cur)
            cur = ""
    if cur:
        parts.append(cur)
    return [p for p in parts if re.search(r"\w", p)]


def _json_candidate(answer):
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", answer)
    if m:
        return m.group(1).strip()
    starts = [i for i in (answer.find("{"), answer.find("[")) if i >= 0]
    if not starts:
        return None
    s = min(starts)
    end = max(answer.rfind("}"), answer.rfind("]"))
    return answer[s:end + 1] if end > s else answer[s:]


_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def format_findings(question, answer):
    """Checks of the form the question explicitly asks for."""
    q, a, out = question or "", answer or "", []
    if re.search(r"\b(?:in|as|with)\s+(?:one|a\s+single)\s+(?:clear\s+|short\s+|concise\s+)?sentence\b|"
                 r"\bone\s+(?:clear\s+|short\s+|grammatical\s+)?sentence\b", q, re.I):
        n = len(_sentences(a))
        if n > 1:
            out.append(f"Form: the question asks for one sentence; your answer has {n}.")
    m = re.search(r"\b(\d+|" + "|".join(_WORDNUM) + r")\s+(?:bullet(?:\s+point)?s|bullets)\b", q, re.I)
    if m:
        want = int(m.group(1)) if m.group(1).isdigit() else _WORDNUM[m.group(1).lower()]
        got = len(re.findall(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", a, re.M))
        if got != want:
            out.append(f"Form: the question asks for {want} bullets; your answer has {got}.")
    m = re.search(r"\b(?:in\s+)?(under|fewer\s+than|at\s+most|no\s+more\s+than|up\s+to)\s+(\d+)\s+words\b", q, re.I)
    if m:
        limit = int(m.group(2)) - (1 if m.group(1).lower().startswith(("under", "fewer")) else 0)
        words = len(re.findall(r"\b\w[\w'\-]*\b", re.sub(r"```[\s\S]*?```", " ", a)))
        if words > limit:
            out.append(f"Form: the question allows at most {limit} words; your answer has {words}.")
    if re.search(r"\b(?:in|as|valid|return|output|respond\s+with|give)\s+(?:a\s+|an\s+|the\s+)?(?:valid\s+)?json\b", q, re.I):
        cand = _json_candidate(a)
        if cand is None:
            out.append("Form: the question asks for JSON; your answer contains none.")
        else:
            try:
                json.loads(cand)
            except ValueError as exc:
                out.append(f"Form: the JSON in your answer does not parse ({exc}).")
    m = re.search(r"\bend\s+(?:your|the)\s+(?:response|answer)\s+with\s+(#+)\s*<\s*number\s*>", q, re.I)
    if m and not re.search(re.escape(m.group(1)) + r"\s*-?\$?[\d,]+(?:\.\d+)?\s*$", a.strip()):
        out.append(f"Form: the question asks you to end with '{m.group(1)} <number>'; your answer does not.")
    m = re.search(r"\bround(?:ed)?\s+(?:it\s+|your\s+answer\s+|the\s+answer\s+)?to\s+the\s+nearest\s+"
                  r"(whole\s+number|integer|tenth|hundredth|thousandth|\d+|[a-z]+)", q, re.I)
    if m:
        unit = m.group(1).lower()
        fin = _final_number_text(a)
        if fin is not None:
            def ok(s):
                v = _num(s)
                dec = len(s.split(".")[1]) if "." in s else 0
                if unit.isdigit():
                    return dec == 0 and v % int(unit) == 0
                if unit in ("tenth", "hundredth", "thousandth"):
                    return dec <= {"tenth": 1, "hundredth": 2, "thousandth": 3}[unit]
                return dec == 0 or float(v).is_integer()
            if not ok(fin):
                out.append(f"Form: the question asks to round to the nearest {unit}; your final answer ({fin}) is not "
                           "rounded that way.")
    return out


def _final_number_text(text):
    """The answer's final number as written: after 'answer is' / 'about' / '####' when present, else the last one."""
    t = normalize(text)
    m = re.findall(r"(?:####|answer\s+is|result\s+is|total\s+is|=)\s*(?:about|approximately|roughly|around|~)?\s*"
                   r"(-?\d[\d,]*(?:\.\d+)?)", t, re.I)
    if m:
        return m[-1]
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", t)
    return nums[-1] if nums else None


def outcome_findings(question, answer):
    """What the deterministic Outcome checks found in `answer` for `question` (empty: nothing found)."""
    return arithmetic_findings(answer) + format_findings(question, answer)


# ---------- one final answer: a second, independent solution compared by code ----------
def is_choice(question):
    q = question or ""
    return bool(re.search(r"^\s*\(A\)", q, re.M) and re.search(r"^\s*\(B\)", q, re.M)) or "option letter" in q.lower()


def is_numeric(question):
    q = question or ""
    if is_choice(q) or re.search(r"\b(?:function|code|program|script|summari[sz]e|rewrite|translate|explain\s+why)\b", q, re.I):
        return False
    if re.search(r"####\s*<number>|\bround(?:ed)?\s+to\s+the\s+nearest\b|\b(?:just|only)\s+(?:the\s+|a\s+)?(?:final\s+)?"
                 r"number\b", q, re.I):
        return True
    # more than one thing asked (several values, a tuple, or a number and a statement): not one value
    if re.search(r"\b(?:ordered\s+(?:pair|triple)|tuple|coordinates|list\s+(?:all|the|every))\b|,\s*(?:and\s+)?what\b|"
                 r"\band\s+(?:state|explain|say|tell|describe|decide)\s+(?:whether|if|why|how)\b", q, re.I) or \
            len(re.findall(r"\bhow\s+(?:many|much|long|far|fast|old)\b|\bwhat\s+(?:is|are|was|were)\b", q, re.I)) > 1:
        return False
    if re.search(r"\bwhat(?:'s|\s+is|\s+are)\s+\(?-?\d[\d,.]*\s*(?:[+\-*/x\u00d7\u00f7^]|times\b|plus\b|minus\b|"
                 r"divided\s+by\b|multiplied\s+by\b|squared\b|cubed\b|to\s+the\s+power\b|percent\b|%)", q, re.I):
        return True
    return bool(re.search(r"\d", q)) and bool(re.search(
        r"\bhow\s+(?:many|much|long|far|fast|old)\b|\bcalculate\b|\bcompute\b|\bwhat\s+is\s+the\b|\bfind\s+the\b", q, re.I))


def single_answer(question):
    return is_choice(question) or is_numeric(question)


def final_answer(question, text):
    """The answer's final value: an option letter or a number (None when it cannot be read)."""
    t = normalize(text)
    if is_choice(question):
        for pat in (r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?((?-i:[A-D]))\)?(?![\w])",
                    r"^\s*\(?((?-i:[A-D]))\)?(?:[.:),\-\u2013\u2014]|\s*$|\s+[\-\u2013\u2014:])", r"\(((?-i:[A-D]))\)"):
            m = re.findall(pat, t, re.I | re.M)
            if m:
                return m[0].upper()
        return None
    if is_numeric(question):
        m = re.findall(r"####\s*(-?[\d,]*\.?\d+)", t) or re.findall(r"ANSWER:\s*(-?[\d,]*\.?\d+)", t, re.I)
        if m:
            return _num(m[-1])
        tail = " ".join(t.strip().splitlines()[-3:])
        # a stated result ("There are 15 primes: 2, 3, ..., 47", "the answer is 21") before the last number in a list
        m = re.findall(r"\b(?:there\s+(?:are|is|were)|answer\s+is|result\s+is|total\s+(?:is|of)|equals?|comes\s+to|"
                       r"(?:it|this|that)\s+(?:is|takes|has))\s+(?:about\s+|approximately\s+|exactly\s+|roughly\s+)?"
                       r"(-?\d[\d,]*(?:\.\d+)?)", tail, re.I)
        if m:
            return _num(m[-1])
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", tail)
        return _num(nums[-1]) if nums else None
    return None


def same(a, b):
    if a is None or b is None:
        return False
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b)):
        return True
    return (float(b).is_integer() and round(a) == b) or (float(a).is_integer() and round(b) == a)


def second_solve_prompt(question_text, question, k=1, note=None):
    """note: what the user added when they said an earlier answer was not right (their account of their world)."""
    end = "ANSWER: <option letter>" if is_choice(question) else "ANSWER: <the final number>"
    how = ("Solve this on your own, step by step." if k == 1 else
           "Solve this problem carefully in a different way from the obvious one, checking each step.")
    added = f"\n\nThe user adds: \"{note}\"" if note else ""
    return question_text + added + f"\n\n{how} On the last line write only: {end}"


def _as_answer(question, solution, value):
    """An independent solution's text as the answer, ending the way the question asks."""
    text = solution.strip()
    for _ in range(3):  # the solution may already end with its own answer lines
        text = re.sub(r"\s*(?:ANSWER:|Answer:|\*\*Answer:?\*\*:?|#+)\s*\S+\s*$", "", text).strip()
    m = re.search(r"\bend\s+(?:your|the)\s+(?:response|answer)\s+with\s+(#+)\s*<\s*number\s*>", question or "", re.I)
    shown = value if isinstance(value, str) else (f"{int(value)}" if float(value).is_integer() else f"{value:g}")
    return text + (f"\n{m.group(1)} {shown}" if m else f"\n\nAnswer: {shown}")


def _words(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _content(text):
    """The words that make a question what it is (filler and plural endings removed)."""
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text) - _FILLER}


def retry_requested(text, previous_question=None):
    """How the user's message says the last answer was not right: "said" (in words, or the Try again button),
    "asked again" (the same question as `previous_question`: the same content words), or None (a new message)."""
    t = (text or "").strip()
    if not t:
        return None
    if len(t) <= 400 and _RETRY_ASK.search(t):
        return "said"
    a = _content(t)
    if len(a) >= 3 and a == _content(previous_question):
        return "asked again"
    return None


def user_detail(feedback):
    """What the user said beyond "try again" / "that's wrong" ("" when that is all they said)."""
    rest = " ".join(_RETRY_ASK.sub(" ", feedback or "").split()).strip(" ,.;:!?-")
    return (feedback or "").strip() if len(_words(rest)) >= 3 else ""


def is_retry_message(m):
    """A user message that asked to try again, rather than a question of its own."""
    if "retry" in m:
        return bool(m["retry"])
    t = m.get("content")
    return isinstance(t, str) and len(t.strip()) <= 400 and bool(_RETRY_ASK.search(t))


def last_exchange(history):
    """(question message, latest answer) a retry is about: the latest answer, and the latest user message before it
    that was a question rather than a request to try again. None when there is nothing to try again."""
    answer = None
    for m in reversed(history or []):
        if answer is None:
            if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
                answer = m["content"]
        elif m.get("role") == "user" and not is_retry_message(m):
            return m, answer
    return None


def _fmt(v):
    return v if isinstance(v, str) else (f"{int(v):,}" if float(v).is_integer() else f"{v:,.6g}")


def _still(found):
    return ("\n\n(Clover's check still finds a problem: " + " ".join(found[:2])[:300] + ")") if found else ""


def web_query(question):
    """What to search for: the question itself, without its answer options and answer-format instructions (the
    sentence that asks, when the question is longer)."""
    q = re.split(r"\n\s*\(?A\)\s", question or "", maxsplit=1)[0]
    q = " ".join(_FORM.sub(" ", re.sub(r"^\s*Question(?:\s*\([^)]*\))?:\s*", "", q, flags=re.I)).split())
    asks = [s for s in re.split(r"(?<=[.?])\s+(?=[A-Z])", q) if s.endswith("?")]
    return (asks[-1] if asks else q)[:300]


def needs_web(question):
    """Whether facts from the web can help a retry: not for code, creative writing, material the question brings, an
    explicit calculation, or a word problem that gives its numbers (two or more) without naming a real-world quantity."""
    q = question or ""
    stem = _FORM.sub(" ", re.split(r"\n\s*\(?A\)\s", q, maxsplit=1)[0])
    if len(q) > 1500 or _CODE.search(q) or _GIVEN.search(q) or (_CREATIVE.search(q) and not single_answer(q)):
        return False
    if re.search(r"\bwhat(?:'s|\s+is|\s+are)\s+\(?-?\d", q, re.I):
        return False
    numbers = len(re.findall(r"\d+(?:[.,]\d+)?", stem)) + len(_NUMWORD.findall(stem))
    return numbers < 2 or bool(_FACT.search(stem))


def python_solve_prompt(question_text, note=None):
    """An independent solution as a program: Clover runs it, so its arithmetic is the computer's, not the model's."""
    q = " ".join(_FORM.sub(" ", question_text or "").split())
    added = f"\n\nThe user adds: \"{note}\"" if note else ""
    return ("Write a short Python program that computes the answer to this question. Use the numbers the question gives "
            "and well-known values where needed, one step per line with clear variable names, and end with "
            "print(\"ANSWER:\", <the final number>). Reply with the program only, in a ```python block.\n\n"
            f"Question: {q}{added}")


def python_answer(output):
    """The value a program printed as `ANSWER: <number>` (the last one), or None."""
    m = re.findall(r"ANSWER:?\s*\(?(-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)", output or "")
    return _num(m[-1]) if m else None


_FORMAT_CALLS = {"print", "round", "int", "float", "str", "abs", "format"}


def computes(code):
    """True when a program calculates (arithmetic, a loop, or a function such as math.sqrt or sum). A program that
    only writes down a value and prints it restates the model's belief, so it is not an independent solution."""
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.AugAssign, ast.For, ast.While, ast.comprehension)):
            return True
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name not in _FORMAT_CALLS:
                return True
    return False


def _dissent(vals, value, program_at=None):
    """The solutions that disagree with `value`, named in parentheses (the Python program by name)."""
    def one(i, v):
        got = _fmt(v) if v is not None else "no clear result"
        return f"the Python program gives {got}" if i == program_at else got
    others = [(i, v) for i, v in enumerate(vals) if not same(v, value)]
    if not others:
        return ""
    if len(others) == 1:
        i, v = others[0]
        return f" ({one(i, v)})" if i == program_at else (f" (the other gives {_fmt(v)})" if v is not None else
                                                          " (the other gave no clear result)")
    return " (the others give " + " and ".join(one(i, v) for i, v in others) + ")"


def _again(detail, used):
    """"I checked again", naming what helped: the user's words and the tools Clover used."""
    parts = (["what you told me"] if detail else []) + list(dict.fromkeys(used or []))
    if not parts:
        return "I checked again"
    return "I checked again, with " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1])


def _keep_code(previous, new):
    """A new answer that dropped the code of the previous one (the model wrote "the code is correct" instead of the
    code): the model's words, then the previous code unchanged."""
    if "```" in previous and "```" not in new:
        blocks = re.findall(r"```[^\n]*\n[\s\S]*?```", previous)
        if blocks:
            return new.strip() + "\n\n" + "\n\n".join(blocks), True
    return new, False


def _redo(question, said, detail):
    """The Direction for a new answer: the user's words when they gave details; otherwise another version of a
    creative answer, or a check of every fact and step of any other answer."""
    if detail:
        return "redo", REDO.format(said=said)
    if _CREATIVE.search(question or "") and not _CODE.search(question or ""):
        return "another", ANOTHER.format(said=said)
    return "recheck", RECHECK.format(said=said)


def retry_cycle(question, previous, feedback, rewrite, solve=None, code_check=None, emit=None, used=None, program=None):
    """The cycle after the user said `previous` was not right.
    rewrite(instruction) -> a new answer from the model, which sees the conversation so far (the question, `previous`,
    and any earlier rewrite with its instruction); solve(k, note) -> an independent solution of the question alone
    (note: what the user added, or None); program(note) -> a third independent solution for a number, computed by a
    program Clover runs ('' when none); code_check(answer) -> compiler feedback or None; used: a list the caller fills
    with the tools that gave more Context ("a web search", "a Python calculation"), named in the reply.
    One-value answers are decided by majority: another value replaces the earlier one only when at least two
    independent solutions give it and more of them back it than back the earlier answer.
    Returns (answer, info); info["lesson"] (with info["lesson_kind"]) when the correction taught something."""
    emit = emit or (lambda e: None)
    said = (" ".join((feedback or "").split()) or "no details given")[:300]
    detail = user_detail(feedback)[:300]

    def check(ans):
        found = outcome_findings(question, ans)
        if code_check is not None:
            fb = code_check(ans)
            if fb:
                found.append("Code: " + fb.strip()[:600])
        return found

    def fix(found):
        emit({"type": "cycle", "stage": "retry", "findings": found})
        return rewrite(FIX.format(said=said, findings="\n".join("- " + f for f in found)))

    def repair(cands, rounds, first=0):
        """Fix the model's latest answer (cands[-1]) on its findings, up to `rounds` times; the attempt from
        cands[first:] with the fewest findings is kept (the later one on ties)."""
        for _ in range(rounds):
            if not cands[-1][1]:
                break
            new = fix(cands[-1][1])
            cands.append((new, check(new)))
        return min(reversed(cands[first:]), key=lambda c: len(c[1]))

    emit({"type": "cycle", "stage": "check"})
    found = check(previous)
    info = {"layer": "cycle", "trigger": "user", "feedback": said, "first_findings": found, "solutions": []}
    if found:  # concrete defects in the answer: fixed on that evidence
        new, nf = repair([(previous, found)], 2, first=1)
        info.update(action="corrected" if len(nf) < len(found) else "rewritten", final_findings=nf)
        if not nf:
            info.update(lesson=lesson_text(question, found, detail, passed=True), lesson_kind="verified")
        return new + _still(nf), info
    if solve is not None and single_answer(question):  # one value: solved again independently; the majority decides
        mine = final_answer(question, previous)
        texts, vals = [], []
        for k in (1, 2):
            emit({"type": "cycle", "stage": "second" if k == 1 else "third"})
            texts.append(solve(k, detail or None) or "")
            vals.append(final_answer(question, texts[-1]))
        program_at = None
        if program is not None and is_numeric(question):
            ran = program(detail or None) or ""
            if final_answer(question, ran) is not None:
                program_at = len(vals)
                texts.append(ran)
                vals.append(final_answer(question, ran))
        info["solutions"] = vals
        groups = []
        for i, v in enumerate(vals):
            if v is None:
                continue
            g = next((g for g in groups if same(g[0], v)), None)
            if g:
                g[1].append(i)
            else:
                groups.append((v, [i]))
        backing = [i for i, v in enumerate(vals) if same(v, mine)]
        best = max((g for g in groups if not same(g[0], mine)), key=lambda g: len(g[1]), default=None)
        again, total = _again(detail, used), len(vals)
        words = {1: "one", 2: "two", 3: "three"}
        if best is not None and len(best[1]) >= 2 and len(best[1]) > len(backing):
            v1, n = best[0], len(best[1])
            info.update(action="overturned" if mine is not None else "answered", overturned={"from": mine, "to": v1})
            head = again + ": " + ("two independent solutions both give" if n == total == 2 else
                                   "all three independent solutions give" if n == total == 3 else
                                   f"{words[n]} of {words[total]} independent solutions give")
            head += f" **{_fmt(v1)}**" + _dissent(vals, v1, program_at)
            if mine is not None:
                head += ", so my earlier answer" + ("" if is_choice(question) else f" ({_fmt(mine)})") + " was wrong"
                info.update(lesson=lesson_text(question, [f"{words[n]} independent solutions gave {_fmt(v1)}, not "
                                                          f"{_fmt(mine)}"], detail), lesson_kind="user")
            return head + ". Here is the working:\n\n" + _as_answer(question, texts[best[1][0]], v1), info
        if backing:
            info["action"] = "kept"
            n = len(backing)
            if n == total:
                phrase = "both independent solutions give" if n == 2 else "all three independent solutions give"
            else:
                phrase = f"{words[n]} of {words[total]} independent solutions " + ("gives" if n == 1 else "give")
            phrase += f" the same result, **{_fmt(mine)}**" + _dissent(vals, mine, program_at)
            ask = ("If I have misunderstood what you meant, tell me what you expected and why." if detail else
                   "If you expected a different result, tell me what you expected and why; I may have misread the "
                   "question.")
            return (f"{again}: {phrase}. Here is that working:\n\n"
                    + _as_answer(question, texts[backing[0]], mine) + "\n\n" + ask), info
        info["action"] = "unconfirmed"
        how, direction = _redo(question, said, detail)
        info["direction"] = how
        emit({"type": "cycle", "stage": "rewrite", "how": how})
        new = rewrite(direction)
        seen = list(dict.fromkeys(_fmt(v) for v in [mine] + vals if v is not None))
        why = (f": independent checks gave different results ({', '.join(seen)})" if len(seen) > 1 else
               " with independent checks")
        return new + f"\n\n(I could not confirm this answer{why}. If you know the expected result, tell me.)", info
    how, direction = _redo(question, said, detail)  # an open answer: a new one, Directed by what the user said
    emit({"type": "cycle", "stage": "rewrite", "how": how})
    new = rewrite(direction)
    new, kept_code = _keep_code(previous, new)
    new, nf = repair([(new, check(new))], 1)  # a concrete defect in the new answer: one corrective round
    info.update(action="rewritten", direction=how, final_findings=nf, kept_code=kept_code)
    if detail:
        info.update(lesson=lesson_text(question, [], detail), lesson_kind="user")
    return new + _still(nf), info


def lesson_text(question, findings, said="", passed=False):
    """A short, reusable lesson from a correction (no full answers stored)."""
    q = " ".join((question or "").split())
    parts = [f"On a question like \"{q[:160]}\""]
    if said:
        parts.append(f"the user said the first answer was not right: \"{said[:200]}\"")
    if findings:
        parts.append("Clover's checks found: " + " ".join(findings)[:500])
    if passed:
        parts.append("the corrected answer passed those checks")
    return "; ".join(parts) + "."
