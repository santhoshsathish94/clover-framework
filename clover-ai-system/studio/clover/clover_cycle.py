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
                       r"rewrite|rephrase|paraphrase|invent|imagine|draw|paint|sketch|illustrate|render|"
                       r"poem|haiku|story|joke|slogan|tagline|lyrics|caption|names?\s+for|title\s+for|"
                       r"(?:image|picture|drawing|illustration|photo)\s+of)\b", re.I)
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
    r"(?:that'?s|that\s+is|this\s+is|it'?s|its|it\s+is|your\s+(?:answer|code|program|solution)\s+is|the\s+(?:answer|code|output|result)\s+is|you'?re|you\s+are)\s+"
    r"(?:wrong|incorrect|not\s+(?:quite\s+)?(?:right|correct))\b|"
    r"(?:wrong|incorrect)\s+(?:for|on|when|with|if|because)\b|"
    r"(?:wrong(?:\s+answer)?|incorrect|nope|not\s+(?:it|right|correct))" + _END + r"|"
    r"(?:that'?s\s+|this\s+is\s+|it'?s\s+|its\s+)?not\s+what\s+i\s+(?:expected|wanted|asked(?:\s+for)?|meant)\b|"
    r"i\s+don'?t\s+think\s+(?:that'?s|this\s+is|it'?s|its|that\s+is|it\s+is)\s+(?:right|correct)\b|"
    r"are\s+you\s+sure\b|"
    r"(?:check\s+(?:it\s+|that\s+|this\s+)?again|double[\s-]?check|re-?check)"
    r"(?:\s+(?:it|that|this|again|the\s+answer|your\s+(?:answer|work|math|maths|calculation|steps)))?"
    r"(?:\s+please)?" + _END + r")", re.I)
_FILLER = {"a", "an", "the", "please", "pls", "again", "so", "ok", "okay", "hey", "clover", "now", "just", "i", "me",
           "tell", "can", "could", "would", "you", "ask", "asking", "question", "same", "once", "more"}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# [ \t] not \s: "(1.07)^3\nA = 25,000" was read as "3 A = 25" (2026-11-09)
_UNIT = r"(?:[ \t]+per[ \t]+[A-Za-z]+|[ \t]+(?!x\b|and\b|of\b|is\b|are\b|to\b|or\b|than\b)[A-Za-z][A-Za-z\-]*(?:/[A-Za-z]+)?|/[A-Za-z]+){0,2}"
_TERM = rf"\(?[ \t]*-?(?:{_NUM})(?!,\d){_UNIT}[ \t]*\)?"
_ARITH = re.compile(rf"(?<![\w.)\]%])(?<!\d,)({_TERM}(?:[ \t]*(?:[+\-\u00d7*/\u00f7^]|x(?=[ \t]*\(?[ \t]*\d))[ \t]*{_TERM})+)[ \t]*(?:\n[ \t]*)?(=|\u2248)[ \t]*(-?(?:{_NUM}))(?![\w^(]|\.\d|,\d|[ \t]*[\^*/\u00d7\u00f7%])")
_ASSIGN = re.compile(r"^[ \t]*(?:[-*\u2022][ \t]*)?([A-Za-z][A-Za-z0-9_]{0,15})[ \t]*(=|\u2248)[ \t]*(.+?)[ \t]*$", re.M)
_PLAIN_RESULT = re.compile(rf"^(-?(?:{_NUM}))(?![\d.,])(?:[ \t]+[A-Za-z][A-Za-z\-]*){{0,2}}[ \t]*[.,;]?$")
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


_DIRECT_EXPR = re.compile(
    r"^\s*(?:(?:what(?:'?s|\s+is|\s+are)|calculate|compute|evaluate)\s+)?([0-9\s(),.+\-*/x\u00d7\u00f7\u2212^]+?)"
    r"\s*(?:=\s*\?)?[?.!]?(?:\s+(?:and\s+)?(?:return|reply\s+with|output|give|provide|write|respond\s+with)\s+(?:only|just)\b[\s\S]*)?$",
    re.I,
)


def direct_arithmetic_value(question):
    """Exact value when `question` is a pure arithmetic expression (e.g. 'What is 137 × 29 + 46? Return only the number.'),
    or None when it is a word problem or contains anything else."""
    m = _DIRECT_EXPR.match(normalize(question or "").strip())
    if not m:
        return None
    expr = m.group(1).strip()
    if not re.search(r"[+\-*/x\u00d7\u00f7^]", expr) or len(re.findall(r"\d+", expr)) < 2:
        return None
    return _eval(expr)


def direct_arithmetic_answer(question):
    """'7 * 8 + 3 = 59' for a pure arithmetic question, or None: Clover writes it when the model's draft got the sum
    wrong, instead of asking the model again (a retry took about 10 s and showed this cycle's wording, 2026-11-10)."""
    v = direct_arithmetic_value(question)
    m = _DIRECT_EXPR.match(normalize(question or "").strip())
    if v is None or not m:
        return None
    shown = f"{int(v)}" if float(v).is_integer() else f"{v:g}"
    return f"{' '.join(m.group(1).split())} = {shown}"


def only_arithmetic(findings):
    """The draft's only problems are arithmetic ones ("you wrote 7 * 8 + 3 = 56, but ...", "the exact value is ...")."""
    return bool(findings) and all(str(f).startswith("Arithmetic:") for f in findings)


def trim_only_value(question, reply):
    """When `question` explicitly asks to return/reply with ONLY a single value (number, year, or code), strips wrapper
    words like 'The archive code is: **CLOVER-7429**' or 'Augusta Ada King — 1815' down to the exact value."""
    q, r = (question or "").strip(), (reply or "").strip()
    if not q or not r or "\n\n" in r or len(r) > 240:
        return reply
    if not re.search(
        r"\b(?:return|reply\s+with|output|respond\s+with|give|provide|write)\s+(?:only|just)\b|\bonly\s+the\s+(?:number|year|code|value|result|answer)\b",
        q,
        re.I,
    ):
        return reply
    clean = re.sub(r"[*_`]+", "", r).strip()
    if re.search(r"\bonly\s+(?:the\s+|a\s+)?(?:[\w'\-]+\s+){0,3}?(?:year|number|result|value|total|sum|product|count)\b", q, re.I):
        if re.search(r"\byear\b", q, re.I):
            yrs = re.findall(r"\b(1\d{3}|20\d{2})\b", clean)
            if len(yrs) == 1:
                return yrs[0]
        v = direct_arithmetic_value(q)
        if v is not None:
            return f"{int(v)}" if float(v).is_integer() else f"{v:g}"
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", clean)
        if len(nums) == 1:
            return nums[0].replace(",", "")
    if re.search(r"\bonly\s+(?:the\s+|a\s+)?(?:[\w'\-]+\s+){0,3}?code\b", q, re.I):
        codes = re.findall(r"\b[A-Z]{2,16}-\d{2,10}\b", clean)
        if len(codes) == 1:
            return codes[0]
        m = re.match(r"^(?:the\s+)?(?:[\w\s]{1,35}?)\s+(?:is|was|:)\s*:?\s*([A-Za-z0-9_\-]+)[.!]?$", clean, re.I)
        if m:
            return m.group(1)
    return reply


def _applied(applied, name):
    if applied is not None and name not in applied:
        applied.append(name)


def _is_wrong(true, rhs, sign, expr):
    """True when the result `rhs` the answer shows for `expr` is wrong, allowing the rounding it shows."""
    shown = _num(rhs)
    decimals = len(rhs.split(".")[1]) if "." in rhs else 0
    diff = abs(true - shown)
    if sign == "\u2248":
        return diff > max(0.5 * 10 ** -decimals, abs(true) * 0.01)
    if float(true).is_integer():
        return diff > 1e-9 * max(1.0, abs(true))
    bad = diff >= 0.5 * 10 ** -decimals - 1e-12  # a rounded result is fine only if it is the value rounded as shown
    whole = all("." not in n for n in re.findall(_NUM, expr))
    if bad and whole and re.search(r"[/\u00f7]", expr) and shown == math.floor(true):
        return False  # whole-number division in a counting step (100 / 125 = 0 trailing zeros)
    return bad


def _shown(true, rhs=""):
    """The true value with two more decimals than the answer showed: '30,626.10, but ... = 30,626.1' hid the error."""
    if float(true).is_integer():
        return f"{int(true):,}"
    decimals = len(rhs.split(".")[1]) if "." in rhs else 0
    return f"{true:,.{decimals + 2}f}".rstrip("0").rstrip(".")


def _chain_findings(text, applied):
    """A value carried down lines that name it: 'A = 25,000 × 1.225043', then 'A ≈ 28,251.30' (code blocks left out:
    reassigning a variable there is not a claim)."""
    out, last = [], {}
    for m in _ASSIGN.finditer(re.sub(r"```[\s\S]*?```", " ", text)):
        name, sign, rhs = m.group(1), m.group(2), m.group(3)
        res = _PLAIN_RESULT.match(rhs)
        if res is None:
            v = _eval(rhs) if re.search(r"[+\-\u00d7*/\u00f7^]", rhs) else None
            last[name] = (rhs, v) if v is not None else None
            continue
        prev = last.pop(name, None)
        if prev is None:
            continue
        _applied(applied, "arithmetic")
        if _is_wrong(prev[1], res.group(1), sign, prev[0]):
            out.append(f"Arithmetic: you wrote {name} = {prev[0]} and then {name} {sign} {res.group(1)}, but "
                       f"{prev[0]} = {_shown(prev[1], res.group(1))}.")
    return out


def arithmetic_findings(answer, applied=None):
    """'a op b = c' statements in the answer whose c is wrong (allowing the rounding the answer shows), and values
    carried down lines that name them. applied: a list that gets "arithmetic" when a statement was checked."""
    out = []
    text = normalize(answer)
    for m in _ARITH.finditer(text):
        expr, sign, rhs = m.group(1), m.group(2), m.group(3)
        true = _eval(expr)
        if true is None:
            continue
        _applied(applied, "arithmetic")
        if _is_wrong(true, rhs, sign, expr):
            out.append(f"Arithmetic: you wrote {' '.join(m.group(0).split())}, but {expr.strip()} = {_shown(true, rhs)}.")
    return (out + _chain_findings(text, applied))[:4]


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


def format_findings(question, answer, applied=None):
    """Checks of the form the question explicitly asks for. applied: a list that gets "requested form" when one ran."""
    q, a, out = question or "", answer or "", []
    if re.search(r"\b(?:in|as|with)\s+(?:one|a\s+single)\s+(?:clear\s+|short\s+|concise\s+)?sentence\b|"
                 r"\bone\s+(?:clear\s+|short\s+|grammatical\s+)?sentence\b", q, re.I):
        _applied(applied, "requested form")
        n = len(_sentences(a))
        if n > 1:
            out.append(f"Form: the question asks for one sentence; your answer has {n}.")
    m_sent = re.search(r"\b(?:in|with|as|exactly)\s+(\d+|" + "|".join(_WORDNUM) + r")\s+(?:clear\s+|short\s+|concise\s+)?sentences\b", q, re.I)
    if m_sent:
        _applied(applied, "requested form")
        want_s = int(m_sent.group(1)) if m_sent.group(1).isdigit() else _WORDNUM[m_sent.group(1).lower()]
        n_s = len(_sentences(a))
        if n_s != want_s:
            out.append(f"Form: the question asks for {want_s} sentences; your answer has {n_s}.")
    m = re.search(r"\b(\d+|" + "|".join(_WORDNUM) + r")\s+(?:bullet(?:\s+point)?s|bullets)\b", q, re.I)
    if m:
        _applied(applied, "requested form")
        want = int(m.group(1)) if m.group(1).isdigit() else _WORDNUM[m.group(1).lower()]
        got = len(re.findall(r"^\s*(?:[-*\u2022]|\d+[.)])\s+\S", a, re.M))
        if got != want:
            out.append(f"Form: the question asks for {want} bullets; your answer has {got}.")
    m = re.search(r"\b(?:in\s+)?(under|fewer\s+than|at\s+most|no\s+more\s+than|up\s+to)\s+(\d+)\s+words\b", q, re.I)
    if m:
        _applied(applied, "requested form")
        limit = int(m.group(2)) - (1 if m.group(1).lower().startswith(("under", "fewer")) else 0)
        words = len(re.findall(r"\b\w[\w'\-]*\b", re.sub(r"```[\s\S]*?```", " ", a)))
        if words > limit:
            out.append(f"Form: the question allows at most {limit} words; your answer has {words}.")
    m_ex = re.search(r"\b(?:in\s+)?exactly\s+(\d+|" + "|".join(_WORDNUM) + r")\s+words\b", q, re.I)
    if m_ex:
        _applied(applied, "requested form")
        want_w = int(m_ex.group(1)) if m_ex.group(1).isdigit() else _WORDNUM[m_ex.group(1).lower()]
        words = len(re.findall(r"\b\w[\w'\-]*\b", re.sub(r"```[\s\S]*?```", " ", a)))
        if words != want_w:
            out.append(f"Form: the question asks for exactly {want_w} words; your answer has {words}.")
    if re.search(r"\b(?:in|as|valid|return|output|respond\s+with|give)\s+(?:a\s+|an\s+|the\s+)?(?:valid\s+)?json\b", q, re.I):
        _applied(applied, "requested form")
        cand = _json_candidate(a)
        if cand is None:
            out.append("Form: the question asks for JSON; your answer contains none.")
        else:
            try:
                json.loads(cand)
            except ValueError as exc:
                out.append(f"Form: the JSON in your answer does not parse ({exc}).")
    m = re.search(r"\bend\s+(?:your|the)\s+(?:response|answer)\s+with\s+(#+)\s*<\s*number\s*>", q, re.I)
    if m:
        _applied(applied, "requested form")
    if m and not re.search(re.escape(m.group(1)) + r"\s*-?\$?[\d,]+(?:\.\d+)?\s*$", a.strip()):
        out.append(f"Form: the question asks you to end with '{m.group(1)} <number>'; your answer does not.")
    m = re.search(r"\bround(?:ed)?\s+(?:it\s+|your\s+answer\s+|the\s+answer\s+)?to\s+the\s+nearest\s+"
                  r"(whole\s+number|integer|tenth|hundredth|thousandth|\d+|[a-z]+)", q, re.I)
    if m:
        unit = m.group(1).lower()
        fin = _final_number_text(a)
        if fin is not None:
            _applied(applied, "requested form")

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


def outcome_findings(question, answer, applied=None):
    """What the deterministic Outcome checks found in `answer` for `question` (empty: nothing found, which is not the
    same as checked: `applied` gets the checks that ran)."""
    return arithmetic_findings(answer, applied) + format_findings(question, answer, applied)


def outcome_label(checks, cycles=1):
    """The Outcome stage's words when no finding is left: the checks that ran on the answer, or that none applies."""
    if not checks:
        return "No automatic check applies to this answer"
    return f"Checked: {', '.join(checks)} (passed" + (f" in {cycles} Clover cycles)" if cycles > 1 else ")")


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
        m = re.findall(r"####\s*(-?[\d,]*\.?\d+)", t)
        if m:
            return _num(m[-1])
        for line in reversed(re.findall(r"ANSWER:\s*([^\n]+)", text or "", re.I)):
            v = _stated_value(line)
            if v is not None:
                return v
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


def _stated_value(line):
    """The amount an answer line states: its only number, else its last money amount, else its last number.
    "Answer: 30 pens cost about $20.01" was read as 30 (2026-11-01)."""
    money = re.findall(r"[$\u20ac\u00a3\u20b9]\s*(-?\d[\d,]*(?:\.\d+)?)", line)
    if money:
        return _num(money[-1])
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", normalize(line))
    return _num(nums[-1]) if nums else None


def _decimals(x):
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0


def same(a, b):
    """Two answers agree: equal letters, or numbers equal at the precision they were written. Roundings of one value
    agree (3.33 and 3.333); an exact whole number does not (a program's 20.0 and the model's $20.01 differ: within 0.2%
    they counted as one answer and made a false "verified" lesson, 2026-11-01). Float noise (19.999999999999996) is
    ignored."""
    if a is None or b is None:
        return False
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    a, b = round(a, 6), round(b, 6)
    da, db = _decimals(a), _decimals(b)
    d = min(da, db) if da and db else max(da, db)
    return abs(a - b) <= 0.5 * 10 ** -d + 1e-9 * max(1.0, abs(a), abs(b))


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


_FOLLOW = re.compile(r"^\W*(?:and|but|so|then|also|now|ok(?:ay)?|what\s+about|how\s+about|what\s+if)\b|"
                     r"\b(?:it|that|this|those|these|them|they|same|instead)\b", re.I)


def follow_up(question):
    """A message that leans on the turn before it ("And how much would 30 pens cost?", "what about it?")."""
    return bool(_FOLLOW.search(question or "")) or len(_content(question)) < 3


def context_before(history, msg, chars=700):
    """The exchange before `msg` (a user message in `history`) when `msg` follows from it, as text that goes before the
    question in the checks; "" otherwise. A retry of "And how much would 30 pens cost?" was solved without "3 for $2"."""
    if not isinstance(msg.get("content"), str) or not follow_up(msg["content"]):
        return ""
    at = next((i for i, m in enumerate(history or []) if m is msg), None)
    q = a = None
    for m in reversed((history or [])[:at] if at else []):
        if a is None:
            if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
                a = m["content"]
        elif m.get("role") == "user" and isinstance(m.get("content"), str) and not is_retry_message(m):
            q = m["content"]
            break
    if not (q and a):
        return ""
    clip = lambda s: " ".join(s.split())[:chars]  # noqa: E731
    return (f"Earlier in this conversation the user asked: \"{clip(q)}\"\nThe answer was: \"{clip(a)}\"\n"
            "The question below follows from that exchange.\n\n")


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
                # a lesson needs more than a majority of the model's own solves: what the user said, or the program
                # (a wrong overturn became a "user_stated_belief" and leaked into an unrelated question, 2026-11-01)
                if detail or (program_at is not None and program_at in best[1]):
                    info.update(lesson=lesson_text(question, [f"{words[n]} independent solutions gave {_fmt(v1)}, not "
                                                              f"{_fmt(mine)}"], detail),
                                lesson_kind="user" if detail else "verified")
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


AUTO_CYCLE_FIX = (
    "Clover Cycle Outcome Verification checked your draft above against reality and found:\n"
    "{findings}\n"
    "Return to Context: use the verified evidence and tools above, fix the cause of each problem, and reply with the complete corrected answer."
)


def decompose_task_todos(question: str, offered_tools=None) -> list[dict]:
    """Break a user request into structured, actionable task todo items (`id`, `title`, `status`) following the
    Clover operating cycle."""
    q = str(question or "").strip()
    tools_list = [str(t) for t in (offered_tools or ()) if t and t != "manage_todo_list"]
    explicit_steps = []
    for m in re.finditer(r"(?:^|\n)\s*(?:\d+[.)]|[-*•]|step\s+\d+[:.)])\s+([^\n]{4,120})", q, re.I):
        item = " ".join(m.group(1).split()).rstrip(".;,")
        if item and item not in explicit_steps:
            explicit_steps.append(item[:90])
    if len(explicit_steps) < 2 and re.search(r"\b(?:first|1\))\b[\s\S]+?\b(?:then|second|next|2\)|finally)\b", q, re.I):
        parts = re.split(r"(?:[,;.]|\band\b)\s*(?=(?:first|second|third|then|next|after\s+that|finally)\b)", q, flags=re.I)
        for p in parts:
            cl = re.sub(r"^(?:first|second|third|then|next|after\s+that|finally)\s*[,:]?\s*", "", p.strip(), flags=re.I)
            cl = " ".join(cl.split()).rstrip(".;,")
            if len(cl) >= 6 and cl not in explicit_steps:
                explicit_steps.append(cl[:90])
    q_short = " ".join(q.split())[:68] or "user request"
    todos = [
        {
            "id": 1,
            "title": "Read context files, memory & task constraints",
            "status": "not-started",
        }
    ]
    if len(explicit_steps) >= 2:
        for step_txt in explicit_steps[:5]:
            todos.append({
                "id": len(todos) + 1,
                "title": step_txt[0].upper() + step_txt[1:] if step_txt else "Execute step",
                "status": "not-started",
            })
    else:
        exec_title = (
            f"Run tools ({', '.join(tools_list[:3])}) & build solution"
            if tools_list
            else f"Solve & draft response ({q_short})"
        )
        todos.append({
            "id": 2,
            "title": exec_title[:90],
            "status": "not-started",
        })
    todos.append({
        "id": len(todos) + 1,
        "title": "Verify outcome against reality & update context/growth",
        "status": "not-started",
    })
    return todos


def normalize_todo_list(raw_items) -> list[dict]:
    """Normalize a list of todo dicts (`id`, `title`, `status` in `not-started | in-progress | completed`), ensuring
    at most ONE item is `in-progress` at a time."""
    if not isinstance(raw_items, list):
        return []
    out = []
    seen_in_progress = False
    for idx, it in enumerate(raw_items[:20], start=1):
        if not isinstance(it, dict):
            continue
        title = " ".join(str(it.get("title") or it.get("text") or it.get("step") or f"Step {idx}").split())[:120]
        if not title:
            continue
        st = str(it.get("status") or "not-started").strip().lower().replace("_", "-")
        if st not in ("not-started", "in-progress", "completed"):
            st = "completed" if st in ("done", "complete", "finished") else (
                "in-progress" if st in ("active", "running", "working", "doing") else "not-started"
            )
        if st == "in-progress":
            if seen_in_progress:
                st = "not-started"
            else:
                seen_in_progress = True
        out.append({"id": int(it.get("id") or idx), "title": title, "status": st})
    return out


class OperatingCyclePlan:
    """Tracks the 5-Stage Clover Operating Cycle (`Context -> Direction -> Execution -> Outcome -> Growth`) AND the
    per-turn actionable Task Todo List (`self.todos`) step-by-step during a turn, emitting structured `operating_cycle`
    and `todo_list` SSE events in real time."""

    STAGE_ORDER = ("context", "direction", "execution", "outcome", "growth")
    STAGE_TITLES = {
        "context": "1. Context",
        "direction": "2. Direction",
        "execution": "3. Execution",
        "outcome": "4. Outcome",
        "growth": "5. Growth",
    }

    def __init__(self, emit_fn=None, model_name="LFM3-Omni-2B", query="", offered_tools=None, persist_todos_fn=None):
        self.emit = emit_fn or (lambda e: None)
        self.model_name = model_name or "LFM3-Omni-2B"
        self.cycle = 1
        self.persist_todos_fn = persist_todos_fn
        q_short = " ".join(str(query or "").split())[:90]
        tools_list = list(offered_tools) if isinstance(offered_tools, (list, tuple, set)) else []
        t_desc = (", ".join(tools_list[:4])) if tools_list else "direct model reasoning"
        self.stages = {
            "context": {
                "id": "context",
                "title": self.STAGE_TITLES["context"],
                "status": "not-started",
                "detail": "Read AGENTS.md operating boundaries, S1 memory, S2 knowledge graph/recall & workspace files",
            },
            "direction": {
                "id": "direction",
                "title": self.STAGE_TITLES["direction"],
                "status": "not-started",
                "detail": f"Establish human intent, constraints & C2 HybridAgent/JEPA plan ({t_desc})",
            },
            "execution": {
                "id": "execution",
                "title": self.STAGE_TITLES["execution"],
                "status": "not-started",
                "detail": f"Run S3 tools & C1 ({self.model_name}) -> C3 (11-Level NCM) execution",
            },
            "outcome": {
                "id": "outcome",
                "title": self.STAGE_TITLES["outcome"],
                "status": "not-started",
                "detail": "Verify output against reality (compiler, Python sandbox, arithmetic & format checks)",
            },
            "growth": {
                "id": "growth",
                "title": self.STAGE_TITLES["growth"],
                "status": "not-started",
                "detail": "Align epistemic state (verified/known/inferred) & preserve learned context for next cycle",
            },
        }
        self._q_short = q_short
        self.todos = decompose_task_todos(query, offered_tools=tools_list)
        self._custom_todos = False

    def todos_markdown(self) -> str:
        lines = ["# Clover Task Todo List", ""]
        for it in self.todos:
            mark = "x" if it["status"] == "completed" else ("~" if it["status"] == "in-progress" else " ")
            lines.append(f"- [{mark}] {it['id']}. {it['title']} ({it['status']})")
        return "\n".join(lines) + "\n"

    def todo_snapshot(self):
        return {
            "type": "todo_list",
            "cycle": self.cycle,
            "items": [dict(x) for x in self.todos],
        }

    def _emit_todos(self):
        snap = self.todo_snapshot()
        self.emit(snap)
        if callable(self.persist_todos_fn):
            try:
                self.persist_todos_fn(self.todos, self.todos_markdown())
            except Exception:
                pass
        return snap

    def set_todos(self, raw_items, custom=True):
        norm = normalize_todo_list(raw_items)
        if not norm:
            return self.todo_snapshot()
        self.todos = norm
        if custom:
            self._custom_todos = True
        return self._emit_todos()

    def _auto_advance_todo(self, stage_key, completed_all=False):
        if not self.todos:
            return
        if completed_all:
            for it in self.todos:
                it["status"] = "completed"
            self._emit_todos()
            return
        if self._custom_todos:
            return
        n = len(self.todos)
        if stage_key in ("context", "direction"):
            self.todos[0]["status"] = "in-progress"
            for j in range(1, n):
                self.todos[j]["status"] = "not-started"
        elif stage_key == "execution":
            self.todos[0]["status"] = "completed"
            for j in range(1, n):
                self.todos[j]["status"] = "not-started"
            if n >= 2:
                self.todos[1]["status"] = "in-progress"
        elif stage_key in ("outcome", "growth"):
            for j in range(0, max(1, n - 1)):
                self.todos[j]["status"] = "completed"
            self.todos[-1]["status"] = "in-progress"
        self._emit_todos()

    def advance_execution_step(self, step_note=""):
        """When multiple execution subtasks exist and a tool finishes, advance to the next middle todo item."""
        if self._custom_todos or len(self.todos) <= 3:
            return
        mid_indices = list(range(1, len(self.todos) - 1))
        cur_idx = next((i for i in mid_indices if self.todos[i]["status"] == "in-progress"), None)
        if cur_idx is not None and cur_idx + 1 < len(self.todos) - 1:
            self.todos[cur_idx]["status"] = "completed"
            self.todos[cur_idx + 1]["status"] = "in-progress"
            self._emit_todos()

    def snapshot(self, active_stage=""):
        return {
            "type": "operating_cycle",
            "cycle": self.cycle,
            "active_stage": active_stage,
            "model": self.model_name,
            "stages": [dict(self.stages[k]) for k in self.STAGE_ORDER],
            "todos": [dict(x) for x in self.todos],
        }

    def start_stage(self, stage_key, detail=None, cycle=None):
        if cycle is not None:
            self.cycle = int(cycle)
        for k in self.STAGE_ORDER:
            if k == stage_key:
                self.stages[k]["status"] = "in-progress"
                if detail:
                    self.stages[k]["detail"] = str(detail)[:220]
                break
            elif self.stages[k]["status"] == "in-progress":
                self.stages[k]["status"] = "completed"
        self._auto_advance_todo(stage_key, completed_all=False)
        self.emit(self.snapshot(active_stage=stage_key))

    def complete_stage(self, stage_key, detail=None):
        if stage_key in self.stages:
            self.stages[stage_key]["status"] = "completed"
            if detail:
                self.stages[stage_key]["detail"] = str(detail)[:220]
        if stage_key == "growth":
            self._auto_advance_todo("growth", completed_all=True)
        self.emit(self.snapshot(active_stage=stage_key))

    def retry_from_context(self, next_cycle, finding_summary=""):
        self.cycle = int(next_cycle)
        sum_short = " ".join(str(finding_summary or "").split())[:140]
        self.stages["context"]["status"] = "completed"
        self.stages["context"]["detail"] = f"Cycle {self.cycle} Context: incorporated Outcome finding ({sum_short})"
        self.stages["direction"]["status"] = "completed"
        self.stages["direction"]["detail"] = f"Cycle {self.cycle} Direction: repair root cause & re-verify against reality"
        self.stages["execution"]["status"] = "in-progress"
        self.stages["execution"]["detail"] = f"Cycle {self.cycle} Execution: running corrected pass on {self.model_name}"
        self.stages["outcome"]["status"] = "not-started"
        self.stages["growth"]["status"] = "not-started"
        if not self._custom_todos and len(self.todos) >= 2:
            self.todos[0]["status"] = "completed"
            for j in range(1, len(self.todos)):
                self.todos[j]["status"] = "not-started"
            self.todos[1]["status"] = "in-progress"
            self.todos[1]["title"] = f"Cycle {self.cycle} repair: fix Outcome finding ({sum_short[:65]})"
            self._emit_todos()
        self.emit(self.snapshot(active_stage="execution"))



def extract_time_examples(question: str) -> list[tuple[str, str]]:
    """Extract `(4_digit_str, expected_hh_mm)` pairs from questions like:
    'example 1234 23:41 is the right answer' or 'for 1230 the max possible time is 23:10'."""
    q = str(question or "")
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in re.finditer(r"\b(\d{4})\b[^0-9:\n]{1,50}?\b((?:[01]\d|2[0-3]):[0-5]\d)\b", q):
        p = (m.group(1), m.group(2))
        if p not in seen:
            seen.add(p)
            pairs.append(p)
    if not pairs:
        d4 = re.findall(r"\b(\d{4})\b", q)
        tms = [m.group(0) for m in re.finditer(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b", q)]
        if len(d4) == 1 and len(tms) == 1:
            pairs.append((d4[0], tms[0]))
    return pairs


def synthesize_code_verification_harness(question: str, code: str) -> str:
    """When `code` defines a Python function (`def fn(...)`), synthesize a non-destructive verification harness that:
    1. Runs `code` as written.
    2. If `question` contains 4-digit -> HH:MM time examples (e.g. `1234 -> 23:41`, `1230 -> 23:10`), calls the defined
       function on each example (trying `[int(c) for c in d]`, `d`, and `tuple(int(c) for c in d)`) and prints
       `CLOVER_TIME_CHECK <digits> => <actual> (expected <hh:mm>)`.
    """
    code_s = (code or "").strip()
    if not code_s:
        return ""
    try:
        tree = ast.parse(code_s)
    except SyntaxError:
        return code_s
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
    time_pairs = extract_time_examples(question)
    if fn_names and time_pairs:
        fn_target = fn_names[-1]
        pairs_repr = repr(time_pairs)
        harness = (
            f"\n\n# --- Clover Outcome Verification Harness ---\n"
            f"def _clover_call_time_fn(_fn, _d_str):\n"
            f"    for _arg in ([int(_c) for _c in _d_str], _d_str, [ _c for _c in _d_str ], tuple(int(_c) for _c in _d_str)):\n"
            f"        try:\n"
            f"            _r = _fn(_arg)\n"
            f"            if _r is not None:\n"
            f"                return str(_r).strip()\n"
            f"        except Exception:\n"
            f"            pass\n"
            f"    try:\n"
            f"        return str(_fn(*[int(_c) for _c in _d_str])).strip()\n"
            f"    except Exception as _e:\n"
            f"        return f'ERROR:{{_e}}'\n"
            f"for _d_in, _exp_t in {pairs_repr}:\n"
            f"    _got_t = _clover_call_time_fn({fn_target}, _d_in)\n"
            f"    print(f'CLOVER_TIME_CHECK {{_d_in}} => {{_got_t}} (expected {{_exp_t}})')\n"
        )
        return code_s + harness
    return code_s


def verify_turn_outcome(
    question,
    answer,
    rounds=None,
    *,
    kit=None,
    layers=(),
    meaning=None,
    execute_python_fn=None,
):
    """Run Clover's deterministic Outcome stage (`Context -> Direction -> Execution -> Outcome -> Growth`) on a turn draft.
    Checks:
      1. Empty / truncated non-answer.
      2. Compiler & AST syntax verification (`kit.compiler_check`), plus sandbox execution (`execute_python_fn`) for
         self-contained computational Python blocks (including synthesized test harnesses for functions with examples in `question`).
      3. Arithmetic equation verification (`arithmetic_findings`) & explicit format constraints (`format_findings`),
         plus exact direct-expression evaluation (`direct_arithmetic_value`).
      4. Unrecovered tool execution failures in `rounds`.
    Returns `(findings, meta)`. Empty `findings` means nothing was found; `meta["checks"]` names the checks that ran on
    this answer (empty: no automatic check applies, so the answer is not verified).
    """
    q = (question or "").strip()
    a = (answer or "").strip()
    rounds_list = list(rounds or [])
    findings = []
    meta = {
        "compiler_checked": False,
        "compiler_ok": True,
        "compiler_info": None,
        "auto_python_ran": False,
        "python_sandbox_checked": False,
        "checks": [],
    }
    checks = meta["checks"]

    if not a or a.startswith("(Clover "):
        findings.append("Outcome check: the draft answer was empty or incomplete; write a complete direct answer.")
        return findings, meta

    if "compiler" in (layers or ()) and kit is not None and hasattr(kit, "compiler_check"):
        fb, info = kit.compiler_check(q, a)
        meta["compiler_info"] = info
        meta["compiler_checked"] = bool(info and info.get("checked"))
        meta["compiler_ok"] = not bool(fb)
        if meta["compiler_checked"]:
            _applied(checks, "code compiles")
        if fb:
            findings.append("Code compiler check:\n" + fb.strip()[:600])
        elif meta["compiler_checked"] and execute_python_fn is not None:
            try:
                import lfm3_layers
                code = lfm3_layers.extract_python(a)
            except Exception:
                code = ""
            already_ran_py = any(rd.get("tool") == "run_python" and rd.get("ran") and rd.get("worked") for rd in rounds_list)
            time_pairs = extract_time_examples(q)
            run_code = synthesize_code_verification_harness(q, code) if code else ""
            has_print_or_harness = bool(re.search(r"\b(?:print|assert)\b", run_code))
            if (
                run_code
                and (not already_ran_py or time_pairs)
                and (computes(code) or computes(run_code))
                and has_print_or_harness
                and not re.search(
                    r"\b(?:input\s*\(|open\s*\(|os\.remove|os\.unlink|os\.rmdir|shutil\.|subprocess\.|socket\.|urllib\.|requests\.|fastapi|flask|django|uvicorn|plt\.show)\b",
                    code,
                )
            ):
                py_res = execute_python_fn(run_code)
                if len(py_res) == 3:
                    ran, out, worked = py_res
                else:
                    worked, out = py_res
                    ran = True
                meta["auto_python_ran"] = bool(ran)
                meta["python_sandbox_checked"] = bool(ran)
                if ran:
                    _applied(checks, "code runs")
                if ran and not worked:
                    findings.append(f"Code execution check: running your Python code raised an error:\n{str(out).strip()[:500]}")
                elif ran and worked:
                    rounds_list.append({"tool": "run_python", "ran": True, "worked": True, "result": str(out)})
                    for m_tc in re.finditer(r"CLOVER_TIME_CHECK\s+(\d{4})\s*=>\s*([^\r\n(]+?)\s*\(expected\s+([0-9:]+)\)", str(out)):
                        d_in, got_t, exp_t = m_tc.group(1), m_tc.group(2).strip(), m_tc.group(3).strip()
                        if got_t != exp_t:
                            findings.append(
                                f"Code output check: for digits `{d_in}`, the expected max 24-hour time is `{exp_t}`, "
                                f"but your function returned `{got_t}`. Use `itertools.permutations` over all 4 digits "
                                f"(`hh = d[0]*10 + d[1]`, `mm = d[2]*10 + d[3]`, `0 <= hh < 24 and 0 <= mm < 60`) and "
                                f"format the maximum valid `(hh, mm)` as `f'{{hh:02d}}:{{mm:02d}}'`."
                            )
            if not findings:
                exp_times = re.findall(r"\b([01]\d|2[0-3]):[0-5]\d\b", q)
                if exp_times:
                    full_times = [m.group(0) for m in re.finditer(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b", q)]
                    py_outs = "\n".join(str(rd.get("result") or "") for rd in rounds_list if rd.get("tool") == "run_python" and rd.get("ran"))
                    if py_outs and any(ft not in py_outs for ft in full_times):
                        findings.append(
                            f"Code output check: the question specifies expected 24-hour time output {', '.join(full_times)}, "
                            f"but running your code printed `{py_outs.strip()[:200]}`. Check all 4-digit permutations "
                            "`itertools.permutations` with `0 <= hh < 24 and 0 <= mm < 60` and return `f'{hh:02d}:{mm:02d}'`."
                        )

    af = outcome_findings(q, a, checks)
    if af:
        findings.extend(af)

    exact_v = direct_arithmetic_value(q)
    if exact_v is not None:
        _applied(checks, "arithmetic")
        got_v = final_answer(q, a)
        if not same(got_v, exact_v):
            shown_v = f"{int(exact_v)}" if float(exact_v).is_integer() else f"{exact_v:g}"
            findings.append(f"Arithmetic: the exact value is {shown_v} (your draft gave {got_v}).")

    if rounds_list and not findings:
        last_rd = rounds_list[-1]
        if last_rd.get("ran") and not last_rd.get("worked"):
            res_txt = str(last_rd.get("result") or "").strip()
            if not res_txt.startswith(("[Clover permission]", "Not run:")) and not any(
                rd.get("ran") and rd.get("worked") for rd in rounds_list
            ):
                _applied(checks, "tool error reported")
                if not re.search(r"\b(?:error|failed|could not|unable to|did not work|doesn'?t exist|not found)\b", a, re.I):
                    findings.append(
                        f"Tool execution check: `{last_rd.get('tool')}` failed with `{res_txt[:240]}`, "
                        "but your draft did not resolve the error or report it honestly."
                    )

    return findings, meta
