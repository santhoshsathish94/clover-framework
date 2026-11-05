"""Verification suite for Clover AI Studio local deterministic tool packs (Phase 1 & Phase 2)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOVER = HERE.parent / "clover"
if str(CLOVER) not in sys.path:
    sys.path.insert(0, str(CLOVER))

import lfm3_native
import local_tool_packs
import permissions
import semantics


class DummyEngine:
    def __init__(self, base_dir, workdir=None, perms_dict=None):
        self.base_dir = Path(base_dir)
        self.workdir = Path(workdir) if workdir else None
        self.workname = self.workdir.name if self.workdir else ""
        self.perms = permissions.PermissionStore(self.base_dir / "config" / "permissions.json")
        if perms_dict:
            self.perms.update(perms_dict)
        self.turn_cancel = None
        self.turn_images = []
        self.turn_unknown = set()
        self.turn_ran = set()
        self.turn_web = 0
        self.user_paths = ()
        self._extra = {}

    def _path(self, raw, default=""):
        p = Path(os.path.expandvars(os.path.expanduser(str(raw or "").strip() or default)))
        return Path(self.workdir) / p if self.workdir and not p.is_absolute() else p


def check(cond, msg):
    if not cond:
        print("FAIL", msg)
        raise AssertionError(msg)
    print("PASS", msg)


def test_calculate():
    eng = DummyEngine(tempfile.mkdtemp(prefix="clover_calc_"))
    try:
        r1 = local_tool_packs.run_pack_tool("calculate", {"expression": "137 * 29 + 46"}, eng)
        check(r1 == "4019", f"calculate('137 * 29 + 46') == '4019' (got {r1!r})")

        r2 = local_tool_packs.run_pack_tool("calculate", {"expression": "19.99 * 1.18"}, eng)
        check(r2 == "23.5882", f"calculate('19.99 * 1.18') == '23.5882' (got {r2!r})")

        r3 = local_tool_packs.run_pack_tool("calculate", {"expression": "sqrt(144) + gcd(48, 18) + factorial(5)"}, eng)
        check(r3 == "138", f"calculate functions sqrt/gcd/factorial == '138' (got {r3!r})")

        bad1 = local_tool_packs.run_pack_tool("calculate", {"expression": "__import__('os').system('calc')"}, eng)
        check(bad1.startswith("Error:"), f"calculate rejects __import__: {bad1}")

        bad2 = local_tool_packs.run_pack_tool("calculate", {"expression": "9**9**9"}, eng)
        check(bad2.startswith("Error:"), f"calculate rejects runaway exponent 9**9**9: {bad2}")

        bad3 = local_tool_packs.run_pack_tool("calculate", {"expression": "1 / 0"}, eng)
        check("division by zero" in bad3.lower(), f"calculate division by zero: {bad3}")
    finally:
        shutil.rmtree(eng.base_dir, ignore_errors=True)


def test_convert_units():
    eng = DummyEngine(tempfile.mkdtemp(prefix="clover_units_"))
    try:
        r1 = local_tool_packs.run_pack_tool("convert_units", {"value": 100, "from_unit": "km", "to_unit": "mi"}, eng)
        check("62.137119" in r1 and "mi" in r1, f"convert 100 km -> mi: {r1}")

        r2 = local_tool_packs.run_pack_tool("convert_units", {"value": 37, "from_unit": "C", "to_unit": "F"}, eng)
        check("98.6" in r2 and "F" in r2, f"convert 37 C -> F: {r2}")

        r3 = local_tool_packs.run_pack_tool("convert_units", {"value": 1, "from_unit": "GiB", "to_unit": "MB"}, eng)
        check("1073.741824" in r3 and "MB" in r3, f"convert 1 GiB -> decimal MB: {r3}")

        bad_temp = local_tool_packs.run_pack_tool("convert_units", {"value": -300, "from_unit": "C", "to_unit": "K"}, eng)
        check(bad_temp.startswith("Error:") and "below absolute zero" in bad_temp, f"below absolute zero rejected: {bad_temp}")

        bad_cat = local_tool_packs.run_pack_tool("convert_units", {"value": 10, "from_unit": "km", "to_unit": "kg"}, eng)
        check(bad_cat.startswith("Error:") and "incompatible" in bad_cat, f"incompatible units rejected: {bad_cat}")
    finally:
        shutil.rmtree(eng.base_dir, ignore_errors=True)


def test_search_files_and_file_info():
    tmp = Path(tempfile.mkdtemp(prefix="clover_files_"))
    base = tmp / "base"
    ws = tmp / "workspace"
    outside = tmp / "outside"
    (base / "vault").mkdir(parents=True)
    (base / "vault" / "secret.txt").write_text("TOP_SECRET_TOKEN", encoding="utf-8")
    ws.mkdir(parents=True)
    outside.mkdir(parents=True)
    (outside / "escape.txt").write_text("def route(): pass\n", encoding="utf-8")
    (ws / "alpha.py").write_text("def route(frame):\n    return frame\n", encoding="utf-8")
    (ws / "sub").mkdir()
    (ws / "sub" / "notes.md").write_text("Project notes\ndef route helper\n", encoding="utf-8")
    (ws / "clover-ai-studio").mkdir()
    (ws / "clover-ai-studio" / "private.md").write_text("def route internal\n", encoding="utf-8")

    eng = DummyEngine(base, workdir=ws)
    try:
        # 1. Glob search inside workspace
        g = local_tool_packs.run_pack_tool("search_files", {"pattern": "*.py"}, eng)
        check("alpha.py" in g and "clover-ai-studio" not in g, f"search_files glob: {g}")

        # 2. Content search inside workspace
        c = local_tool_packs.run_pack_tool("search_files", {"query": "def route"}, eng)
        check("alpha.py:1:" in c and "sub/notes.md:2:" in c and "private.md" not in c, f"search_files content: {c}")

        # 3. Truncation flag when max_results=1
        t = local_tool_packs.run_pack_tool("search_files", {"query": "def route", "max_results": 1}, eng)
        check("truncated=true" in t, f"search_files truncation flag: {t}")

        # 4. Refuses vault
        v = local_tool_packs.run_pack_tool("search_files", {"root": str(base / "vault"), "query": "TOP_SECRET"}, eng)
        check(v.startswith("Error:"), f"search_files refuses vault: {v}")

        # 5. Refuses unauthorized folder outside workspace and not named by user
        u = local_tool_packs.run_pack_tool("search_files", {"root": str(outside), "query": "def route"}, eng)
        check(u.startswith("Error:") and "authorized" in u, f"search_files refuses unauthorized outside folder: {u}")

        # 6. Allows outside folder when explicitly named by user in prompt (user_paths)
        eng.user_paths = (str(outside),)
        ok_out = local_tool_packs.run_pack_tool("search_files", {"root": str(outside), "query": "def route"}, eng)
        check("escape.txt:1:" in ok_out, f"search_files allows user-named folder: {ok_out}")
        eng.user_paths = ()

        # 7. Refuses junction/symlink escaping outside workspace
        junc = ws / "linked_outside"
        if os.name == "nt":
            subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(junc), str(outside)], capture_output=True)
        else:
            os.symlink(str(outside), str(junc), target_is_directory=True)
        if junc.exists() or junc.is_symlink():
            j_res = local_tool_packs.run_pack_tool("search_files", {"query": "def route"}, eng)
            check("escape.txt" not in j_res and "linked_outside" not in j_res, f"search_files skips junction/symlink escape: {j_res}")

        # 8. file_info with sha256
        fi = local_tool_packs.run_pack_tool("file_info", {"path": "alpha.py", "sha256": True}, eng)
        expected_sha = hashlib.sha256((ws / "alpha.py").read_bytes()).hexdigest()
        fi_obj = json.loads(fi)
        check(fi_obj.get("encoding_kind") == "text" and fi_obj.get("lines") == 2 and fi_obj.get("sha256") == expected_sha,
              f"file_info sha256 & lines: {fi}")

        # 9. file_info refuses vault and clover-ai-studio
        fi_v = local_tool_packs.run_pack_tool("file_info", {"path": str(base / "vault" / "secret.txt")}, eng)
        check(fi_v.startswith("Error:") and "vault" in fi_v, f"file_info refuses vault: {fi_v}")
        fi_c = local_tool_packs.run_pack_tool("file_info", {"path": str(ws / "clover-ai-studio" / "private.md")}, eng)
        check(fi_c.startswith("Error:"), f"file_info refuses clover-ai-studio: {fi_c}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_git_inspect():
    tmp = Path(tempfile.mkdtemp(prefix="clover_git_"))
    base = tmp / "base"
    repo = tmp / "repo"
    base.mkdir(parents=True)
    repo.mkdir(parents=True)
    eng = DummyEngine(base, workdir=repo)
    try:
        subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@clover.local"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Clover Test"], cwd=str(repo), check=True, capture_output=True)
        (repo / "hello.txt").write_text("line 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "hello.txt"], cwd=str(repo), check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(repo), check=True, capture_output=True)
        (repo / "hello.txt").write_text("line 1\nline 2\n", encoding="utf-8")

        st = local_tool_packs.run_pack_tool("git_inspect", {"operation": "status"}, eng)
        check("hello.txt" in st and "[exit code 0]" in st, f"git_inspect status: {st}")

        lg = local_tool_packs.run_pack_tool("git_inspect", {"operation": "log", "limit": 5}, eng)
        check("Initial commit" in lg and "[exit code 0]" in lg, f"git_inspect log: {lg}")

        df = local_tool_packs.run_pack_tool("git_inspect", {"operation": "diff", "target": "hello.txt"}, eng)
        check("+line 2" in df and "[exit code 0]" in df, f"git_inspect diff: {df}")

        sh = local_tool_packs.run_pack_tool("git_inspect", {"operation": "show", "target": "HEAD"}, eng)
        check("Initial commit" in sh and "[exit code 0]" in sh, f"git_inspect show: {sh}")

        # Reject flag injection in target
        bad1 = local_tool_packs.run_pack_tool("git_inspect", {"operation": "diff", "target": "--output=pwned.txt"}, eng)
        check(bad1.startswith("Error:") and "flags are not allowed" in bad1, f"git_inspect blocks --output flag injection: {bad1}")

        bad2 = local_tool_packs.run_pack_tool("git_inspect", {"operation": "show", "target": "-c"}, eng)
        check(bad2.startswith("Error:") and "flags are not allowed" in bad2, f"git_inspect blocks -c flag injection: {bad2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_phase2_documents_data_and_charts():
    tmp = Path(tempfile.mkdtemp(prefix="clover_phase2_"))
    base = tmp / "base"
    ws = tmp / "ws"
    base.mkdir(parents=True)
    ws.mkdir(parents=True)
    eng = DummyEngine(base, workdir=ws)
    try:
        # 1. Create a CSV fixture
        csv_path = ws / "expenses.csv"
        rows = ["date,category,amount,note"]
        for i in range(1, 31):
            cat = "Groceries" if i % 2 == 0 else "Transport"
            amt = 10.5 * i
            rows.append(f"2026-10-{i:02d},{cat},{amt:.2f},item {i}")
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        # 2. read_document paginated CSV
        doc_p1 = local_tool_packs.run_pack_tool("read_document", {"path": "expenses.csv", "start_row": 1, "max_rows": 5}, eng)
        check("rows 1..5 of 30" in doc_p1 and "2026-10-01" in doc_p1, f"read_document CSV page 1: {doc_p1}")

        doc_p2 = local_tool_packs.run_pack_tool("read_document", {"path": "expenses.csv", "start_row": 26, "max_rows": 10}, eng)
        check("rows 26..30 of 30" in doc_p2 and "2026-10-30" in doc_p2, f"read_document CSV page 2: {doc_p2}")

        # 3. read_document JSON
        json_path = ws / "config.json"
        json_path.write_text(json.dumps([{"studio": "Clover", "score": 47}, {"studio": "Baseline", "score": 40}]), encoding="utf-8")
        doc_j = local_tool_packs.run_pack_tool("read_document", {"path": "config.json"}, eng)
        check("Clover" in doc_j and "47" in doc_j, f"read_document JSON: {doc_j}")

        # 4. analyze_table: filter, group_by + agg, sort
        an = local_tool_packs.run_pack_tool("analyze_table", {
            "path": "expenses.csv",
            "group_by": "category",
            "target_column": "amount",
            "agg": "sum",
            "sort_by": "amount",
            "descending": True,
        }, eng)
        an_obj = json.loads(an)
        check(an_obj["rows_read"] == 30 and not an_obj["truncated"] and len(an_obj["rows"]) == 2, f"analyze_table group_by sum: {an}")
        check("2520.0" in an and "2362.5" in an, f"analyze_table sums match: {an}")

        an_f = local_tool_packs.run_pack_tool("analyze_table", {
            "path": "expenses.csv",
            "filter_column": "amount",
            "filter_op": ">",
            "filter_value": "200",
            "sort_by": "amount",
            "descending": False,
        }, eng)
        an_f_obj = json.loads(an_f)
        check(an_f_obj["rows_matched"] == 11, f"analyze_table filter > 200: {an_f}")

        an_bad = local_tool_packs.run_pack_tool("analyze_table", {
            "path": "expenses.csv",
            "filter_column": "nonexistent",
            "filter_op": "==",
            "filter_value": "x",
        }, eng)
        check(an_bad.startswith("Error:") and "filter_column" in an_bad, f"analyze_table rejects unknown column: {an_bad}")

        # 5. chart_table: generates PNG in media_dir and registers in eng.turn_images
        ch = local_tool_packs.run_pack_tool("chart_table", {
            "path": "expenses.csv",
            "chart_type": "bar",
            "x_column": "category",
            "y_column": "amount",
            "agg": "sum",
            "title": "Expenses by Category",
        }, eng)
        check("shown to the user in the chat" in ch and len(eng.turn_images) == 1, f"chart_table output: {ch}")
        import paths
        png_file = paths.media_dir(base) / eng.turn_images[0]["file"]
        check(png_file.is_file() and png_file.read_bytes()[:4] == b"\x89PNG", f"chart_table wrote valid PNG: {png_file}")

        ch_bad = local_tool_packs.run_pack_tool("chart_table", {"path": "expenses.csv", "chart_type": "radar"}, eng)
        check(ch_bad.startswith("Error:"), f"chart_table rejects invalid chart_type: {ch_bad}")

        # 6. XLSX, DOCX, PPTX fixtures
        import openpyxl
        wb = openpyxl.Workbook()
        sh_a = wb.active
        sh_a.title = "Q1"
        sh_a.append(["Region", "Sales"])
        sh_a.append(["North", 120])
        sh_a.append(["South", 180])
        sh_b = wb.create_sheet("Q2")
        sh_b.append(["Region", "Sales"])
        sh_b.append(["East", 250])
        xlsx_p = ws / "sales.xlsx"
        wb.save(xlsx_p)
        doc_xl = local_tool_packs.run_pack_tool("read_document", {"path": "sales.xlsx", "sheet": "Q2"}, eng)
        check("East" in doc_xl and "250" in doc_xl and "sheet='Q2'" in doc_xl, f"read_document XLSX sheet Q2: {doc_xl}")

        import docx
        d = docx.Document()
        d.add_paragraph("First paragraph of policy")
        d.add_paragraph("Second paragraph of policy")
        docx_p = ws / "policy.docx"
        d.save(docx_p)
        doc_w = local_tool_packs.run_pack_tool("read_document", {"path": "policy.docx", "start_row": 2, "max_rows": 1}, eng)
        check("paragraphs 2..2 of 2" in doc_w and "Second paragraph" in doc_w, f"read_document DOCX: {doc_w}")

        from pptx import Presentation
        pr = Presentation()
        s1 = pr.slides.add_slide(pr.slide_layouts[0])
        s1.shapes.title.text = "Slide One Title"
        s2 = pr.slides.add_slide(pr.slide_layouts[0])
        s2.shapes.title.text = "Slide Two Title"
        pptx_p = ws / "deck.pptx"
        pr.save(pptx_p)
        doc_p = local_tool_packs.run_pack_tool("read_document", {"path": "deck.pptx", "page": 2, "end_page": 2}, eng)
        check("[slide 2]" in doc_p and "Slide Two Title" in doc_p and "Slide One Title" not in doc_p, f"read_document PPTX slide 2: {doc_p}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_media_tools():
    tmp = Path(tempfile.mkdtemp(prefix="clover_media_"))
    base = tmp / "base"
    ws = tmp / "ws"
    base.mkdir(parents=True)
    ws.mkdir(parents=True)
    eng = DummyEngine(base, workdir=ws)
    try:
        clip = ws / "sample.mp4"
        clip.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

        # Negative timestamp rejected immediately
        neg = local_tool_packs.run_pack_tool("extract_frame", {"path": "sample.mp4", "timestamp_s": -2.5}, eng)
        check(neg == "Error: timestamp_s must be >= 0.", f"extract_frame rejects negative timestamp: {neg}")

        # Verify behavior when ffmpeg/ffprobe is missing vs present
        has_ffprobe = local_tool_packs.find_media_binary(base, "ffprobe") is not None
        has_ffmpeg = local_tool_packs.find_media_binary(base, "ffmpeg") is not None
        if not has_ffprobe:
            mi = local_tool_packs.run_pack_tool("media_info", {"path": "sample.mp4"}, eng)
            check("needs ffprobe" in mi, f"media_info clean message without ffprobe: {mi}")
        if not has_ffmpeg:
            ef = local_tool_packs.run_pack_tool("extract_frame", {"path": "sample.mp4", "timestamp_s": 1.0}, eng)
            check("needs ffmpeg" in ef, f"extract_frame clean message without ffmpeg: {ef}")

        # If a real CloverAI ffmpeg installation exists on this machine, test real ffprobe + frame extraction
        real_base = Path(r"C:\Users\santh\CloverAI")
        ff_exe = local_tool_packs.find_media_binary(real_base, "ffmpeg")
        fp_exe = local_tool_packs.find_media_binary(real_base, "ffprobe")
        if ff_exe and fp_exe:
            (base / "tools" / "ffmpeg").mkdir(parents=True, exist_ok=True)
            (base / "tools" / "ffmpeg" / "BIN.txt").write_text(str(Path(ff_exe).parent), encoding="utf-8")
            real_mp4 = ws / "real_clip.mp4"
            subprocess.run([ff_exe, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc=duration=2:size=320x240:rate=15", "-pix_fmt", "yuv420p", str(real_mp4)],
                           check=True, capture_output=True)
            mi_real = local_tool_packs.run_pack_tool("media_info", {"path": "real_clip.mp4"}, eng)
            mi_obj = json.loads(mi_real)
            check(mi_obj["streams"][0]["width"] == 320 and mi_obj["streams"][0]["height"] == 240,
                  f"media_info real ffprobe 320x240: {mi_real}")
            ef_real = local_tool_packs.run_pack_tool("extract_frame", {"path": "real_clip.mp4", "timestamp_s": 0.5}, eng)
            import paths
            check("shown in the chat" in ef_real and len(eng.turn_images) == 1, f"extract_frame real ffmpeg: {ef_real}")
            frame_png = paths.media_dir(base) / eng.turn_images[0]["file"]
            check(frame_png.is_file() and frame_png.read_bytes()[:4] == b"\x89PNG", f"extract_frame wrote PNG: {frame_png}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_permissions_and_routing():
    tmp = Path(tempfile.mkdtemp(prefix="clover_perms_"))
    base = tmp / "base"
    ws = tmp / "ws"
    base.mkdir(parents=True)
    ws.mkdir(parents=True)
    (ws / "notes.txt").write_text("hello world\n", encoding="utf-8")
    try:
        # 1. Permission matrix check via LFM3Native.execute_tool
        dummy = DummyEngine(base, workdir=ws, perms_dict={"files": "deny"})
        eng = lfm3_native.LFM3Native.__new__(lfm3_native.LFM3Native)
        eng.base_dir = base
        eng.workdir = ws
        eng.workname = ws.name
        eng.perms = dummy.perms
        eng.turn_cancel = None
        eng.turn_images = []
        eng.turn_unknown = set()
        eng.turn_ran = set()
        eng.turn_web = 0
        eng.user_paths = ()
        eng._extra = {}

        # calculate & convert_units require no permission even when files is deny
        ran_calc, res_calc, why_calc = eng.execute_tool("calculate", {"expression": "6 * 7"})
        check(ran_calc and res_calc == "42", f"calculate runs without permission: {res_calc} ({why_calc})")

        ran_conv, res_conv, why_conv = eng.execute_tool("convert_units", {"value": 1000, "from_unit": "m", "to_unit": "km"})
        check(ran_conv and "1 km" in res_conv, f"convert_units runs without permission: {res_conv} ({why_conv})")

        # search_files denied when files == deny
        ran_sf, res_sf, why_sf = eng.execute_tool("search_files", {"query": "hello"})
        check(not ran_sf and "turned off" in res_sf, f"search_files blocked when files=deny: {res_sf} ({why_sf})")

        # search_files asks when files == ask
        eng.perms.update({"files": "ask"})
        asked_log = []
        ran_ask, res_ask, why_ask = eng.execute_tool(
            "search_files", {"query": "hello"},
            ask=lambda perm, detail, *a: (asked_log.append((perm, detail)) or True),
        )
        check(ran_ask and asked_log and asked_log[0][0] == "files" and "notes.txt:1:" in res_ask,
              f"search_files asks and runs when approved: {res_ask} ({why_ask})")

        # 2. Semantics routing in "When I ask" mode
        rt = lfm3_native.RUNTIME
        all_tools = [t["function"]["name"] for t in lfm3_native.STUDIO_TOOLS]

        # "Search for 'def route' in my project" -> search_files, NOT web_search!
        f_search = semantics.interpret("Search for 'def route' in my project", rt)
        only_s, on_s = semantics.route(f_search, "ask", True, said="Search for 'def route' in my project", offer=all_tools)
        check(on_s and "search_files" in (only_s or []) and "web_search" not in (only_s or []),
              f"'Search for def route in my project' routes to search_files without web_search: {only_s}")

        # "What is git status in this repo?" -> git_inspect
        f_git = semantics.interpret("What is git status in this repo?", rt)
        only_g, _ = semantics.route(f_git, "ask", True, said="What is git status in this repo?", offer=all_tools)
        check("git_inspect" in (only_g or []), f"'What is git status in this repo?' routes to git_inspect: {only_g}")

        # "Convert 72 F to C" -> convert_units
        f_unit = semantics.interpret("Convert 72 F to C", rt)
        only_u, _ = semantics.route(f_unit, "ask", True, said="Convert 72 F to C", offer=all_tools)
        check("convert_units" in (only_u or []), f"'Convert 72 F to C' routes to convert_units: {only_u}")

        # "Calculate 137 * 29 + 46" -> calculate
        f_calc = semantics.interpret("Calculate 137 * 29 + 46", rt)
        only_c, _ = semantics.route(f_calc, "ask", True, said="Calculate 137 * 29 + 46", offer=all_tools)
        check("calculate" in (only_c or []), f"'Calculate 137 * 29 + 46' routes to calculate: {only_c}")

        # "Give me the SHA256 hash and file size of notes.txt" -> file_info
        f_info = semantics.interpret("Give me the SHA256 hash and file size of notes.txt", rt)
        only_i, _ = semantics.route(f_info, "ask", True, said="Give me the SHA256 hash and file size of notes.txt", offer=all_tools)
        check("file_info" in (only_i or []), f"'SHA256 hash and file size' routes to file_info: {only_i}")

        # "Read page 2 of report.pdf" -> read_document
        f_doc = semantics.interpret("Read page 2 of report.pdf", rt)
        only_d, _ = semantics.route(f_doc, "ask", True, said="Read page 2 of report.pdf", offer=all_tools)
        check("read_document" in (only_d or []), f"'Read page 2 of report.pdf' routes to read_document: {only_d}")

        # "Group by category and sum amount in expenses.csv" -> analyze_table
        f_tab = semantics.interpret("Group by category and sum amount in expenses.csv", rt)
        only_t, _ = semantics.route(f_tab, "ask", True, said="Group by category and sum amount in expenses.csv", offer=all_tools)
        check("analyze_table" in (only_t or []), f"'Group by category and sum amount' routes to analyze_table: {only_t}")

        # "Plot a bar chart of sales.csv" -> chart_table
        f_chart = semantics.interpret("Plot a bar chart of sales.csv", rt)
        only_ch, _ = semantics.route(f_chart, "ask", True, said="Plot a bar chart of sales.csv", offer=all_tools)
        check("chart_table" in (only_ch or []), f"'Plot a bar chart of sales.csv' routes to chart_table: {only_ch}")

        # "What codec and resolution is clip.mp4?" -> media_info
        f_mi = semantics.interpret("What codec and resolution is clip.mp4?", rt)
        only_mi, _ = semantics.route(f_mi, "ask", True, said="What codec and resolution is clip.mp4?", offer=all_tools)
        check("media_info" in (only_mi or []), f"'What codec and resolution is clip.mp4?' routes to media_info: {only_mi}")

        # "Extract a frame at 5 seconds from clip.mp4" -> extract_frame
        f_ef = semantics.interpret("Extract a frame at 5 seconds from clip.mp4", rt)
        only_ef, _ = semantics.route(f_ef, "ask", True, said="Extract a frame at 5 seconds from clip.mp4", offer=all_tools)
        check("extract_frame" in (only_ef or []), f"'Extract a frame at 5 seconds from clip.mp4' routes to extract_frame: {only_ef}")

        # 3. Verify server.tool_inventory reports base, conditional, installed, and unavailable distinctly
        import server
        inv = server.tool_inventory()
        check(set(inv.keys()) == {"base", "conditional", "installed", "unavailable"}, f"tool_inventory keys: {list(inv.keys())}")
        check("calculate" in inv["base"] and "extract_frame" in inv["base"] and len(inv["base"]) == 19,
              f"tool_inventory has all 19 base tools: {len(inv['base'])}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_calculate()
    test_convert_units()
    test_search_files_and_file_info()
    test_git_inspect()
    test_phase2_documents_data_and_charts()
    test_media_tools()
    test_permissions_and_routing()
    print("ALL PASS")
