#!/usr/bin/env python3
"""Pull the engine's real configuration / summary text out of the COMMITTED engine sources, byte for byte (CRLF kept), into ref/x/*.inc.

Source of truth: `git show 4d8d891:clover-kimi-k3/src/code/gpu/backend-gpu.cu` and `.../decode.cuh` (the working tree is NOT read: someone else edits it).
Checks, any failure exits 1 and leaves no output behind (ref/x is emptied first):
  - the git blob read must have the pinned sha256 (whole file), so line numbers mean what they meant when this table was written;
  - every slice: the first line must start with the expected text, the last line must end with the expected text, function slices must have balanced braces
    (counted per character; comments and strings are not parsed);
  - every written slice must have the pinned sha256 (a slice the table does not pin is an error, unless --print-pins);
  - the files are written, read back and compared; a MANIFEST.txt lists name, source lines, bytes and sha256 of every slice.
Usage:  python extract.py [--repo DIR] [--out DIR] [--src-dir DIR] [--print-pins]
  --repo     git work tree to run `git show` in (default: the clover-framework checkout containing this file)
  --src-dir  instead of git: a directory holding backend-gpu.cu and decode.cuh dumped from 4d8d891 (same pin checks apply)
"""
import argparse, hashlib, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMIT = "4d8d891"
BG_PATH = "clover-kimi-k3/src/code/gpu/backend-gpu.cu"
DC_PATH = "clover-kimi-k3/src/code/gpu/decode.cuh"
BLOB_PINS = {
    "backend-gpu.cu": "cc1950a108900a62f42f9e278012753ecbe64f2928e7b95fe6958e7b9207c3f3",
    "decode.cuh": "fc7682e9bc6a043af4ebe59b1a7cb27f750ff7e6cde4dcb8beeaed5294955dee",
}

# (name, file, [(first_line, last_line), ...], start_text, end_text, is_function)   lines are 1-based, inclusive, of the pinned blob
S = []
def add(name, f, ranges, start, end="", func=False):
    S.append((name, f, ranges, start, end, func))

# declarations the extracted code needs, kept as the original wrote them (types and initial values come from the engine, not from us)
add("decls", "BG", [(31, 31), (43, 43), (45, 46), (48, 48), (56, 56), (58, 58), (83, 83), (99, 99), (101, 102), (175, 176), (295, 298)], "enum { MAX_CARDS")
add("decls_dc", "DC", [(207, 207), (211, 211), (311, 311), (437, 437), (486, 486), (495, 496), (500, 500), (523, 523)], "static int g_policy")
# whole functions
add("env_int", "BG", [(50, 50)], "static int env_int(", "}")
add("card_of", "BG", [(526, 526)], "static int card_of(int L)", "}")
add("force_routes", "BG", [(374, 387)], "static void force_routes(int L, int n, int *sel)", "}", True)
add("print_summary", "BG", [(465, 485)], "static void print_summary()", "}", True)
add("configure", "BG", [(487, 524)], "static void configure()", "}", True)
# statements of configure() (one file per statement group; wrapper code supplies the variables they name)
add("cfg_cards", "BG", [(492, 495)], "    g_cards = std::min(", "<< 20;")
add("cfg_policy", "BG", [(497, 497)], "    slotcache_set_policy(env_int(\"K3GPU_POLICY\", 1));", ";")
add("cfg_chunks", "BG", [(499, 499)], "    g_chunks = env_int(\"K3GPU_CHUNKS\", 0);", ";")
add("cfg_trace", "BG", [(500, 500)], "    { const char *tr = getenv(\"K3GPU_TRACE\");", "} } }")
add("cfg_host", "BG", [(501, 501)], "    g_host.init((size_t)env_int(\"K3GPU_HOST_GB\", 150) << 30);", ";")
add("cfg_rc_head", "BG", [(502, 503)], "    { const char *rc = getenv(\"K3GPU_ROUTECACHE\");", "if (rc && *rc) {")
add("cfg_lookahead", "BG", [(505, 505)], "          g_lookahead = env_int(\"K3GPU_LOOKAHEAD\", 92);", ";")
add("cfg_pendcap", "BG", [(506, 506)], "          g_pend_cap = g_host.cap_slots * 6 / 10;", ";")
add("cfg_vq", "BG", [(509, 511)], "          g_vq = env_int(\"K3GPU_VQUEUE\", 0) != 0 && g_policy == 2;", ";")
add("cfg_rc_msg", "BG", [(512, 512)], "          fprintf(stderr, \"K3GPU route cache: ", ");")
add("cfg_layout", "BG", [(514, 514)], "    g_layout = env_int(\"K3GPU_LAYOUT\", 0);", ";")
add("cfg_prefill_flag", "BG", [(515, 515)], "    g_prefill = env_int(\"K3GPU_PREFILL\", 0) != 0;", ";")
add("cfg_prefill_check", "BG", [(517, 517)], "        if (g_rc.empty()) { fprintf(stderr, \"K3GPU_PREFILL needs K3GPU_ROUTECACHE\\n\"); exit(1); }", "exit(1); }")
add("cfg_prefill_msg", "BG", [(519, 519)], "        fprintf(stderr, \"K3GPU PREFILL ON:", ");")
add("cfg_start_msg", "BG", [(523, 523)], "    fprintf(stderr, \"K3GPU: %d cards, host expert cache %d GB, %d drive readers\\n\"", ");")
# selectors: variable parsing only. parse = the three lines up to the fatal check (built variant), lib = default library name, nb = the not-built variant, announce = the line after dlopen succeeded
for tag, comp, parse, lib, nb, ann in (("copyq", "COPYQ", (185, 187), 188, (225, 227), 200), ("slotcache", "SLOTCACHE", (244, 246), 247, (261, 263), 256), ("routetable", "ROUTETABLE", (444, 446), 447, (459, 461), 454)):
    add("sel_%s_parse" % tag, "BG", [parse], "    const char *m = getenv(\"K3GPU_%s\");" % comp, "exit(1); }")
    add("sel_%s_lib" % tag, "BG", [(lib, lib)], "    const char *lib = getenv(\"K3GPU_%s_LIB\");" % comp, ";")
    add("sel_%s_nb" % tag, "BG", [nb], "    const char *m = getenv(\"K3GPU_%s\");" % comp, ");")
    add("sel_%s_announce" % tag, "BG", [(ann, ann)], "    fprintf(stderr, \"K3GPU: ", ", lib);")

PINS = {
    "decls": "1edc85067e6e9298e67b7d9583911d11bacb0176c232af220bf52bd569169463",
    "decls_dc": "01bd1bb55762e078a77e41ebe732bc4ba3888c73d023c0f68049ee59ae4cea5b",
    "env_int": "7ba6ef7e1db7fdfd61f4fe8e191628c558b7e0863c37bf0a6ab3c097f14d508b",
    "card_of": "00f53c2ec30bfc6922d679ba8a75cd6e02753d45fcaf98d38b6f8c18c2cbdcc2",
    "force_routes": "ea17f5ae377c42adc49f1b3aa35778193281ee35c5c45564bac2f4d67aef81da",
    "print_summary": "70d7ecd88d61a7760bc18bfc7b77d9b8edd8e785c9e4f95878b21a955e08137e",
    "configure": "a9b1c2804c71bbdd793eb40b32a4fc800a336bf182cf602ad9cdac5ed7598404",
    "cfg_cards": "790a836c7720fc8e71363815663fec5aacb841b8910579836be9ef5077bb7305",
    "cfg_policy": "4e02e6ba0f7a1be695dd8e24e52f00506c420123bab93109f59d6fa2db68ab22",
    "cfg_chunks": "d97e0f208207e1f6f7cb1614d63c5bd8d317609fcb2557d41f0803d32a7406a7",
    "cfg_trace": "62ab730e5dfb9e175dc54766b69f33857f4e611e0568b149e708443e45d3fb03",
    "cfg_host": "0fabf67c27a8274012afff58980023a7ec025781caa179a5cf730555d94ada31",
    "cfg_rc_head": "792692537d7d546d621986795764d82e117e2ec94f128aeeb20f34faf7ebbd6e",
    "cfg_lookahead": "1bb9967b0056e0c8d79092396949ca2d8606b2872445bf5242fcca59ff847154",
    "cfg_pendcap": "9705ea360fdcc7278313bab2b58caba29197b5813015c7bcf9c6b7c7e5fcd8df",
    "cfg_vq": "451cab1108c626b9439767b61a90a35061cd0e94559717c8aa56f75476658828",
    "cfg_rc_msg": "6b095045840c9d037cdd078426294a93d60f6cc9c3d963edcaafaf3e07063db3",
    "cfg_layout": "aa1d423d91a8f19820b9c6bd906490d92edb5657a346c0678a70cd188d0d2eab",
    "cfg_prefill_flag": "72f50087a41672d938fa3463bee9aed19f0fa5ae25230972e38a7db56ae0a975",
    "cfg_prefill_check": "3b2092c254d4d8c2e622ad48fe9c3361364a121e0a9fcde6fe92f1bec3d16cbf",
    "cfg_prefill_msg": "ed82a4057d4b789e7e9230b41a26fb5d41696041a11675d914acf5db76cfead7",
    "cfg_start_msg": "b62795603f86fe9bdd89bad2ec648904b23bfe2c4f6e79fc44a4e3956c3e24ed",
    "sel_copyq_parse": "5687ed824a936c0b2c11d715fde13d3dcadbca78bab7009e58ef28c523bb499e",
    "sel_copyq_lib": "ddf6e685dd4e6b442277d9e65ac5160a28ef02eb52d60cdd39492acf887752a4",
    "sel_copyq_nb": "0e763097bda00f8e1105aa1765a8535a0ce7758695c0225268d932a3b5985aa5",
    "sel_copyq_announce": "f0c6974ed36e26d40877d0c0592dc5e973395c927db41578b6a5d4850c1f9388",
    "sel_slotcache_parse": "c9c4bebd52b1cf1ef63b9b2d51fbfd1e9be9b701e91340ec4225b235e91be171",
    "sel_slotcache_lib": "341eb0aadfb7c5985ae4118a0fb258e730f196c5f18fc37f452f2a3e5ca99f7f",
    "sel_slotcache_nb": "51b02709decc0769160c208aeb32a198a46228163a64f9ced96f057d1395b246",
    "sel_slotcache_announce": "69722336759ea0d699bc3c211ff0fcc7f77cfb0e6c4f4238808ea815df12a41c",
    "sel_routetable_parse": "45cff5e02771036af2df682691ba1bd97202af5c3ae85870214a80a3054717dd",
    "sel_routetable_lib": "b74e5a14916514131176376b10a950a668675f17ef1d2d8ede6730e6c617c55b",
    "sel_routetable_nb": "785cd9c5abcc13452cac87ac5c3e72b1ffebcf19da8100176a6461a99ce07ab6",
    "sel_routetable_announce": "8210928c66d70557ef53c1f03dc198b45aed9c6c5b452c398fe83c07e52495ff",
}


def fail(msg):
    sys.stderr.write("extract.py: FAIL: " + msg + "\n")
    sys.exit(1)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def split_lines(data):
    parts = data.split(b"\n")
    lines = [p + b"\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def load_blob(args, name, gitpath):
    if args.src_dir:
        p = os.path.join(args.src_dir, name)
        if not os.path.isfile(p):
            fail("source file not found: " + p)
        data = open(p, "rb").read()
        where = p
    else:
        r = subprocess.run(["git", "-C", args.repo, "show", "%s:%s" % (COMMIT, gitpath)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            fail("git show %s:%s failed: %s" % (COMMIT, gitpath, r.stderr.decode(errors="replace").strip()))
        data = r.stdout
        where = "git %s:%s" % (COMMIT, gitpath)
    if args.print_pins:
        print("blob %s sha256 %s" % (name, sha(data)))
    elif sha(data) != BLOB_PINS[name]:
        fail("%s has sha256 %s, expected %s (not the blob this table was written for)" % (where, sha(data), BLOB_PINS[name]))
    return data, where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..")))
    ap.add_argument("--out", default=os.path.join(HERE, "x"))
    ap.add_argument("--src-dir", default=None)
    ap.add_argument("--print-pins", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):
        os.remove(os.path.join(a.out, f))

    bg, bgw = load_blob(a, "backend-gpu.cu", BG_PATH)
    dc, dcw = load_blob(a, "decode.cuh", DC_PATH)
    L = {"BG": split_lines(bg), "DC": split_lines(dc)}
    print("source BG %s sha256 %s (%d lines)" % (bgw, sha(bg), len(L["BG"])))
    print("source DC %s sha256 %s (%d lines)" % (dcw, sha(dc), len(L["DC"])))

    out = {}
    info = []
    for name, f, ranges, start, end, func in S:
        lines = L[f]
        data = b""
        for (i0, i1) in ranges:
            if i0 < 1 or i1 > len(lines) or i1 < i0:
                fail("%s: bad line range %d-%d" % (name, i0, i1))
            data += b"".join(lines[i0 - 1:i1])
        first = lines[ranges[0][0] - 1].rstrip(b"\r\n").decode("latin-1")
        last = lines[ranges[-1][1] - 1].rstrip(b"\r\n").decode("latin-1")
        if not first.startswith(start):
            fail("%s: first line (source line %d) does not start with %r: %r" % (name, ranges[0][0], start, first[:100]))
        if end and not last.rstrip().endswith(end):
            fail("%s: last line (source line %d) does not end with %r: %r" % (name, ranges[-1][1], end, last[-100:]))
        if func:
            depth = 0; opened = False
            for i in range(ranges[0][0] - 1, ranges[-1][1]):
                depth += lines[i].count(b"{") - lines[i].count(b"}")
                opened = opened or b"{" in lines[i]
                if depth < 0 or (opened and depth == 0 and i != ranges[-1][1] - 1):
                    fail("%s: braces do not balance as one block (depth %d at source line %d)" % (name, depth, i + 1))
            if depth != 0:
                fail("%s: braces unbalanced at the end line (depth %d)" % (name, depth))
        out[name] = data
        info.append((name, f, ranges, len(data), sha(data)))

    pins_mode = a.print_pins
    if pins_mode:
        print("PINS = {")
        for name, f, ranges, n, h in info:
            print('    "%s": "%s",' % (name, h))
        print("}")
    else:
        for name, f, ranges, n, h in info:
            if name not in PINS:
                fail("slice %s has no pinned sha256 (run once with --print-pins and review)" % name)
            if PINS[name] != h:
                fail("slice %s sha256 %s differs from the pinned %s" % (name, h, PINS[name]))
        extra = set(PINS) - set(out)
        if extra:
            fail("pinned slices not produced: %s" % sorted(extra))

    for name, data in out.items():
        p = os.path.join(a.out, name + ".inc")
        with open(p, "wb") as fh:
            fh.write(data)
        with open(p, "rb") as fh:
            if fh.read() != data:
                fail("self-check: %s differs from the extracted bytes" % p)
    with open(os.path.join(a.out, "MANIFEST.txt"), "w", newline="\n") as fh:
        fh.write("# engcfg ref slices of %s (BG=%s, DC=%s); name, source, lines, bytes, sha256\n" % (COMMIT, BG_PATH, DC_PATH))
        for name, f, ranges, n, h in info:
            fh.write("%s %s %s %d %s\n" % (name, f, ",".join("%d-%d" % r for r in ranges), n, h))
    for name, f, ranges, n, h in info:
        print("%-24s %s lines %-28s %5d bytes sha256 %s" % (name, f, ",".join("%d-%d" % r for r in ranges), n, h))
    crlf = sum(d.count(b"\r\n") for d in out.values()); lf = sum(d.count(b"\n") for d in out.values())
    print("wrote %d slices to %s (CRLF lines %d of %d)" % (len(out), a.out, crlf, lf))


if __name__ == "__main__":
    main()
