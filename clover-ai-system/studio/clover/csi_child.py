"""CSI (research/semantic-interface, frozen v0.1r3) as a child process of the Studio, on a free loopback port.

It runs the framework's own csi_service.py (copied into clover/csi_service.py with the csi package, byte-identical)
with spaCy en_core_web_trf 3.8.0, the frozen verb lemmatiser (data/csi/morph-verb-v1.1) and Open English WordNet 2025
(data/csi/english-wordnet-2025.xml.gz), as in Experiment 1f. Semantics reads each message through it (declined tools,
claims read from the main clause) and personal learning uses its lemmas. While it starts, or when it is missing or
stops, Clover works as without CSI (semantics.csi_reader returns None)."""
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARSER = "en_core_web_trf"
MORPH = ("morph-verb-v1.1", ("morphology.json", "morphology.npz"))
OEWN = "english-wordnet-2025.xml.gz"


def data_dir(base):
    return Path(base) / "data" / "csi"


def _csi_python():
    p = os.environ.get("CLOVER_CSI_PYTHON")
    if p and Path(p).is_file():
        return Path(p)
    py = Path(sys.executable)
    if py.name.lower() == "pythonw.exe" and (py.parent / "python.exe").is_file():
        py = py.parent / "python.exe"
    return py


def missing(base):
    """What CSI still needs on this machine (names), [] when everything is there."""
    d = data_dir(base)
    out = [f"{MORPH[0]}/{f}" for f in MORPH[1] if not (d / MORPH[0] / f).is_file()]
    if not (d / OEWN).is_file():
        out.append(OEWN)
    if not (HERE / "csi_service.py").is_file() or not (HERE / "csi" / "analyze.py").is_file():
        out.append("csi package")
    return out


def remote_csi_url(base=None):
    """On Android owner devices where local spaCy cannot run inside Chaquopy, use the k3 CSI service over HTTPS."""
    import paths
    if not paths.ANDROID:
        return ""
    try:
        import k3_host_engine
        if k3_host_engine.is_owner_device(base):
            return f"{k3_host_engine.K3_URLS[0]}/api/worker/csi"
    except Exception:
        pass
    return ""


class CSIService:
    def __init__(self, base, threads=2):
        self.base, self.threads = Path(base), int(threads)
        self.proc, self.port, self.state, self.detail, self.started = None, None, "off", "", None
        self._remote_url = ""
        self.lock = threading.Lock()

    @property
    def url(self):
        if self.state != "ready":
            return ""
        return self._remote_url or f"http://127.0.0.1:{self.port}"

    def describe(self):
        return {"state": self.state, "detail": self.detail, "parser": PARSER, "url": self.url,
                "lemmatiser": MORPH[0], "dictionary": "Open English WordNet 2025"}

    def start(self, timeout_s=240):
        """Start in the background; state goes starting -> ready (or missing / failed)."""
        with self.lock:
            if self.state in ("starting", "ready"):
                return
            need = missing(self.base)
            if need:
                rurl = remote_csi_url(self.base)
                if rurl:
                    self._remote_url = rurl
                    self.state, self.detail = "ready", "connected to k3 CSI service"
                    return
                self.state, self.detail = "missing", "not installed: " + ", ".join(need) + " (run Clover AI Studio setup)"
                return
            self.state, self.detail, self.started = "starting", "loading the parser and the dictionary", time.time()
        threading.Thread(target=self._run, args=(timeout_s,), daemon=True, name="csi-start").start()

    def _run(self, timeout_s):
        import llamacpp_engine as L
        import paths
        d = data_dir(self.base)
        os.environ["CSI_MORPH"] = str(d / MORPH[0])
        os.environ["CSI_DICT"] = str(d / OEWN)
        env = dict(os.environ, CSI_PARSER=PARSER, CSI_MORPH=str(d / MORPH[0]), CSI_DICT=str(d / OEWN),
                   OMP_NUM_THREADS=str(self.threads), MKL_NUM_THREADS=str(self.threads), HF_HUB_OFFLINE="1",
                   PYTHONIOENCODING="utf-8")
        for k in ("CSI_SENSES", "CSI_CLOSENESS"):  # the frozen v0.1r3 reading: lemmas, no word senses
            env.pop(k, None)
        self.port = L._free_port()
        if paths.ANDROID:
            try:
                from http.server import ThreadingHTTPServer
                import csi_service
                from csi import analyze as csi_analyze_fn
                csi_mod = sys.modules.get(csi_analyze_fn.__module__)
                if csi_mod is not None:
                    csi_mod.CSI_MORPH = str(d / MORPH[0])
                    csi_mod.CSI_DICT = str(d / OEWN)
                csi_analyze_fn("Please search the web.")
                self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), csi_service.Handler)
                threading.Thread(target=self._httpd.serve_forever, daemon=True, name="csi-inproc").start()
                self.state = "ready"
                self.detail = f"ready in {time.time() - self.started:.0f} s"
                return
            except Exception as exc:
                self.state, self.detail = "failed", f"could not start in-process CSI: {exc}"
                return
        py = _csi_python()
        log = self.base / "logs" / "csi-service.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        try:
            out = open(log, "w", encoding="utf-8", errors="replace")
            self.proc = subprocess.Popen([str(py), str(HERE / "csi_service.py"), "--port", str(self.port)], env=env,
                                         stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(HERE),
                                         creationflags=0x08000000 if os.name == "nt" else 0)
            self._job = L._kill_with_parent(self.proc)
        except OSError as exc:
            self.state, self.detail = "failed", f"could not start: {exc}"
            return
        while time.time() - self.started < timeout_s:
            if self.proc.poll() is not None:
                self.state, self.detail = "failed", f"stopped while starting (exit {self.proc.returncode}); see {log}"
                return
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        self.state = "ready"
                        self.detail = f"ready in {time.time() - self.started:.0f} s"
                        return
            except OSError:
                pass
            time.sleep(1)
        self.state, self.detail = "failed", f"not ready in {timeout_s} s; see {log}"
        self.stop(state="failed")

    def alive(self):
        if self.state == "ready" and self.proc is not None and self.proc.poll() is not None:
            self.state, self.detail = "failed", f"stopped (exit {self.proc.returncode})"
        return self.state == "ready"

    def stop(self, state="off"):
        with self.lock:
            p, self.proc = self.proc, None
            if p is not None and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
            self.state = state
            if state == "off":
                self.detail = ""
