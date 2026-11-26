"""Validate all local site pages, navigation, versions and shared assets."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, unquote, urlsplit


VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr", "use", "path"}


class Checker(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []
        self.ids = set()
        self.references = []
        self.navigation = []
        self.nav_count = 0
        self.in_nav = False
        self.nav_link = None
        self.redirect = False
        self.standalone_app = False
        self.code_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("code", "pre", "script", "style"):
            self.code_depth += 1
        attributes = dict(attrs)
        if tag == "body" and "clover-app-body" in attributes.get("class", "").split():
            self.standalone_app = True
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))
        identity = attributes.get("id")
        if identity:
            if identity in self.ids:
                self.errors.append("duplicate id: " + identity)
            self.ids.add(identity)
        for name in ("href", "src"):
            if attributes.get(name):
                self.references.append(attributes[name])
        if tag == "meta" and attributes.get("http-equiv", "").lower() == "refresh":
            target = re.search(r"url\s*=\s*(.+)", attributes.get("content", ""), re.I)
            if target:
                self.redirect = True
                self.references.append(target.group(1).strip(" '\""))
        if tag == "nav" and "site-nav" in attributes.get("class", "").split():
            self.in_nav = True
            self.nav_count += 1
        if self.in_nav and tag == "a":
            self.nav_link = [attributes.get("href", ""), []]

    def handle_data(self, data):
        if self.nav_link is not None:
            self.nav_link[1].append(data)
        if self.code_depth == 0 and "**" in data:
            self.errors.append("line %d: unrendered Markdown '**' in HTML text" % self.getpos()[0])

    def handle_endtag(self, tag):
        if tag in ("code", "pre", "script", "style") and self.code_depth > 0:
            self.code_depth -= 1
        if tag == "a" and self.nav_link is not None:
            target, pieces = self.nav_link
            self.navigation.append((target, " ".join("".join(pieces).split())))
            self.nav_link = None
        if tag == "nav":
            self.in_nav = False
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append("line %d: stray </%s>" % (self.getpos()[0], tag))
        elif self.stack[-1][0] != tag:
            self.errors.append("line %d: </%s> closes <%s> opened line %d"
                               % (self.getpos()[0], tag, self.stack[-1][0], self.stack[-1][1]))
            self.stack.pop()
        else:
            self.stack.pop()


def resolve_target(site, page, target):
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None, parsed
    path = unquote(parsed.path)
    if not path:
        destination = page
    elif path.startswith("/"):
        destination = site / path.lstrip("/")
    else:
        destination = page.parent / path
    destination = destination.resolve()
    if destination.is_dir():
        destination /= "index.html"
    return destination, parsed


def check_site(root):
    root = Path(root).resolve()
    site = root / "site"
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    pages = {}
    errors = []
    signatures = {}
    for page in sorted(site.rglob("*.html")):
        parser = Checker()
        parser.feed(page.read_text(encoding="utf-8"))
        parser.close()
        pages[page] = parser
        label = page.relative_to(site).as_posix()
        errors.extend(label + ": " + error for error in parser.errors)
        if parser.stack:
            errors.append(label + ": unclosed HTML tags")
        if not parser.redirect and not parser.standalone_app and parser.nav_count != 1:
            errors.append(label + ": expected one primary navigation")
        if parser.nav_count:
            signature = []
            for target, text in parser.navigation:
                destination, parsed = resolve_target(site, page, target)
                normalized = str(destination) + "#" + parsed.fragment if destination else target
                signature.append((normalized, text))
            signatures[page] = signature

    home = site / "index.html"
    if home not in pages:
        errors.append("missing site index.html")
    baseline = signatures.get(home, [])
    if not baseline:
        errors.append("home navigation is empty")
    for page, signature in signatures.items():
        if signature != baseline:
            errors.append(page.relative_to(site).as_posix() + ": navigation differs from home")

    for page, parser in pages.items():
        label = page.relative_to(site).as_posix()
        for target in parser.references:
            destination, parsed = resolve_target(site, page, target)
            if destination is None:
                continue
            if not destination.is_relative_to(site) or not destination.is_file():
                errors.append(label + ": missing or outside site: " + target)
                continue
            if parsed.fragment and destination in pages:
                if unquote(parsed.fragment) not in pages[destination].ids:
                    errors.append(label + ": dead anchor: " + target)
            if destination.suffix in (".css", ".js"):
                if parse_qs(parsed.query).get("v") != [version]:
                    errors.append(label + ": asset version must match VERSION: " + target)

    assets_checked = 0
    for copy in sorted((site / "assets").glob("*.svg")):
        candidates = list((root / "assets").rglob(copy.name))
        if len(candidates) > 1:
            errors.append("ambiguous canonical shared asset: " + copy.name)
        elif candidates:
            canonical = candidates[0]
            assets_checked += 1
            if canonical.read_bytes() != copy.read_bytes():
                errors.append("shared asset differs: " + copy.name)
        elif copy.name.startswith(("case-study-", "reference-")):
            errors.append("missing canonical shared asset: " + copy.name)

    # Guard against withheld public app/download directories or binary installers in public/release folders
    for forbidden_dir in ("clover-ai", "admin", "download"):
        if (site / forbidden_dir).exists():
            errors.append(f"forbidden public directory present in site/: {forbidden_dir}")
    checked_bin_roots = [
        p for p in (
            site,
            root / "download",
            root / "release",
            root / "clover-ai-system" / "mobile",
            root / "clover-ai-system" / "studio" / "studio" / "dist",
        )
        if p.is_dir()
    ]
    skip_parts = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", ".gradle"}
    for scan_dir in checked_bin_roots:
        for bin_ext in ("*.apk", "*.aab", "*.msix", "*.msixupload", "*.exe"):
            for bad_bin in scan_dir.rglob(bin_ext):
                if not (skip_parts & set(bad_bin.parts)):
                    errors.append("withheld installer binary present in tree: " + bad_bin.relative_to(root).as_posix())

    # Validate raw.githubusercontent.com paths in site/llms.txt when present
    llms_txt = site / "llms.txt"
    if llms_txt.is_file():
        raw_pfx = "https://raw.githubusercontent.com/santhoshsathish94/clover-framework/main/"
        for m_url in re.finditer(r"https://raw\.githubusercontent\.com/santhoshsathish94/clover-framework/main/([^\s)\]]+)", llms_txt.read_text(encoding="utf-8")):
            rel_p = m_url.group(1)
            if not (root / rel_p).is_file():
                errors.append(f"llms.txt: missing repository file for {raw_pfx}{rel_p}")

    # Validate byte-exact parity between root AGENTS.md and site/getting-started/index.html copy payload
    agents_md = root / "AGENTS.md"
    gs_html = site / "getting-started" / "index.html"
    if agents_md.is_file() and gs_html.is_file():
        gs_txt = gs_html.read_text(encoding="utf-8")
        m_raw = re.search(r'var\s+RAW_AGENTS_MD\s*=\s*("(?:\\.|[^"\\])*");', gs_txt)
        if not m_raw:
            errors.append("getting-started/index.html: missing RAW_AGENTS_MD payload")
        else:
            try:
                embedded = json.loads(m_raw.group(1)).replace("\r\n", "\n")
                source_md = agents_md.read_text(encoding="utf-8").replace("\r\n", "\n")
                if embedded != source_md:
                    errors.append("getting-started/index.html: RAW_AGENTS_MD differs from root AGENTS.md")
            except ValueError as exc:
                errors.append(f"getting-started/index.html: invalid RAW_AGENTS_MD JSON literal ({exc})")

    return {"pages": len(pages), "navigation_pages": len(signatures),
            "shared_assets": assets_checked, "errors": errors}


def main():
    report = check_site(Path(__file__).resolve().parents[1])
    print("pages checked        : %d" % report["pages"])
    print("navigation compared  : %d" % report["navigation_pages"])
    print("shared assets checked: %d" % report["shared_assets"])
    print("site errors          : %d" % len(report["errors"]))
    for error in report["errors"]:
        print("   " + error)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
