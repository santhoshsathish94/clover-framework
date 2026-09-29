"""Validate all local site pages, navigation, versions and shared assets."""
from html.parser import HTMLParser
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

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
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

    def handle_endtag(self, tag):
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
        if not parser.redirect and parser.nav_count != 1:
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
