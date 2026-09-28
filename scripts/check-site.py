"""Check the site: HTML well-formedness, internal links, and nav consistency."""
import html.parser
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr", "use", "path"}


class Checker(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag):
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


pages = []
for dirpath, _, filenames in os.walk(SITE):
    for f in filenames:
        if f.endswith(".html"):
            pages.append(os.path.join(dirpath, f))

bad_html, bad_link, no_nav = [], [], []
for p in sorted(pages):
    rel = os.path.relpath(p, SITE).replace("\\", "/")
    text = open(p, encoding="utf-8").read()

    c = Checker()
    c.feed(text)
    for e in c.errors:
        bad_html.append("%s %s" % (rel, e))
    if c.stack:
        bad_html.append("%s unclosed: %s" % (rel, [t for t, _ in c.stack]))

    base = os.path.dirname(p)
    for m in re.finditer(r'(?:href|src)="([^"]+)"', text):
        t = m.group(1)
        if t.startswith(("http://", "https://", "mailto:", "#", "data:")):
            continue
        t = t.split("#")[0].split("?")[0]
        if not t:
            continue
        dest = os.path.normpath(os.path.join(base, t.replace("/", os.sep)))
        if os.path.isdir(dest):
            dest = os.path.join(dest, "index.html")
        if not os.path.exists(dest):
            bad_link.append("%s -> %s" % (rel, m.group(1)))

    if 'class="site-nav"' in text and "clover-k3/" not in text:
        no_nav.append(rel)

print("pages checked        : %d" % len(pages))
print("html structure errors: %d" % len(bad_html))
for b in bad_html:
    print("   " + b)
print("broken internal refs : %d" % len(bad_link))
for b in bad_link:
    print("   " + b)
print("nav missing Clover K3: %d" % len(no_nav))
for b in no_nav:
    print("   " + b)
sys.exit(1 if (bad_html or bad_link or no_nav) else 0)
