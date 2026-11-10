"""Verification suite for the unified Clover AI Studio UI across Laptop, Mobile, and Web."""
import sys
from pathlib import Path

STUDIO = Path(__file__).resolve().parent.parent
REPO = STUDIO.parent.parent
APP_HTML = STUDIO / "clover" / "web" / "app.html"
SITE_HTML = REPO / "site" / "clover-ai" / "index.html"
ANDROID_HTML = STUDIO.parent / "mobile" / "android-app" / "assets" / "index.html"

ok = True
n = 0


def check(label: str, cond: bool) -> bool:
    global ok, n
    n += 1
    status = "PASS" if cond else "FAIL"
    print(f"{status} {label}")
    if not cond:
        ok = False
    return cond


app = APP_HTML.read_text(encoding="utf-8")
site = SITE_HTML.read_text(encoding="utf-8")
android = ANDROID_HTML.read_text(encoding="utf-8")

# 1. Theme colors and Apple design palette preserved
for token in (
    "--cream: #F7F3E9",
    "--green: #176B2C",
    "--deep: #0D4D20",
    "--leaf: #83B83D",
    "--rail: #F1ECDF",
    "--surface: #FFFDF8",
):
    check(f"Theme token preserved: {token}", token in app)

# 2. Left sidebar brand header & 6 primary navigation items using existing icons only
check(
    "Left sidebar brand header has Clover AI Studio and v5.0.2",
    '<span class="brand__title">Clover AI Studio</span>' in app
    and '<span class="brand__ver">v5.0.2</span>' in app,
)
for nav_id, label, icon_id in (
    ("chat", "Chat", "#i-chat"),
    ("workspaces", "Workspaces", "#i-folder"),
    ("memory", "Memory", "#i-models"),
    ("tools", "Tools", "#i-code"),
    ("learning", "Learning", "#i-spark"),
    ("settings", "Settings", "#i-settings"),
):
    check(
        f"Sidebar navigation item '{label}' ({nav_id}) uses existing icon {icon_id}",
        f'data-nav="{nav_id}"' in app and f'<span>{label}</span>' in app and f'href="{icon_id}"' in app,
    )

check(
    "Sidebar footer displays Clover Ready with green status dot",
    'id="status-dot"' in app and 'id="status-text">Clover Ready</span>' in app,
)

# 3. Welcome Hero, Composer, 5-Stage Clover Cycle Strip, Collab Pill, and 3 Feature Cards
check(
    "Welcome Hero displays Clover emblem, Clover title, and Work · Create · Learn · Grow tagline",
    'class="hero__logo" src="icon.png"' in app
    and '<h1 id="greet">Clover</h1>' in app
    and "Work &middot; Create &middot; Learn &middot; Grow" in app,
)
check(
    "Composer has 'What would you like to do?' placeholder and bottom toolbar (#attach, #screen-btn, #clip-btn, #send)",
    'placeholder="What would you like to do?"' in app
    and 'id="attach"' in app
    and 'id="screen-btn"' in app
    and 'id="clip-btn"' in app
    and 'id="send"' in app,
)

for stage, subtitle, icon_id in (
    ("Context", "Understand", "#i-search"),
    ("Direction", "Plan together", "#i-models"),
    ("Execution", "Do the work", "#i-code"),
    ("Outcome", "Check the result", "#i-success"),
    ("Growth", "Learn and improve", "#i-graph"),
):
    check(
        f"5-Stage Clover Cycle strip includes '{stage}' ({subtitle})",
        f"<b>{stage}</b><small>{subtitle}</small>" in app and f'href="{icon_id}"' in app,
    )

check(
    "Tagline pill shows 'System · Human · AI working together'",
    "System &middot; Human &middot; AI working together" in app,
)

for card_nav, title, desc in (
    ("memory", "Memory", "Your work, knowledge and context"),
    ("tools", "Tools", "Search, files, code and apps"),
    ("learning", "Learning", "Gets better with you"),
):
    check(
        f"Bottom feature card '{title}' ({desc}) navigates to {card_nav}",
        f'class="fcard" data-nav-open="{card_nav}"' in app
        and f"<b>{title}</b><small>{desc}</small>" in app,
    )

# 4. Dedicated Memory page and consistent cross-platform sync (Laptop, Mobile, Web)
check(
    "Dedicated Memory page [data-page='memory'] is present in sheet",
    '<section class="page" data-page="memory" hidden>' in app
    and "memory: 'Memory'" in app
    and 'id="memory-personal-summary"' in app,
)
check(
    "Android assets/index.html is byte-identical to studio/clover/web/app.html",
    android == app,
)
expected_site = app.replace("__CLOVER_PLATFORM__", "web").replace(' nonce="__CSP_NONCE__"', "")
check(
    "Web site/clover-ai/index.html is in sync with studio/clover/web/app.html",
    site == expected_site,
)

print(f"{n} checks" + (" passed" if ok else ", SOME FAILED"))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
