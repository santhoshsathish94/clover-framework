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

# 3. Welcome Hero, Composer, 5-Stage Clover Cycle Strip, and Collab Pill
check(
    "Welcome Hero displays Robo Clover mascot in center (and icon.png in sidebar brand), Clover title, and Work · Create · Learn · Grow tagline",
    '<img class="mascot" src="robo-clover.webp"' in app
    and 'class="hero__logo"' not in app
    and '<h1 id="greet">Clover</h1>' in app
    and "Work &middot; Create &middot; Learn &middot; Grow" in app,
)
check(
    "Top bar stays pinned at top in empty-chat (.stream margin-top:auto, .dock margin-bottom:auto) and keeps #set-btn always present across Web, Mobile, and Laptop",
    ".app.empty-chat .stream { flex: 0 0 auto; overflow: visible; margin-top: auto; }" in app
    and ".app.empty-chat .dock { margin-bottom: auto; }" in app
    and 'id="set-btn" aria-label="Settings" title="Settings"><svg class="ic" aria-hidden="true"><use href="#i-settings"/></svg></button>' in app
    and "#set-btn { display: none" not in app
    and ".app:not(.rail-off) #set-btn" not in app,
)
check(
    "Steps box (.work) is always collapsed after work is done (botEl calls workEl(false, false))",
    "var w = workEl(false, false); w.label(workSummary(m));" in app
    and "openByDefault" not in app,
)
check(
    "Single click-only /api/browser/open helper (openBrowserClick) preserves browser-first invariant",
    app.count("/api/browser/open") == 1 and "function openBrowserClick(body, fallbackUrl)" in app,
)
check(
    "Composer has 'What would you like to do?' placeholder and bottom toolbar (#attach, #screen-btn, #clip-btn, #send) without duplicating files/screenshot in #senses-menu",
    'placeholder="What would you like to do?"' in app
    and 'id="attach"' in app
    and 'id="screen-btn"' in app
    and 'id="clip-btn"' in app
    and 'id="send"' in app
    and 'data-sense="files"' not in app
    and 'data-sense="screenshot"' not in app
    and 'data-sense="screen_record"' in app
    and 'data-sense="camera"' in app
    and 'data-sense="microphone"' in app,
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
check(
    "Main screen does not duplicate Memory, Tools, and Learning cards (single navigation in sidebar)",
    'class="feature-grid"' not in app and 'class="fcard"' not in app,
)
check(
    "#senses-menu flips downward (.drop-down) when space above is limited or in empty-chat so it never clips",
    ".senses-pop.drop-down" in app and "m.classList.toggle('drop-down', dropDown)" in app,
)

# 4. Dedicated Memory page, 2-Pane Layout, Left-Pane Submenus, File Viewer, and cross-platform sync
home_sec = app.split('<section class="page" data-page="home">', 1)[1].split("</section>", 1)[0]
check(
    "Settings page [data-page='home'] omits duplicate Workspaces/Memory/Tools/Learning rows and keeps Model, Account, Storage, Debug, and Advanced",
    'data-go="workspaces"' not in home_sec
    and 'data-go="memory"' not in home_sec
    and 'data-go="tools"' not in home_sec
    and 'data-go="learning"' not in home_sec
    and all(f'data-go="{k}"' in home_sec for k in ("model", "account", "storage", "debug", "advanced")),
)
check(
    "Dedicated Memory page [data-page='memory'] owns clover-ai-studio persistent memory (#mem-parent-view, #mem-conn, #mem-gh-token, #memory-personal-summary) and has #mem-file-slot for file viewing",
    '<section class="page" data-page="memory" hidden>' in app
    and 'id="mem-parent-view"' in app
    and 'id="mem-conn"' in app
    and 'id="mem-gh-token"' in app
    and 'id="memory-personal-summary"' in app
    and 'id="mem-file-slot"' in app,
)
check(
    "Strict 2-pane layout: #sheet lives inside <section class=\"main\"> and .app.view-page replaces #stream and .dock in the center pane",
    app.index('<section class="main">') < app.index('id="sheet"')
    and ".app.view-page .dock { display: none !important; }" in app
    and 'class="sheet-bd"' not in app,
)
check(
    "Left pane (#rail) has collapsible Chat submenu (#rail-chat-sub), Workspaces submenu (#rail-ws-sub), Memory submenu (#rail-mem-sub), and Tools submenu (#rail-tools-sub -> Terminal, Browser, Camera, Microphone, VS Code)",
    'id="rail-chat-sub"' in app
    and 'id="rail-ws-sub"' in app
    and 'id="rail-mem-sub"' in app
    and 'id="rail-tools-sub"' in app
    and all(f'data-tool-app="{t}"' in app for t in ("terminal", "browser", "camera", "microphone", "vscode"))
    and app.index('id="rail"') < app.index('id="rail-chat-sub"') < app.index('id="chats"') < app.index('id="rail-ws-sub"') < app.index('id="vsc-tree"') < app.index('id="rail-mem-sub"') < app.index('id="mem-tree"') < app.index('id="rail-tools-sub"') < app.index('id="tools-tree"') < app.index('<section class="main">'),
)
check(
    "Center pane Workspaces page is scoped to user's own project repos and separates parent workspace settings (#ws-parent-view) from File Viewer (#vsc-workbench hidden by default)",
    app.index('data-page="workspaces"') < app.index('id="ws-parent-view"') < app.index('id="vsc-workbench"')
    and 'id="vsc-workbench" aria-label="Workspace file viewer" hidden' in app
    and "#ws-parent-view[hidden], #mem-parent-view[hidden], #tools-parent-view[hidden], .vsc-view[hidden], .tool-app[hidden] { display: none !important; }" in app
    and "function showWorkspaceParentView()" in app
    and "function showWorkspaceFileViewer(fileTitle)" in app
    and "function showMemoryParentView()" in app
    and "function showMemoryFileViewer(fileTitle)" in app,
)
tools_sec = app.split('<section class="page" data-page="tools" hidden>', 1)[1].split("</section>", 1)[0]
check(
    "Tools page [data-page='tools'] provides interactive center-pane apps (#tool-app-terminal, #tool-app-browser, #tool-app-camera, #tool-app-microphone, #tool-app-vscode) and parent overview (#tools-parent-view)",
    all(rid in tools_sec for rid in ("tools-parent-view", "tool-row-terminal", "tool-row-browser", "tool-row-camera", "tool-row-microphone", "tool-row-vscode", "tool-row-github"))
    and all(aid in tools_sec for aid in ("tool-app-terminal", "tool-app-browser", "tool-app-camera", "tool-app-microphone", "tool-app-vscode"))
    and all(pid in tools_sec for pid in ("tool-perm-terminal", "tool-perm-browser", "tool-perm-camera", "tool-perm-vscode"))
    and "function openToolApp(appName)" in app
    and "/api/tools/terminal" in app
    and "/api/tools/browser" in app,
)

# 5. Backend workspace tree, file viewer, local repo fallback, and secret blocking
import tempfile  # noqa: E402

sys.path.insert(0, str(STUDIO / "clover"))
sys.path.insert(0, str(STUDIO))
from clover import workspaces as ws_mod  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    td_path = Path(td)
    ws = ws_mod.Workspaces(td_path / "workspaces.json", td_path / "clones", td_path / "app")
    tree_res = ws.browse_tree("__local_repo__", "", local_repo_dir=REPO)
    check(
        "Workspaces.browse_tree lists default local repo (__local_repo__) before GitHub is connected",
        tree_res.get("source") == "local_repo"
        and any(e.get("name") == "AGENTS.md" for e in tree_res.get("entries", [])),
    )
    file_res = ws.read_workspace_file("__local_repo__", "AGENTS.md", local_repo_dir=REPO)
    check(
        "Workspaces.read_workspace_file reads text files with language metadata from local repo",
        file_res.get("name") == "AGENTS.md"
        and file_res.get("secret_blocked") is False
        and len(file_res.get("content", "")) > 100,
    )
    secret_res = ws.read_workspace_file("__local_repo__", ".env", local_repo_dir=REPO)
    vault_res = ws.read_workspace_file("__local_repo__", "vault/keys.json", local_repo_dir=REPO)
    check(
        "Workspaces.read_workspace_file blocks secret files and vault directory",
        secret_res.get("secret_blocked") is True and vault_res.get("secret_blocked") is True,
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
