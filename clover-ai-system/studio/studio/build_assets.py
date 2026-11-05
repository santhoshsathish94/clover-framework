"""Build the Studio assets on the dev machine: app icons, the mascot and the VS Code extension package (.vsix).

Icons come from the Clover icon pack (clover-framework/assets/robo-icon): the Windows icon for the exe and installer,
the clover mark for the Studio page, its favicon and the VS Code extension, and Robo Clover for the empty chat.
A .vsix is an OPC zip: [Content_Types].xml + extension.vsixmanifest + extension/<files>. Built here without node/vsce.
Usage: python studio/build_assets.py   (from the laptop/ folder; needs Pillow for the mascot)
"""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # laptop/
EXT = ROOT / "vscode-extension"
DIST = ROOT / "studio" / "dist"
WEB = ROOT / "clover" / "web"
PACK = Path(r"C:\personal\oss\clover-framework\assets\robo-icon")
CREAM = (247, 243, 233)  # the pack's surface colour, the Studio page background


def build_icons():
    copies = {"app/windows/clover.ico": ROOT / "studio" / "clover.ico",
              "brand/clover-mark-128.png": EXT / "icon.png",
              "brand/clover-mark-256.png": WEB / "icon.png",
              "brand/clover-mark.svg": WEB / "favicon.svg",
              "brand/clover-mark-512.png": ROOT / "studio" / "icon-512.png"}
    for name, dest in copies.items():
        shutil.copyfile(PACK / name, dest)
        print("icon:", name, "->", dest)


def build_mascot(width=480):
    """Robo Clover for the empty chat: cropped to the robot, 480 px wide (240 px on screen at 2x). The render's
    background is a pinker white than the page, so light pixels are shifted to the page colour and the edges fade
    into it."""
    from PIL import Image
    im = Image.open(PACK / "robo-clover" / "robo-clover-master.png").convert("RGB").crop((120, 0, 1320, 1024))
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    bg = im.getpixel((2, im.height // 2))
    shift = [c - b for c, b in zip(CREAM, bg)]
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r, g, b = px[x, y]
            w = min(1.0, max(0.0, ((0.299 * r + 0.587 * g + 0.114 * b) / 255 - 0.72) / 0.2))  # light pixels only
            px[x, y] = tuple(min(255, max(0, round(c + w * s))) for c, s in zip((r, g, b), shift))
    # Image.radial_gradient is 0 at the centre and 255 at the corners, about 180 at the middle of each edge
    edge = 255 / 2 ** 0.5
    inner, outer = 0.88 * edge, 1.0 * edge  # the background already matches: fade only the rim, not the leaf or cape

    def fade(v):  # -> opacity, smoothstep from the inner to the outer ellipse
        t = min(1.0, max(0.0, (v - inner) / (outer - inner)))
        return round(255 * (1 - t * t * (3 - 2 * t)))
    mask = Image.radial_gradient("L").resize(im.size, Image.BICUBIC).point(fade)
    out = Image.composite(im, Image.new("RGB", im.size, CREAM), mask)
    dest = WEB / "robo-clover.webp"
    out.save(dest, "WEBP", quality=86, method=6)
    print("mascot:", dest, out.size, dest.stat().st_size, "bytes")


def build_vsix():
    pkg = json.loads((EXT / "package.json").read_text(encoding="utf-8"))
    ident = f"{pkg['publisher']}.{pkg['name']}"
    files = ["package.json", "extension.js", "README.md", "icon.png"]
    manifest = f"""<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011" xmlns:d="http://schemas.microsoft.com/developer/vsx-schema-design/2011">
  <Metadata>
    <Identity Language="en-US" Id="{pkg['name']}" Version="{pkg['version']}" Publisher="{pkg['publisher']}" />
    <DisplayName>{pkg['displayName']}</DisplayName>
    <Description xml:space="preserve">{pkg['description']}</Description>
    <Tags>ai,chat,clover,lfm3,omni,local</Tags>
    <Categories>{','.join(pkg['categories'])}</Categories>
    <GalleryFlags>Public</GalleryFlags>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{pkg['engines']['vscode']}" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionDependencies" Value="" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionPack" Value="" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionKind" Value="workspace" />
      <Property Id="Microsoft.VisualStudio.Code.LocalizedLanguages" Value="" />
      <Property Id="Microsoft.VisualStudio.Services.GitHubFlavoredMarkdown" Value="true" />
    </Properties>
    <Icon>extension/icon.png</Icon>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code" />
  </Installation>
  <Dependencies />
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true" />
    <Asset Type="Microsoft.VisualStudio.Services.Content.Details" Path="extension/README.md" Addressable="true" />
    <Asset Type="Microsoft.VisualStudio.Services.Icons.Default" Path="extension/icon.png" Addressable="true" />
  </Assets>
</PackageManifest>
"""
    content_types = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension=".json" ContentType="application/json" />
  <Default Extension=".js" ContentType="application/javascript" />
  <Default Extension=".md" ContentType="text/markdown" />
  <Default Extension=".png" ContentType="image/png" />
  <Default Extension=".vsixmanifest" ContentType="text/xml" />
</Types>
"""
    DIST.mkdir(parents=True, exist_ok=True)
    out = DIST / f"{pkg['name']}-{pkg['version']}.vsix"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("extension.vsixmanifest", manifest)
        for f in files:
            z.write(EXT / f, f"extension/{f}")
    print("vsix:", out, ident, out.stat().st_size, "bytes")


def build_web_site():
    site_ai = Path(r"C:\personal\oss\clover-framework\site\clover-ai")
    if not site_ai.is_dir():
        return
    html = (WEB / "app.html").read_text(encoding="utf-8")
    html = html.replace("__CLOVER_PLATFORM__", "web").replace(' nonce="__CSP_NONCE__"', "")
    (site_ai / "index.html").write_text(html, encoding="utf-8", newline="\n")
    for name in ("icon.png", "favicon.svg", "robo-clover.webp"):
        shutil.copyfile(WEB / name, site_ai / name)
    print("web:", site_ai / "index.html", len(html), "chars")


if __name__ == "__main__":
    build_icons()
    build_mascot()
    build_vsix()
    build_web_site()
