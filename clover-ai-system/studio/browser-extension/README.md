# Send to Clover (browser extension)

Sends the page you are reading in Chrome or Edge to Clover AI Studio on the same computer. Nothing is read until you
click the extension's button (or "Send page to Clover" in the page's right-click menu); then only that tab is read
(`activeTab`) and handed to Clover's host program over native messaging. No server is involved and nothing runs in the
background.

1. In Clover AI Studio: Settings > Tools and search > Send pages from your browser > **Set up for Chrome and Edge**
   (writes a host manifest under the Clover folder and one key per browser under `HKCU\Software\...\NativeMessagingHosts`;
   no administrator rights). **Remove** undoes it.
2. Open `chrome://extensions` or `edge://extensions`, turn on Developer mode, **Load unpacked**, pick this folder.
   The `key` in `manifest.json` fixes the ID to `eibfjkaeioghjapmaejcingainoiepdf`, the only caller the host accepts.
   It is a development key; a store listing has its own key, and `EXTENSION_ID` in `clover/browser_host.py` must match.
3. On a page, click the Clover button. The page waits above Clover's message box: **Add to message** attaches it as a
   document (readable text, title, site, date and link), or dismiss it.

Pages are web content: Clover treats them as material to read, never as instructions.
