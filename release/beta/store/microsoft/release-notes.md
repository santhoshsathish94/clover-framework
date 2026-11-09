Clover AI Studio v5.0.2 (Package Version 5.0.2.0) — Release:
• Checked writes: Clover reads a written file back, and the step says "Checked on disk" only when the file holds what was written.
• If Clover stopped in the middle of a step (for example the laptop restarted), the next answer says which step did not finish. It is never re-run by itself.
• Clover's decision helper can only make Clover more careful; it never approves an action on its own.
• Packaged as an x64 MSIX / `.msixupload` bundle (`SanthoshNarayanan.CloverAIStudio`, `5.0.2.0`) alongside the Authenticode-signed standalone x64 installer (`CloverAIStudio-Setup.exe`).
