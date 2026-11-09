Clover AI Studio v5.0.2 (versionCode 16, targetSdk 36) — Release:
• Fix: on phones that load Clover's planning layer, an answer could stop with an error. Fixed.
• Checked writes: Clover reads a file back and says "Checked on disk" only when it holds what was written.
• If Clover was closed mid-step, the next answer says which step did not finish. It is never re-run by itself.
• Clover's decision helper can only make it more careful; it never approves an action.
