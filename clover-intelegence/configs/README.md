# Configuration

[tokenizer_config.json](../code/client/bin/configs/tokenizer_config.json) now lives
only inside code/client/bin/configs for this runtime, on AX102 and in the workspace.
The previous server path here and its original /root/k3model alias were explicitly
removed. This directory contains documentation only, not runtime configuration.
No contents were modified. It defines16added token spellings and the model's
control-token settings. No duplicate configuration file remains here locally.

The [C exact mapper](../code/client/README.md) uses the added-token spellings without
adding BOS/EOS automatically or treating input text as an executable command.