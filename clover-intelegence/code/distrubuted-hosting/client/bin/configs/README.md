# Client Configuration

[tokenizer_config.json](tokenizer_config.json) contains the unchanged added-token
spellings used by the C exact vocabulary mapper. It lives inside the bin runtime
so the mapper does not depend on any configuration path outside code/client/bin.

An identical local copy is included. Previous server configuration aliases were
explicitly removed; use this bin/configs file. The client does not insert BOS/EOS tokens
automatically or perform BPE splitting.