# Server Dataset

The redundant main-dataset qkv-all/layer-0 payload has been deleted after full
byte equality and SHA256 verification. The one canonical payload remains
trunk-0-qkv/qkv.bin. Its companion qkv-all-layer-0-manifest.json preserves the
duplicate's different provenance, alongside the original manifest.json.
The old duplicate folder and its legacy link no longer exist. Both retained
manifests and the canonical payload are unchanged. Other model aliases remain.

Both actual layer-0 datasets are now together on AX102:

```text
/opt/clover-k3/clover-intelegence/code/server/bin/dataset/
	trunk-0-qkv/
		qkv.bin
		manifest.json
	trunk-0/
		common.bin
		kda.bin
		dense.bin
		index.bin
		build-values.json
		manifest.json
```

The 265008216-byte payload was moved, not regenerated or copied. Both previous
operator paths resolve through compatibility links. No symlinks are inside the
new QKV directory. The local workspace contains manifests and move records,
not the large payloads.

The complete six-file trunk-0 directory was subsequently moved here unchanged.
Its previous main-dataset path and older clover-data/trunk-0 path are aliases,
not duplicate payloads. Full file hashes/identities and directory identity passed;
the server loads both datasets using internal relative paths. The workspace has
small manifest copies beside this note, not the numeric payloads. See the
[server documentation](../../README.md).

Both real dataset directories now sit inside bin next to the executable, matching
the client layout. The old server/dataset sibling is absent, and existing model
aliases point directly to these new locations. No runtime configuration file is
needed by the current server. Original numeric payloads and manifests are unchanged.