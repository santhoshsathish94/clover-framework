# Transformer 1 Datasets

The complete runtime datasets are on AX102 at:

```text
/opt/clover-k3/clover-intelegence/code/tansformers/transformer-1/bin/dataset/
  trunk-1/
  root-1/
  operators/qkv-all/layer-1/
```

All forty existing files were moved unchanged, including historical observation
and validation records. The executable reads prepared trunk/operator parameters
and root constants/placement blocks; it never reads observations for inference.
Old main-dataset and clover-data paths remain compatibility aliases. The package
contains real dataset directories, not aliases to external payloads.

The workspace mirrors the directory layout with location notes and manifest
copies only. Large numeric payloads remain on AX102. See the
[program documentation](../../README.md) and its separate move plan/journal.