# Normalization Data

[leaves.json](leaves.json) holds three global output parameter tensors: aggregation
normalization, aggregation projection and final normalization gains. It contains
43,008 raw BF16 bytes as Base64 with shape and source metadata, not model outputs.

The real AX102 file is at
/opt/clover-k3/clover-intelegence/code/normalization/bin/dataset/leaves.json.
It moved unchanged; old main-dataset and clover-data paths remain compatibility
aliases. This small file also has an identical local copy for testing.

Normalization does not open the whole-model index or large checkpoint shard.
Those remain in the main dataset. See [the stage documentation](../../README.md).