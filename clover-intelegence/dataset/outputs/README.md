# Output Dataset Location

The complete output dataset is on AX102 at:

```text
/opt/clover-k3/clover-intelegence/code/client/bin/dataset/outputs
```

It contains the original fruit.bin, numeric dictionary, metadata and reader
scripts, moved unchanged. The large numeric payload is not downloaded into this
workspace. Old client output aliases were explicitly removed from the server;
use the bin path above. No new output-head implementation or generated model
output is included in this packaging change.