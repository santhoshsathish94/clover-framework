#!/bin/bash
# Instrument validation: sha256 implementation vs sha256sum, generator vs files on disk.
cd "$(dirname "$0")"
./diff_driver - selftest
./diff_driver - mkfiles
echo "--- sha256sum of record 0 and 3 sliced from the /dev/shm file (must equal the driver's rec 0 / rec 3)"
for i in 0 3; do dd if=/dev/shm/copyq_verify/rec8.bin bs=17547264 skip=$i count=1 2>/dev/null | sha256sum; done
echo "--- whole-file sha256sum, shm vs disk copy"
sha256sum /dev/shm/copyq_verify/rec8.bin data/rec8.bin
df -h /dev/shm / | cat
free -g | head -2
