#!/bin/bash
# Final cleanup + integrity check (verifier). Keeps: mutants/ (scripts, diffs, md5 lists, units-logs.tar.gz, probe copy of the driver), deletes work/ and out/ (all created by this exercise).
cd "$(dirname "$0")/.."
echo "== crate and driver still the recorded ones?"
md5sum ../src/*.rs ../include/loader.h ../Cargo.toml ../Cargo.lock ld_driver ld_driver.cpp ../ref/libloader_ref.so ../target/release/libloader.so run.sh unit.sh final.sh build.sh summarize.sh stats.sh
echo "== archive of the unit logs"
(cd out/mut && ls -d */ | wc -l)
tar -czf mutants/units-logs.tar.gz -C out/mut --wildcards '*/units.log' '*/summary.txt' progress.txt
ls -la mutants/units-logs.tar.gz
tar -tzf mutants/units-logs.tar.gz | wc -l
echo "== sizes before"
du -xsh out work 2>/dev/null
rm -rf work out mutants/lw_probe mutants/probe/ld_driver_bk
rm -f /dev/shm/ldprobe-*.txt
echo "== leftovers"
ls -d /tmp/ldloader-verify.* /dev/shm/ldloader-verify.* /dev/shm/ldprobe-* /tmp/ldmut* /tmp/ldsmoke* 2>/dev/null | wc -l
ls verify 2>/dev/null; ls
du -xsh mutants
df -h / | tail -1
