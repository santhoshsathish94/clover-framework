cd ~/rustmig/routetable
bash verify/work/mut/md5set.sh > verify/out/mut/md5-after.txt
if cmp -s verify/out/mut/md5-before.txt verify/out/mut/md5-after.txt; then echo MD5-BEFORE-AFTER-IDENTICAL; else echo MD5-DIFFER; diff verify/out/mut/md5-before.txt verify/out/mut/md5-after.txt; fi
echo "real so now: $(md5sum < target/release/libroutetable.so)"
cd verify/out/mut
rm -rf /tmp/c2rt-pack; mkdir -p /tmp/c2rt-pack/logs
cp summary.txt md5-before.txt md5-after.txt /tmp/c2rt-pack/logs/
for d in M*; do
  [ -d "$d" ] || continue
  mkdir -p /tmp/c2rt-pack/logs/$d
  cp $d/stdout.txt /tmp/c2rt-pack/logs/$d/ 2>/dev/null
  cp $d/build.log /tmp/c2rt-pack/logs/$d/ 2>/dev/null
  ff=$(awk '/^FIRST FAILURE/{print $3}' $d/stdout.txt | tr -d '():' | head -1)
  [ -n "$ff" ] && [ -f "$ff" ] && cp "$ff" /tmp/c2rt-pack/logs/$d/first-failure-report.txt
done
cp -r ../../work/mut/diffs /tmp/c2rt-pack/diffs
cp ../../work/mut/probes-summary.txt ../../work/mut/collect.txt /tmp/c2rt-pack/
cp -r ../../work/mut/probe-in ../../work/mut/probe-out /tmp/c2rt-pack/
tar -C /tmp -czf /tmp/c2rt-pack.tgz c2rt-pack
ls -la /tmp/c2rt-pack.tgz
du -sh /tmp/c2rt-pack
