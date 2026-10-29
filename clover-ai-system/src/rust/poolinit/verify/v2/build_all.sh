#!/bin/bash
# Driver v2, step 1: (re)builds the mutant crates of mutants/ WITHOUT touching mutants/ (no mk_mutants.py: it would rewrite the diffs and md5list.txt of the independent verifier).
# The scratch crates work/mut/<name>/ (sources of the mutants, kept by the verifier) are built with `cargo build --release`, the .so is copied to work/v2so/<name>.so, target/ is removed
# (disk), and the md5 of the .so is compared with mutants/md5list.txt (same toolchain => same md5; a different one is reported as DIFF, not hidden).
# usage (box, ~/rustmig/poolinit/verify): v2/build_all.sh [name ...]       (no names = every name in mutants/md5list.txt)   env J (parallel builds, default 6)
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
mkdir -p work/v2so out/v2
names="$*"
[ -z "$names" ] && names=$(awk '{print $1}' mutants/md5list.txt)
build_one() {
  n=$1
  d=work/mut/$n
  [ -d "$d" ] || { echo "$n NO SCRATCH CRATE"; return; }
  ( cd "$d" && nice -n 10 cargo build --release > /tmp/v2build_$n.log 2>&1 ) || { echo "$n BUILD FAILED"; return; }
  cp "$d/target/release/libpoolinit.so" work/v2so/$n.so
  rm -rf "$d/target" /tmp/v2build_$n.log
  new=$(md5sum work/v2so/$n.so | cut -c1-32)
  old=$(awk -v n="$n" '$1==n {print $2}' mutants/md5list.txt)
  if [ "$new" = "$old" ]; then echo "$n $new same"; else echo "$n $new DIFF old=$old"; fi
}
export -f build_one
echo $names | tr ' ' '\n' | xargs -P ${J:-6} -I{} bash -c 'build_one {}' | sort -V > out/v2/build.txt
echo "built: $(wc -l < out/v2/build.txt) same=$(grep -c ' same$' out/v2/build.txt) diff=$(grep -c ' DIFF ' out/v2/build.txt) failed=$(grep -c FAILED out/v2/build.txt); $(du -xsh work/v2so work/mut | tr '\n' ' ')"
