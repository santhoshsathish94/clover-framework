#!/bin/bash
# Driver v2: rebuilds the previously uncaught real-gap mutants with the verifier's own generator (mutants/mk_mutants.py) and proves that doing so changed nothing in mutants/:
# the diffs and md5list.txt are copied aside first, compared (cmp) afterwards and restored if they differ. The scratch crates are rebuilt from the CURRENT crate sources (the md5 of the .so is compared
# with the recorded one by the generator's output line), then target/ is removed again.
# usage (box, ~/rustmig/poolinit/verify): bash v2/regen_mutants.sh [name ...]       default: M03 M11 M12 M13 M14 M33 M46 M57 M58
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
names="$*"; [ -z "$names" ] && names="M03 M11 M12 M13 M14 M33 M46 M57 M58"
B=$(mktemp -d /tmp/v2regen.XXXXXX)
cp -p mutants/md5list.txt $B/
for n in $names; do cp -p mutants/$n.diff $B/; done
python3 mutants/mk_mutants.py $names | cut -c1-110 > out/v2/regen.txt
cat out/v2/regen.txt
for n in $names; do cmp -s mutants/$n.diff $B/$n.diff && echo "$n.diff unchanged" || { echo "$n.diff DIFFERS: restored"; cp -p $B/$n.diff mutants/$n.diff; }; done
cmp -s mutants/md5list.txt $B/md5list.txt && echo "md5list.txt unchanged" || { echo "md5list.txt DIFFERS: restored"; cp -p $B/md5list.txt mutants/md5list.txt; }
for n in $names; do rm -rf work/mut/$n/target; done
rm -rf $B
