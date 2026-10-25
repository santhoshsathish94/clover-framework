#!/bin/bash
# Tag summary of the probe event files (mutants/logs/probe-<P>.events.txt, lines "<pid> <tag...>"), one block per probe; written to mutants/logs/probe-summary.txt
cd "$(dirname "$0")/logs"
{
for f in probe-P*.events.txt; do
  p=${f#probe-}; p=${p%.events.txt}
  n=$(wc -l < "$f"); np=$(awk '{print $1}' "$f" | sort -u | wc -l); bad=$(grep -vcE '^[0-9]+ [a-z_0-9]+( |$)' "$f")
  echo "== $p: $n events from $np distinct processes; malformed lines: $bad"
  case $p in
    P16|P26|P67|P74|P90|P92|P93|P97) cut -d' ' -f2 "$f" | sort | uniq -c | sort -rn | head -6 ;;
  esac
  case $p in
    P41) awk '{print $2, $4}' "$f" | sort | uniq -c | sort -rn | head -8; echo "   alt_nchunk values != 1: $(grep -vc 'alt_nchunk=1$' "$f")" ;;
    P42|P43) echo "   q8 events: $(grep -c ' q8 ' "$f"); chunk8 != chunk7: $(awk '{split($5,a,"=");split($6,b,"=");split($7,c,"="); if (a[2]!=b[2]) k++} END{print k+0}' "$f"); q>=9: $(awk '{split($3,q,"="); if (q[2]>=9) k++} END{print k+0}' "$f"); chunk8 != chunk9 with q>=9: $(awk '{split($3,q,"=");split($5,a,"=");split($7,c,"="); if (q[2]>=9 && a[2]!=c[2]) k++} END{print k+0}' "$f"); distinct raw: $(awk '{print $4}' "$f" | sort -u | tr '\n' ' ')" ;;
    P52) echo "   job events: $(grep -c ' off ' "$f"); narrow_equal=false: $(grep -c 'narrow_equal=false' "$f"); max e*raw: $(awk '{split($3,e,"=");split($4,r,"="); v=e[2]*r[2]; if (v>m) m=v} END{printf "%d", m}' "$f") (2^31 = 2147483648); distinct raw: $(awk '{print $4}' "$f" | sort -u | tr '\n' ' ')" ;;
    P57) echo "   pr_le0 events: $(grep -c ' pr_le0 ' "$f"); with jobs > 0: $(grep ' pr_le0 ' "$f" | grep -vc 'jobs=0$')" ;;
    P62) echo "   short reads: $(grep -c ' short ' "$f"); loop iterations with got > 0 and r > 0: $(grep -c iteration_with_got_gt0 "$f")" ;;
    P76) echo "   run_parallel calls: $(grep -c ' run_parallel ' "$f"); with n >= 2 (original spawns n-1 threads, mutant spawns n-2 or 0): $(awk '{split($3,a,"="); if (a[2]>=2) k++} END{print k+0}' "$f"); with n >= 3: $(awk '{split($3,a,"="); if (a[2]>=3) k++} END{print k+0}' "$f")" ;;
    P74) echo "   check returned with a non-zero status: $(grep -c check_returned_with_nonzero_status "$f")" ;;
    P93) echo "   inits with a non-zero counter before the reset: $(grep -c ' init ' "$f" | tr -d '\n') inits, non-zero: $(grep ' init ' "$f" | grep -vc 'prev_wbs=0$')" ;;
    P97) echo "   sy_ensure calls with n = 0: $(grep -c ' n=0 ' "$f"); with NULL experts: $(grep -c 'experts_null=true' "$f")" ;;
  esac
done
} > probe-summary.txt
cat probe-summary.txt
