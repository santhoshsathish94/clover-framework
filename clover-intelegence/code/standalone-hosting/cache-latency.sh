#!/bin/sh
# Does the per-layer expert cache help every prompt, and by how much?
#
# Eight different prompts, eight output tokens each, one request at a time so
# this measures single-stream latency rather than throughput. Each prompt is
# reported on its own: an average would hide a cache that only helps a few.
#
# Tokens must be identical at every capacity. A cache that changes an answer is
# a bug, not an optimisation.
S=/opt/clover-k3/clover-intelegence/code/standalone-hosting
cd "$S" || exit 1

for ids in 1008,10484,318,15383,387 1008,10484,318,10417,387 \
           1008,10484,318,19509,387 1008,10604,19645,387 \
           17285,9620,2069,28542 1008,12955,318,276,18985,387 \
           68430,387 1008,17296,318,6903,1900,67244; do
  printf '{"input_ids":[%s],"max_new_tokens":8}\n' "$ids"
done > /tmp/eight8.jsonl

run() {
  cap=$1
  start=$(date +%s.%N)
  env -i PATH=/usr/bin:/bin CLOVER_DATASET="$S/bin/dataset" \
      CLOVER_CONFIG="$S/bin/configs/generation.json" \
      OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
      K3_PAYLOAD_RAM=1 K3_EXPERT_DIRECT=1 K3_EXPERT_CACHE="$cap" \
      "$S/bin/clover-one-cache" < /tmp/eight8.jsonl \
      > "/tmp/cache-$cap.out" 2>"/tmp/cache-$cap.err"
  end=$(date +%s.%N)
  echo "wall $(echo "$end - $start" | bc)" > "/tmp/cache-$cap.wall"
  grep -oE '"request":[0-9]+,"index":[0-9]+,"token":[0-9]+' "/tmp/cache-$cap.out" \
    | tr -d '"' | tr ',' ' ' \
    | awk '{split($1,a,":");split($2,b,":");split($3,c,":");print a[2],b[2],c[2]}' \
    | sort -k1,1n -k2,2n > "/tmp/cache-$cap.ids"
}

for cap in 0 16 24; do
  echo "=== capacity $cap experts/layer"
  run $cap
  cat "/tmp/cache-$cap.wall"
  free -g | sed -n 2p
done

python3 - <<'PY'
import json,re,os
caps=[0,16,24]
print()
print("per-prompt mean decode-step latency, seconds")
print("  %-10s %7s %7s %7s %7s %7s %7s %7s %7s   %8s" % tuple(
      ["capacity"]+["p%d"%i for i in range(1,9)]+["mean"]))
base=None
for cap in caps:
    secs=[]; 
    for line in open("/tmp/cache-%d.err"%cap,errors="ignore"):
        if line.startswith("STEP_JSON"):
            secs.append(float(re.search(r'"seconds":([0-9.eE+-]+)',line).group(1)))
    # 8 prompts x (1 prefill + 7 decode)
    per=[]
    for p in range(8):
        g=secs[p*8:(p+1)*8]
        per.append(sum(g[1:])/len(g[1:]) if len(g)==8 else float("nan"))
    m=sum(per)/len(per)
    if base is None: base=per[:]; bm=m
    print("  %-10d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f   %8.3f"%tuple([cap]+per+[m]))
print()
print("  %-10s %7s %7s %7s %7s %7s %7s %7s %7s   %8s" % tuple(
      ["speedup"]+["p%d"%i for i in range(1,9)]+["mean"]))
for cap in caps[1:]:
    secs=[]
    for line in open("/tmp/cache-%d.err"%cap,errors="ignore"):
        if line.startswith("STEP_JSON"):
            secs.append(float(re.search(r'"seconds":([0-9.eE+-]+)',line).group(1)))
    per=[sum(secs[p*8+1:(p+1)*8])/7 for p in range(8)]
    sp=[base[i]/per[i] for i in range(8)]
    print("  %-10d %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f %7.2f   %8.2f"%tuple(
        [cap]+sp+[bm/(sum(per)/8)]))
print()
for cap in caps:
    hits=miss=0; rb=0
    for line in open("/tmp/cache-%d.err"%cap,errors="ignore"):
        if line.startswith("STEP_JSON"):
            h=re.search(r'"expert_cache_hits":([0-9]+)',line)
            m=re.search(r'"expert_cache_misses":([0-9]+)',line)
            b=re.search(r'"expert_read_bytes":([0-9]+)',line)
            if h: hits=int(h.group(1))
            if m: miss=int(m.group(1))
            if b: rb=int(b.group(1))
    tot=hits+miss
    print("  capacity %-3d  cache %d/%d = %.1f%%   total expert read %.1f GB"%(
        cap,hits,tot,100.0*hits/tot if tot else 0.0,rb/1e9))
print()
ok=all(open("/tmp/cache-%d.ids"%c).read()==open("/tmp/cache-0.ids").read() for c in caps)
print("tokens identical at every capacity: %s"%("YES" if ok else "NO"))
PY
