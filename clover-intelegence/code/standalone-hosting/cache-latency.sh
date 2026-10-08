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
def steps(cap):
    out=[]
    for line in open("/tmp/cache-%d.err"%cap,errors="ignore"):
        if line.startswith("STEP_JSON"):
            out.append(float(re.search(r'"seconds":([0-9.eE+-]+)',line).group(1)))
    return out
def requests(cap):
    out=[]
    for line in open("/tmp/cache-%d.out"%cap,errors="ignore"):
        if line.startswith("DONE_JSON"):
            out.append(float(re.search(r'"seconds":([0-9.eE+-]+)',line).group(1)))
    return out
print()
print("per-prompt mean decode-step latency, seconds")
hdr=["capacity"]+["p%d"%i for i in range(1,9)]+["mean"]
print("  %-9s"%hdr[0]+"".join("%8s"%h for h in hdr[1:]))
base=None; bm=0
for cap in caps:
    s=steps(cap)
    per=[sum(s[p*8+1:(p+1)*8])/7 for p in range(8)]
    m=sum(per)/8
    if base is None: base,bm=per[:],m
    print("  %-9d"%cap+"".join("%8.3f"%v for v in per)+"%8.3f"%m)
print()
print("end-to-end seconds for one prompt: 5-6 input tokens, 8 output tokens")
print("  %-9s"%"capacity"+"".join("%8s"%("p%d"%i) for i in range(1,9))+"%8s"%"mean")
rbase=None
for cap in caps:
    r=requests(cap)
    if len(r)!=8: print("  %-9d  (expected 8 requests, saw %d)"%(cap,len(r))); continue
    m=sum(r)/8
    if rbase is None: rbase=m
    print("  %-9d"%cap+"".join("%8.2f"%v for v in r)+"%8.2f"%m)
print()
print("speedup against no cache")
print("  %-9s"%"capacity"+"".join("%8s"%("p%d"%i) for i in range(1,9))+"%8s"%"mean")
for cap in caps[1:]:
    s=steps(cap)
    per=[sum(s[p*8+1:(p+1)*8])/7 for p in range(8)]
    sp=[base[i]/per[i] for i in range(8)]
    print("  %-9d"%cap+"".join("%8.2f"%v for v in sp)+"%8.2f"%(bm/(sum(per)/8)))
print()
for cap in caps:
    hits=miss=rb=0
    for line in open("/tmp/cache-%d.err"%cap,errors="ignore"):
        if line.startswith("STEP_JSON"):
            for k,setter in (("expert_cache_hits",0),("expert_cache_misses",1),("expert_read_bytes",2)):
                m=re.search(r'"%s":([0-9]+)'%k,line)
                if m:
                    if setter==0: hits=int(m.group(1))
                    elif setter==1: miss=int(m.group(1))
                    else: rb=int(m.group(1))
    tot=hits+miss
    print("  capacity %-3d  cache %d/%d = %.1f%%   total expert read %.1f GB"%(
        cap,hits,tot,100.0*hits/tot if tot else 0.0,rb/1e9))
print()
ref=open("/tmp/cache-0.ids").read()
ok=all(open("/tmp/cache-%d.ids"%c).read()==ref for c in caps)
print("tokens identical at every capacity: %s"%("YES" if ok else "NO"))
PY
