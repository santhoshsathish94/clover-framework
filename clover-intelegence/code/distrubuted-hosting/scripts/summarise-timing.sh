#!/bin/bash
# Aggregate the measured rows. Nothing here invents a number: every figure is a sum
# or a ratio of what /tmp/m.err recorded during the run.
set -uo pipefail
CODE=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting

grep '^TIMING' /tmp/m.err | sed 's/^TIMING [0-9]* //' | grep -v '^owner,' > /tmp/rows.csv

# Which layers are MLA, read from the sources rather than inferred from the timings.
: > /tmp/types.csv
for n in $(seq 1 92); do
    f="$CODE/tansformers/transformer-$n/transformer-$n.c"
    m=$(grep -oP '#define TRANSFORMER_MLA\s+\K[01]' "$f" 2>/dev/null | head -1)
    echo "$n,${m:-none}" >> /tmp/types.csv
done

awk -F, '
    NR==FNR { type[$1]=$2; next }
    {
        name=$2; calls=$3; secs=$4;
        total_all += secs
        if ($1=="-" && name=="request-total") { total=secs; next }
        if ($1=="-" && name=="queue")  { queue=secs; next }
        if (name=="server")            { server=secs; servercalls=calls; next }
        if (name=="normalize")         { norm=secs; next }
        if (name=="head")              { head=secs; headcalls=calls; next }
        if (name=="deliver")           { deliver=secs; next }
        layer_total += secs; layers++
        n=$1
        if (type[n]=="1") { mla_total+=secs; mla++ } else { kda_total+=secs; kda++ }
        if (slowest=="" || secs>slowsec) { slowest=n; slowsec=secs }
        if (fastest=="" || secs<fastsec) { fastest=n; fastsec=secs }
    }
    END {
        printf "measured request total      %10.3f s\n", total
        printf "\n"
        printf "queue wait                  %10.6f s  %6.3f%%\n", queue, 100*queue/total
        printf "server (trunk 0)            %10.3f s  %6.3f%%   %d calls\n", server, 100*server/total, servercalls
        printf "layers 1..92 together       %10.3f s  %6.3f%%   %d layers\n", layer_total, 100*layer_total/total, layers
        printf "layer 93 normalize          %10.6f s  %6.3f%%\n", norm, 100*norm/total
        printf "layer 93 head               %10.3f s  %6.3f%%   %d calls\n", head, 100*head/total, headcalls
        printf "layer 93 deliver            %10.6f s  %6.3f%%\n", deliver, 100*deliver/total
        printf "accounted                   %10.3f s  %6.3f%%\n", queue+server+layer_total+norm+head+deliver, 100*(queue+server+layer_total+norm+head+deliver)/total
        printf "unaccounted                 %10.3f s  %6.3f%%\n", total-(queue+server+layer_total+norm+head+deliver), 100*(total-(queue+server+layer_total+norm+head+deliver))/total
        printf "\n"
        printf "KDA layers  %2d   total %8.3f s   mean %6.4f s\n", kda, kda_total, kda_total/kda
        printf "MLA layers  %2d   total %8.3f s   mean %6.4f s\n", mla, mla_total, mla_total/mla
        printf "slowest layer %s at %.4f s, fastest layer %s at %.4f s\n", slowest, slowsec, fastest, fastsec
    }
' /tmp/types.csv /tmp/rows.csv

echo
echo "per-layer, measured, with type from source"
echo "layer,type,calls,seconds,ms_per_position,pct_of_request"
awk -F, '
    NR==FNR { type[$1]=$2; next }
    $1=="-" && $2=="request-total" { total=$4 }
    { rows[NR]=$0 }
    END {
        for (i=1;i<=NR;i++) {
            if (rows[i]=="") continue
            split(rows[i],f,",")
            if (f[1]=="-") continue
            t = (f[1]=="93") ? "tail" : (type[f[1]]=="1" ? "MLA" : (type[f[1]]=="none" ? "KDA(gen)" : "KDA"))
            if (f[1]=="0") t="server"
            printf "%s,%s,%s,%.4f,%.2f,%.3f\n", f[1], t, f[3], f[4], 1000*f[4]/f[3], 100*f[4]/total
        }
    }
' /tmp/types.csv <(cat /tmp/rows.csv; grep '^TIMING' /tmp/m.err | sed 's/^TIMING [0-9]* //' | grep 'request-total')
