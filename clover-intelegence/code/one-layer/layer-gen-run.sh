#!/bin/bash
# Batched, verified run of ONE layer 2..92 (the run-time-layer build of layer-batch.c), KDA or MLA.
# usage: layer-gen-run2.sh LAYER K NTOKENS WINDOW[,WINDOW...] RESID_F32 SNAPS_F32
#   KIND is chosen from the layer number: MLA at 3,7,11,...,91 and 92, KDA otherwise (override: env LB_KIND=kda|mla).
#   K = number of input snapshots of that layer (see current-status.md).
#   env LB_SAVE_REF=file     pod output of this layer (becomes the next layer's RESID_F32)
#   env LB_SAVE_SNAPS=file   snapshots this layer hands on (becomes the next layer's SNAPS_F32)
#   env LB_POD_ONLY=1        run the pod only (to produce the chain of reference inputs)
# The template is the deployed transformer-2.c (KDA) or transformer-3.c (MLA) with the layer number, the snapshot count and
# the three layer-numbered path strings turned into run-time values. Nothing else in it changes; the sed is the same
# normalisation that showed all layers of one kind and one snapshot count are the same program.
set -u
LAYER=$1; K=$2; N=$3; W=$4; RESID=$5; SNAPS=$6
if [ -z "${LB_KIND:-}" ]; then
  if [ $((LAYER % 4)) -eq 3 ] || [ "$LAYER" -eq 92 ]; then LB_KIND=mla; else LB_KIND=kda; fi
fi
if [ "$LB_KIND" = mla ]; then T=3; else T=2; fi
H=/tmp/one-layer
D=/opt/clover-k3/clover-intelegence/code/distrubuted-hosting
DATA=/opt/clover-k3/clover-intelegence/dataset
cd $H
ulimit -l unlimited; ulimit -c 0
SRC=$D/tansformers/transformer-$T/transformer-$T.c
sed -E "s/^#define TRANSFORMER_LAYER $T\$/#define TRANSFORMER_LAYER lb_layer_no/;
        s/^#define TRANSFORMER_INPUT_SNAPSHOTS 1\$/#define TRANSFORMER_INPUT_SNAPSHOTS lb_k_in/;
        s/\(TRANSFORMER_LAYER - 1\)/($T - 1)/;
        s/\"operators\/qkv-all\/layer-$T\/operator.bin\"/lb_name(0)/;
        s/\"trunk-$T\"/lb_name(1)/;
        s/\"root-$T\"/lb_name(2)/;
        s/root_open\(root_directory, $T\)/root_open(root_directory, lb_layer_no)/" $SRC > tpl-$LB_KIND.c
echo "kind $LB_KIND (template of layer $T); template lines patched or using the run-time values: $(grep -c 'lb_name\|lb_layer_no\|lb_k_in' tpl-$LB_KIND.c)"
BIN=layer-batch-gen-$LB_KIND
if [ ! -x $BIN ] || [ layer-batch.c -nt $BIN ] || [ tpl-$LB_KIND.c -nt $BIN ]; then
  gcc -std=gnu11 ${LB_OPT:--O3} -march=native -ffp-contract=off -fno-fast-math -fopenmp -Wall -Wno-unused-function \
      -DLB_GENERIC -DLB_TEMPLATE="\"tpl-$LB_KIND.c\"" -I$D/common -I. layer-batch.c -lm -o $BIN 2> build-gen-$LB_KIND.log \
      || { grep -E "error" build-gen-$LB_KIND.log | head -30; exit 1; }
  echo "build warnings: $(grep -c warning build-gen-$LB_KIND.log)"
fi
env -i PATH="$PATH" LB_WIDE="${LB_WIDE:-1}" LB_VEC="${LB_VEC:-1}" LB_ROUTE="${LB_ROUTE:-1}" LB_EXPK="${LB_EXPK:-1}" LB_PRED=0 LB_LIN=0 LB_DUMP="" \
    LB_THP="${LB_THP:-1}" LB_SAVE_REF="${LB_SAVE_REF:-}" LB_SAVE_SNAPS="${LB_SAVE_SNAPS:-}" LB_POD_ONLY="${LB_POD_ONLY:-}" \
    LB_IOLO="${LB_IOLO:-}" LB_IOHI="${LB_IOHI:-}" LB_DISK="${LB_DISK:-3}" LB_IO="${LB_IO:-4}" LB_SPEC="${LB_SPEC:-0}" LB_REPEAT="${LB_REPEAT:-}" LB_A8T="${LB_A8T:-}" LB_A8T_GS="${LB_A8T_GS:-}" \
    OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./$BIN $DATA $RESID $SNAPS $N $W $LAYER $K 2>&1 | tee gen-$LAYER-$N-$W.out
