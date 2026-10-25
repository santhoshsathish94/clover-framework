#!/bin/bash
# Coverage statistics of a finished run (what the seeds actually exercised), from the raw output files: ./stats.sh [outdir]
cd "$(dirname "$0")"
O=${1:-out-final}
echo "open+shm units: $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep -c '^C open cfg')"
for k in 'prefill=1 ' 'prefill=0 ' 'mode=0 ' 'mode=1 ' 'hold=1$' 'hold=0$' 'barrier=1 ' 'barrier=0 ' 'raw=4096 ' 'raw=8192 ' 'raw=65536 ' 'cap=0 ' 'cap=1 ' 'cap=7 ' 'cap=512 ' 'cap=4096 ' 'cap=5000 ' 'cap=8192 '; do
  echo "  cfg $k: $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep '^C open cfg' | grep -c " $k\|$k")"
done
echo "  layer lines: $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep -c '^C open layer')  layer 0: $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep -c '^C open layer 0 ')  layer 92: $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep -c '^C open layer 92 ')  layer 93: $(cat $O/open.*.rust.txt | grep -c '^C open layer 93 ')  layer 100000: $(cat $O/open.*.rust.txt | grep -c '^C open layer 100000 ')  layer 4294967295 (L=-1): $(cat $O/open.*.rust.txt | grep -c 'open layer 4294967295 ')  k_in=-1: $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep '^C open layer' | grep -c 'k_in=-1')"
echo "  prefill reads: layers with slab=eq: $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep -c 'slab=eq')  most preads in one layer: $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep -o 'preads=[0-9]*' | cut -d= -f2 | sort -n | tail -1)"
echo "  card lines with two or more layers: $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep '^C open card' | grep -c 'trunk.*trunk')"
echo "fatal units: $(ls $O/fatal.*.rust.txt | wc -l); fatal cases: $(cat $O/fatal.*.rust.txt | grep -c '^C fatal')  with exit=1: $(cat $O/fatal.*.rust.txt | grep '^C fatal' | grep -c 'exit=1 ')  with BUFFERED flushed: $(cat $O/fatal.*.rust.txt | grep '^C fatal' | grep -c 'out=\[BUFFERED\]')"
for k in missing_file_openmode0 missing_file_openmode1 missing_file_openmode2 long_path_total4094 long_path_total4095 long_path_total4096 long_path_total4097 eof_ readerr_ exit_while_other_card_runs; do
  echo "  $k: $(cat $O/fatal.*.rust.txt | grep -c "^C fatal $k")"
done
echo "  eof mode0: $(cat $O/fatal.*.rust.txt | grep -c '^C fatal eof_.*_mode0')  mode1: $(cat $O/fatal.*.rust.txt | grep -c '^C fatal eof_.*_mode1')   readerr errno EINTR(4): $(cat $O/fatal.*.rust.txt | grep -c '^C fatal readerr_.*errno4')  EIO(5): $(cat $O/fatal.*.rust.txt | grep -c '^C fatal readerr_.*errno5')  EAGAIN(11): $(cat $O/fatal.*.rust.txt | grep -c '^C fatal readerr_.*errno11')"
echo "wait waiters: 1=$(cat $O/wait.*.rust.txt | grep -c 'wait waiters 1 ok') 2=$(cat $O/wait.*.rust.txt | grep -c 'wait waiters 2 ok') 3=$(cat $O/wait.*.rust.txt | grep -c 'wait waiters 3 ok') 4=$(cat $O/wait.*.rust.txt | grep -c 'wait waiters 4 ok');  nested trees: $(cat $O/wait.*.rust.txt | grep -h 'wait nested nodes' | awk '{n+=$5; if ($5>m) m=$5} END{print NR" runs, nodes total "n", max "m}')"
echo "fifo: cards_used histogram: $(cat $O/fifo.*.rust.txt | grep '^C fifo cards_used' | awk '{h[$4]++} END{for (k in h) printf "%s:%d ", k, h[k]}')  jobs total: $(cat $O/fifo.*.rust.txt | grep '^C fifo cards_used' | awk '{n+=$6} END{print n}')  zero-job cards: $(cat $O/fifo.*.rust.txt | grep -c '^C fifo card [0-9] jobs 0 ')"
echo "barrier: parties histogram: $(cat $O/barrier.*.rust.txt | grep '^C barrier parties' | awk '{h[$4]++} END{for (k in h) printf "%s:%d ", k, h[k]}')"
echo "stress: $(cat $O/stress.*.rust.txt | grep -h '^N stress' | awk '{s+=$4; j+=$6; w+=$10} END{print NR" runs, submitter threads total "s", jobs "j", concurrent waits returned "w}')"
echo "real: $(cat $O/real.*.rust.txt | grep -c 'slab=eq') of $(ls $O/real.*.rust.txt | wc -l) slab=eq;  open_mode=0 (O_DIRECT, 17547264-byte reads): $(cat $O/real.*.rust.txt | grep -c 'open_mode=0')  open_mode=1 (plain): $(cat $O/real.*.rust.txt | grep -c 'open_mode=1')"
echo "natural: $(cat $O/natural.*.rust.txt | grep -c '^C natural layer') layers, first open refused by procfs with errno 22: $(cat $O/natural.*.rust.txt | grep -c 'errno 22')"
echo "init2: $(cat $O/init2.*.rust.txt | grep -c 'second init ignored') ok"
# driver v2 scenarios
if ls $O/book.*.rust.txt >/dev/null 2>&1; then
echo "book: $(ls $O/book.*.rust.txt | wc -l) units, $(cat $O/book.*.rust.txt | grep -c '^C book layer') handles written on read-only pages (layer 0: $(cat $O/book.*.rust.txt | grep -c '^C book layer 0 '))"
echo "open errno of the refused O_DIRECT open (one line per open / shm unit; only mode 1 units refuse, the others print 22 and refuse nothing): $(cat $O/open.*.rust.txt $O/shm.*.rust.txt | grep -h '^C open refuse errno' | sort | uniq -c | tr '\n' ' ')"
echo "hotwait: $(cat $O/hotwait.*.rust.txt | grep -h '^N hotwait' | awk '{u++; w+=$4; r+=$6; j+=$8; q+=$12; s+=$14; c+=$16; sig+=$18} END{print u" units, waiters "w", rounds "r", jobs "j", quiesce checks "q", stuck "s", units with chaos "c", chaos signals "sig}')"
echo "race2: $(cat $O/race2.*.rust.txt | grep -h '^N race2 children' | awk '{c+=$4; t+=$6; v+=$8} END{print NR" units, "c" fresh processes, "t" first-submit trials, rust violations "v}')  ref violations: $(cat $O/race2.*.ref.txt | grep -h '^N race2 children' | awk '{t+=$6; v+=$8} END{print v" of "t}')"
echo "big: raw histogram $(cat $O/big.*.rust.txt | grep -h '^C big layer' | awk '{h[$8]++} END{for (k in h) printf "%s:%d ", k, h[k]}')  slab bytes up to $(cat $O/big.*.rust.txt | grep -h '^C big layer' | awk '{b=$8*4; if (b>m) m=b} END{print m}')"
echo "fd0: $(cat $O/fd0.*.rust.txt | grep -c 'out=\[FD0 OK fd=0 ') of $(cat $O/fd0.*.rust.txt | grep -c '^C fd0') children got descriptor 0 from the library's open (prefill: $(cat $O/fd0.*.rust.txt | grep -c 'slab=eq'), without: $(cat $O/fd0.*.rust.txt | grep -c 'slab=null'))"
echo "stack: $(cat $O/stack.*.rust.txt | grep -c 'STACK OK') jobs of 3 / 6 MB ran on a loader thread, skipped units: $(cat $O/stack.*.rust.txt | grep -c '^C stack skipped')"
echo "guard: $(cat $O/guard.*.rust.txt | grep -c '^C guard') cases, exit=1: $(cat $O/guard.*.rust.txt | grep '^C guard' | grep -c 'exit=1 ')"
fi
