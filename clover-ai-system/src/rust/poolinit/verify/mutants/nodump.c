/* LD_PRELOAD helper for the mutant runs: makes every process non-dumpable, so a mutant that aborts / segfaults does not start the distribution's crash handler (apport, ~8 s per crash,
   serialised) for each of its hundreds of crashing units. The signal is still delivered; the unit still sees "sig6" / "sig11". Nothing else changes (no driver or crate change). */
#include <sys/prctl.h>
__attribute__((constructor)) static void nodump(void) { prctl(PR_SET_DUMPABLE, 0, 0, 0, 0); }
