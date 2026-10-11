/* The stage number is supposed to replace routing: owner and destination should both be
   arithmetic. This checks the whole range end to end rather than trusting the formulas. */
#include <stdio.h>
#include "stage.h"

int main(void)
{
    int failures = 0;

    /* Every stage belongs to exactly one service, and the services tile the range. */
    for (int stage = 1; stage <= CLOVER_STAGE_LAST; stage++) {
        int owner = clover_stage_owner(stage);
        if (owner < 0 || owner > CLOVER_STAGE_LAYERS + 1) {
            printf("FAIL stage %d has owner %d\n", stage, owner); failures++; break;
        }
        int local = clover_stage_local(stage);
        if (local < 1 || local > CLOVER_STAGE_STRIDE) {
            printf("FAIL stage %d has local %d\n", stage, local); failures++; break;
        }
        if (clover_stage_first(owner) + local - 1 != stage) {
            printf("FAIL stage %d does not rebuild from owner %d local %d\n", stage, owner, local);
            failures++; break;
        }
    }

    /* Walking next() from the first stage must visit every stage once, in order, and
       hand over exactly 93 times: server->layer1, 91 layer boundaries, layer92->tail. */
    int stage = clover_stage_first(0), visited = 0, handovers = 0;
    while (stage && visited <= CLOVER_STAGE_LAST) {
        visited++;
        if (clover_stage_hands_over(stage)) handovers++;
        stage = clover_stage_next(stage, 0);
    }
    if (visited != CLOVER_STAGE_LAST) { printf("FAIL walked %d of %d stages\n", visited, CLOVER_STAGE_LAST); failures++; }
    /* 93 forward boundaries plus the tail looping back to the server for the next token. */
    if (handovers != CLOVER_STAGE_LAYERS + 2) { printf("FAIL %d handovers, expected %d\n", handovers, CLOVER_STAGE_LAYERS + 2); failures++; }

    /* The tail forks: another token restarts at the server, otherwise the walk ends. */
    if (clover_stage_next(CLOVER_STAGE_LAST, 1) != clover_stage_first(0)) { printf("FAIL tail does not loop to server\n"); failures++; }
    if (clover_stage_next(CLOVER_STAGE_LAST, 0) != 0) { printf("FAIL tail does not finish\n"); failures++; }

    printf("server     %5d..%d\n", clover_stage_first(0), clover_stage_first(1) - 1);
    printf("layer 1    %5d..%d\n", clover_stage_first(1), clover_stage_first(2) - 1);
    printf("layer 46   %5d..%d\n", clover_stage_first(46), clover_stage_first(47) - 1);
    printf("layer 92   %5d..%d\n", clover_stage_first(92), clover_stage_first(93) - 1);
    printf("tail       %5d..%d\n", clover_stage_first(93), CLOVER_STAGE_LAST);
    printf("%s: %d stages, %d handovers, owner and next are arithmetic\n",
        failures ? "FAIL" : "PASS", visited, handovers);
    return failures ? 1 : 0;
}
