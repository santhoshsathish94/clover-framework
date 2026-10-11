/* The frame is meant to carry its own route and survive a round trip to storage.
   This walks one frame through every stage and checks it reloads identically. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "frame.h"

int main(void)
{
    int failures = 0;
    printf("CloverFrame = %zu bytes (%.1f KB)\n", sizeof(CloverFrame), sizeof(CloverFrame) / 1024.0);

    CloverFrame frame;
    clover_frame_init(&frame, 7);
    for (unsigned i = 0; i < CLOVER_FRAME_WIDTH; i++) frame.residual[i] = (float)i * 0.5f;
    frame.snapshot_count = 4;

    if (!clover_frame_is_mine(&frame, 0)) { printf("FAIL fresh frame is not the server's\n"); failures++; }
    if (clover_frame_is_mine(&frame, 46)) { printf("FAIL fresh frame claimed by layer 46\n"); failures++; }

    /* Walk every stage and count how many each service is handed. */
    int seen[CLOVER_STAGE_LAYERS + 2] = {0};
    int steps = 0;
    while (frame.stage && steps <= CLOVER_STAGE_LAST) {
        int owner = clover_frame_owner(&frame);
        if (owner < 0) { printf("FAIL stage %d has no owner\n", frame.stage); failures++; break; }
        seen[owner]++;
        steps++;
        clover_frame_advance(&frame, 0);
    }
    if (steps != CLOVER_STAGE_LAST) { printf("FAIL walked %d stages\n", steps); failures++; }
    for (int owner = 0; owner <= CLOVER_STAGE_LAYERS + 1; owner++)
        if (seen[owner] != CLOVER_STAGE_STRIDE) {
            printf("FAIL owner %d saw %d stages, expected %d\n", owner, seen[owner], CLOVER_STAGE_STRIDE);
            failures++; break;
        }

    /* Round trip through a file: the frame must come back byte for byte. */
    clover_frame_init(&frame, 7);
    frame.stage = clover_stage_first(46) + 29;
    frame.position = 3;
    frame.snapshot_count = 4;
    for (unsigned i = 0; i < CLOVER_FRAME_WIDTH; i++) frame.residual[i] = (float)i * -0.25f;

    FILE *file = fopen("/tmp/frame.bin", "wb");
    if (!file || !clover_frame_write(file, &frame)) { printf("FAIL write\n"); failures++; }
    if (file) fclose(file);

    CloverFrame loaded;
    file = fopen("/tmp/frame.bin", "rb");
    if (!file || !clover_frame_read(file, &loaded)) { printf("FAIL read\n"); failures++; }
    if (file) fclose(file);

    if (memcmp(&frame, &loaded, sizeof frame)) { printf("FAIL round trip differs\n"); failures++; }
    if (clover_frame_owner(&loaded) != 46) { printf("FAIL reloaded owner %d, expected 46\n", clover_frame_owner(&loaded)); failures++; }
    if (clover_stage_local(loaded.stage) != 30) { printf("FAIL reloaded local %d, expected 30\n", clover_stage_local(loaded.stage)); failures++; }

    /* A frame addressed elsewhere must be rejected by one integer compare. */
    if (clover_frame_is_mine(&loaded, 45)) { printf("FAIL layer 45 claimed layer 46's frame\n"); failures++; }
    if (!clover_frame_is_mine(&loaded, 46)) { printf("FAIL layer 46 refused its own frame\n"); failures++; }

    printf("%s: %d stages walked, each service saw %d, round trip exact\n",
        failures ? "FAIL" : "PASS", steps, CLOVER_STAGE_STRIDE);
    return failures ? 1 : 0;
}
