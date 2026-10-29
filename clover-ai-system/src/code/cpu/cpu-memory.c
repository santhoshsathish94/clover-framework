#define _GNU_SOURCE
#include <string.h>
#include <sys/mman.h>
#include "backend.h"

/* Anonymous memory on huge pages, touched now so no page is first used during a request. */
unsigned char *backend_expert_memory(size_t bytes)
{
    unsigned char *memory = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) return NULL;
    madvise(memory, bytes, MADV_HUGEPAGE);
    memset(memory, 0, bytes);
    return memory;
}
