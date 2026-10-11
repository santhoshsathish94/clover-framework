#define _GNU_SOURCE
/* How much of a mapped file is actually in memory right now.
   mmap does not load anything: pages arrive on first touch and live in the page
   cache, which the kernel may reclaim at any time unless they are locked. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

int main(int argc, char **argv)
{
    if (argc < 2) { fputs("usage: residency FILE...\n", stderr); return 2; }
    long page = sysconf(_SC_PAGESIZE);

    for (int index = 1; index < argc; index++) {
        int handle = open(argv[index], O_RDONLY);
        struct stat info;
        if (handle < 0 || fstat(handle, &info)) { fprintf(stderr, "cannot open %s\n", argv[index]); if (handle>=0) close(handle); continue; }
        size_t bytes = (size_t)info.st_size;
        size_t pages = (bytes + (size_t)page - 1) / (size_t)page;
        void *mapped = mmap(NULL, bytes, PROT_READ, MAP_SHARED, handle, 0);
        if (mapped == MAP_FAILED) { fprintf(stderr, "cannot map %s\n", argv[index]); close(handle); continue; }

        unsigned char *state = malloc(pages);
        size_t resident = 0;
        if (state && !mincore(mapped, bytes, state))
            for (size_t p = 0; p < pages; p++) resident += state[p] & 1;
        else { fprintf(stderr, "mincore failed for %s\n", argv[index]); }

        printf("%-62s %7.2f GB total, %7.2f GB resident, %5.1f%%\n",
            argv[index], (double)bytes / 1e9,
            (double)resident * (double)page / 1e9,
            pages ? 100.0 * (double)resident / (double)pages : 0.0);

        free(state);
        munmap(mapped, bytes);
        close(handle);
    }
    return 0;
}
