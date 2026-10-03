#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "numeric-table.h"
#include <assert.h>

/* Writes every table block already decoded and hash-checked, so the runtime maps values directly. */
int main(int argc,char **argv)
{
    assert(argc==3);
    const char *path=argv[1], *source=argv[2];
    char target[4096],staging[4096];
    assert(snprintf(target,sizeof target,"%s.direct",path)>0);
    assert(snprintf(staging,sizeof staging,"%s.direct.building",path)>0);

    NumericTable *table=table_open(path,source);
    assert(table);
    assert(!table->direct);

    const size_t bytes=(size_t)CLIENT_BLOCKS*CLIENT_RAW_BYTES;
    int handle=open(staging,O_WRONLY|O_CREAT|O_EXCL,0644);
    assert(handle>=0);
    assert(!ftruncate(handle,(off_t)bytes));

    double started=omp_get_wtime();
    for (unsigned block=0;block<CLIENT_BLOCKS;block++) {
        assert(table_load(table,block));
        ssize_t count=pwrite(handle,table->raw,CLIENT_RAW_BYTES,(off_t)block*CLIENT_RAW_BYTES);
        assert(count>=0 && (size_t)count==CLIENT_RAW_BYTES);
    }
    assert(!fsync(handle) && !close(handle));
    assert(!rename(staging,target));
    printf("CONVERTED %s blocks=%u file_bytes=%zu seconds=%.3f\n",
        target,CLIENT_BLOCKS,bytes,omp_get_wtime()-started);
    table_close(table);
    return 0;
}
