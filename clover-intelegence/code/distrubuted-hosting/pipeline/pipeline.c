#define _GNU_SOURCE
#include "../server/ends.h"
#include "../common/frame.h"
#include <dlfcn.h>
#include <poll.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

enum { PIPELINE_WIDTH = 7168, PIPELINE_SNAPSHOTS = 8, PIPELINE_LAYERS = 92,
       PIPELINE_MAX_TOKENS = 256, PIPELINE_CALLBACK = 1024, PIPELINE_QUEUE = 4096,
       PIPELINE_PER_CLIENT = 8 };

/* The frame is the token. It carries its own stage, so the walk below reads one int
   to know which service owns it and never consults a map of layer to successor. */
typedef CloverFrame PipelineToken;

typedef struct {
    uint32_t ids[128];
    size_t count;
    unsigned wanted;
    char callback[PIPELINE_CALLBACK];
} PipelineRequest;

/* There is no slot count. A request allocates its own conversation state when it
   arrives and frees it when it finishes, so the only limit is memory, measured. */
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t released, drained;
    size_t budget, used, per_request;
    unsigned inflight;
} PipelineAdmission;

/* Submitting is not the same as running. A caller hands work over and is answered at
   once; the pool drains the queue at whatever rate memory allows. The queue depth is
   what absorbs a burst, so nobody is held on a connection waiting for a turn.

   A full queue is refused, never waited on: holding the caller would undo the point.
   There is no reply to collect later, so the only identity a submission has is the
   address it asked to be answered at. That is what a duplicate and a flood are
   counted against. */
typedef struct PipelineWork PipelineWork;

typedef struct {
    uint64_t request;          /* the whole submission: callback, length, ids */
    uint64_t client;           /* the callback address on its own */
    int live;
} PipelineTicket;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t arrived, idle;
    PipelineWork *entries[PIPELINE_QUEUE];
    PipelineTicket tickets[PIPELINE_QUEUE];
    size_t head, tail, pending;
    unsigned active, workers;
    int stopping;
} PipelineQueue;

/* One sequence per stage per in-flight request: the KDA recurrence and MLA cache belong
   to a single token history. The mapped weights are shared, so this is all a request costs. */
typedef struct {
    void *library, *model;
    void *(*create_sequence)(const void *);
    void (*close_model)(void *);
    void (*close_sequence)(void *);
    void (*reset_sequence)(void *);
} PipelineStage;

typedef struct {
    PipelineStage server, tail, layer[PIPELINE_LAYERS + 1];
    int (*server_process)(const void *, void *, const float *, float *, float *);
    int (*layer_93_process)(const void *, void *, const float *, const float *, unsigned,
        int, const char *, uint32_t *, TokenText *);
    void (*layer_93_set_request)(void *, unsigned);
    int (*layer_93_delivered)(const void *);
    int (*layer_93_timing)(const void *, double *, double *, double *);
    int (*layer_one)(void *, void *, const float *, const float *, float *);
    int (*layer_many[PIPELINE_LAYERS + 1])(void *, void *, const float *, const float *, unsigned, float *, float *);
} PipelineFleet;

typedef struct PipelineWork {
    PipelineFleet *fleet;
    PipelineAdmission *admission;
    Tokenizer *tokenizer;
    const char *root;
    PipelineRequest request;
    unsigned index;
    size_t ticket;
    double submitted;
} PipelineWork;

/* The conversation state for one request: one sequence per stage, nothing shared. */
typedef struct {
    void *server;
    void *layer[PIPELINE_LAYERS + 1];
    void *layer_93;
} PipelineSession;

/* Wall time per owner, so the cost of a stage is read off a run rather than argued. */
typedef struct {
    double waited, server, tail, deliver;
    double layer[PIPELINE_LAYERS + 1];
    unsigned calls[PIPELINE_LAYERS + 1];
    unsigned server_calls, tail_calls;
} PipelineTiming;

static double pipeline_clock(void)
{
    struct timespec time; clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec / 1e9;
}

static int pipeline_path(char *path, size_t capacity, const char *root, const char *relative)
{
    int length = snprintf(path, capacity, "%s/%s", root, relative);
    return length > 0 && (size_t)length < capacity;
}

static int pipeline_symbol(void *library, const char *name, void *destination, size_t bytes)
{
    void *symbol = dlsym(library, name);
    if (!symbol || bytes != sizeof symbol) { fprintf(stderr, "missing stage symbol %s\n", name); return 0; }
    memcpy(destination, &symbol, bytes);
    return 1;
}

static void pipeline_stage_close(PipelineStage *stage)
{
    if (stage->model && stage->close_model) stage->close_model(stage->model);
    if (stage->library) dlclose(stage->library);
    memset(stage, 0, sizeof *stage);
}

static int pipeline_stage_library(PipelineStage *stage, const char *directory, const char *prefix)
{
    char path[4096], name[128];
    memset(stage, 0, sizeof *stage);
    if (!pipeline_path(path, sizeof path, directory, "bin/pipeline-stage.so")) return 0;
    stage->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!stage->library) { fprintf(stderr, "stage load: %s\n", dlerror()); return 0; }
    snprintf(name, sizeof name, "%s_close", prefix);
    return pipeline_symbol(stage->library, name, &stage->close_model, sizeof stage->close_model);
}

static int pipeline_sequences(PipelineStage *stage, const char *prefix)
{
    char name[128];
    snprintf(name, sizeof name, "%s_sequence_create", prefix);
    if (!pipeline_symbol(stage->library, name, &stage->create_sequence, sizeof stage->create_sequence)) return 0;
    snprintf(name, sizeof name, "%s_sequence_close", prefix);
    if (!pipeline_symbol(stage->library, name, &stage->close_sequence, sizeof stage->close_sequence)) return 0;
    snprintf(name, sizeof name, "%s_sequence_reset", prefix);
    return pipeline_symbol(stage->library, name, &stage->reset_sequence, sizeof stage->reset_sequence);
}

static void pipeline_session_close(PipelineFleet *fleet, PipelineSession *session)
{
    for (unsigned layer = 1; layer <= PIPELINE_LAYERS; layer++)
        if (session->layer[layer]) fleet->layer[layer].close_sequence(session->layer[layer]);
    if (session->layer_93) fleet->tail.close_sequence(session->layer_93);
    if (session->server) fleet->server.close_sequence(session->server);
    memset(session, 0, sizeof *session);
}

static int pipeline_session_open(PipelineFleet *fleet, PipelineSession *session)
{
    memset(session, 0, sizeof *session);
    session->server = fleet->server.create_sequence(fleet->server.model);
    if (!session->server) return 0;
    session->layer_93 = fleet->tail.create_sequence(fleet->tail.model);
    if (!session->layer_93) { pipeline_session_close(fleet, session); return 0; }
    for (unsigned layer = 1; layer <= PIPELINE_LAYERS; layer++) {
        session->layer[layer] = fleet->layer[layer].create_sequence(fleet->layer[layer].model);
        if (!session->layer[layer]) { pipeline_session_close(fleet, session); return 0; }
    }
    return 1;
}

static void pipeline_fleet_close(PipelineFleet *fleet)
{
    for (unsigned layer = 1; layer <= PIPELINE_LAYERS; layer++) pipeline_stage_close(&fleet->layer[layer]);
    pipeline_stage_close(&fleet->tail);
    pipeline_stage_close(&fleet->server);
    memset(fleet, 0, sizeof *fleet);
}

static int pipeline_fleet_open(const char *root, PipelineFleet *fleet)
{
    char directory[4096], qkv[4096], trunk[4096], leaves[4096], dataset[4096], relative[128];
    memset(fleet, 0, sizeof *fleet);

    int (*server_open)(const char *, const char *, void **);
    if (!pipeline_path(directory, sizeof directory, root, "server") ||
        !pipeline_stage_library(&fleet->server, directory, "server") ||
        !pipeline_symbol(fleet->server.library, "server_open", &server_open, sizeof server_open) ||
        !pipeline_symbol(fleet->server.library, "server_process", &fleet->server_process, sizeof fleet->server_process) ||
        !pipeline_path(qkv, sizeof qkv, directory, "bin/dataset/trunk-0-qkv/qkv.bin") ||
        !pipeline_path(trunk, sizeof trunk, directory, "bin/dataset/trunk-0") ||
        !server_open(qkv, trunk, &fleet->server.model) ||
        !pipeline_sequences(&fleet->server, "server")) return 0;

    /* Layer 93 is a stage like any other: it owns final normalisation, the head and
       the post back to the caller, so the server only has to own the way in. */
    int (*layer_93_open)(const char *, const char *, const char *, const char *, const char *, void **);
    char head[4096], tiktoken[4096], vocabulary[4096], tokenizer_config[4096];
    if (!pipeline_path(directory, sizeof directory, root, "tansformers/transformer-93") ||
        !pipeline_stage_library(&fleet->tail, directory, "transformer_93") ||
        !pipeline_symbol(fleet->tail.library, "transformer_93_open", &layer_93_open, sizeof layer_93_open) ||
        !pipeline_symbol(fleet->tail.library, "transformer_93_process", &fleet->layer_93_process, sizeof fleet->layer_93_process) ||
        !pipeline_symbol(fleet->tail.library, "transformer_93_set_request", &fleet->layer_93_set_request, sizeof fleet->layer_93_set_request) ||
        !pipeline_symbol(fleet->tail.library, "transformer_93_delivered", &fleet->layer_93_delivered, sizeof fleet->layer_93_delivered) ||
        !pipeline_symbol(fleet->tail.library, "transformer_93_timing", &fleet->layer_93_timing, sizeof fleet->layer_93_timing) ||
        !pipeline_path(leaves, sizeof leaves, root, "server/bin/dataset/leaves.json") ||
        !pipeline_path(head, sizeof head, root, "server/bin/dataset/outputs/fruit.bin") ||
        !pipeline_path(tiktoken, sizeof tiktoken, root, "server/bin/dataset/tiktoken.model") ||
        !pipeline_path(vocabulary, sizeof vocabulary, root, "server/bin/dataset/vocabulary.bin") ||
        !pipeline_path(tokenizer_config, sizeof tokenizer_config, root, "server/bin/configs/tokenizer_config.json") ||
        !layer_93_open(leaves, head, tiktoken, vocabulary, tokenizer_config, &fleet->tail.model) ||
        !pipeline_sequences(&fleet->tail, "transformer_93")) return 0;

    for (unsigned layer = 1; layer <= PIPELINE_LAYERS; layer++) {
        int (*transformer_open)(const char *, void **);
        snprintf(relative, sizeof relative, "tansformers/transformer-%u", layer);
        if (!pipeline_path(directory, sizeof directory, root, relative) ||
            !pipeline_stage_library(&fleet->layer[layer], directory, "transformer") ||
            !pipeline_symbol(fleet->layer[layer].library, "transformer_open", &transformer_open, sizeof transformer_open) ||
            !pipeline_path(dataset, sizeof dataset, directory, "bin/dataset") ||
            !transformer_open(dataset, &fleet->layer[layer].model) ||
            !pipeline_sequences(&fleet->layer[layer], "transformer")) return 0;
        void *process = NULL;
        if (!pipeline_symbol(fleet->layer[layer].library, "transformer_process", &process, sizeof process)) return 0;
        if (layer == 1) memcpy(&fleet->layer_one, &process, sizeof process);
        else memcpy(&fleet->layer_many[layer], &process, sizeof process);
    }
    return 1;
}

static int pipeline_advance(PipelineFleet *fleet, PipelineSession *session, unsigned index,
    PipelineToken *tokens, size_t first, size_t count, PipelineTiming *timing)
{
    double mark = pipeline_clock();
    for (size_t position = first; position < count; position++)
        if (!fleet->server_process(fleet->server.model, session->server,
                tokens[position].residual, tokens[position].residual,
                &tokens[position].snapshots[0][0])) return 0;
    timing->server += pipeline_clock() - mark;
    timing->server_calls += (unsigned)(count - first);

    /* The stage is the route. The owner is arithmetic on it and crossing a stride
       hands the frame to the next service, so there is nothing here to look up. */
    for (int stage = clover_stage_first(1); ; ) {
        const int owner = clover_stage_owner(stage);
        if (owner < 1 || owner > PIPELINE_LAYERS) break;
        const unsigned layer = (unsigned)owner;
        const unsigned snapshots = (layer + 11) / 12;
        mark = pipeline_clock();
        for (size_t position = first; position < count; position++) {
            PipelineToken *token = &tokens[position];
            token->stage = stage;
            token->position = (uint32_t)position;
            if (!clover_frame_is_mine(token, owner)) {
                fprintf(stderr, "request %u stage %d is not layer %u's work\n", index, stage, layer);
                return 0;
            }
            int ok = layer == 1
                ? fleet->layer_one(fleet->layer[1].model, session->layer[1],
                      token->residual, &token->snapshots[0][0], token->residual)
                : fleet->layer_many[layer](fleet->layer[layer].model, session->layer[layer],
                      token->residual, &token->snapshots[0][0], snapshots,
                      token->residual, &token->snapshots[0][0]);
            if (!ok) { fprintf(stderr, "request %u layer %u failed at position %zu\n", index, layer, position); return 0; }
            token->snapshot_count = (layer + 12) / 12;
            token->stage = stage + CLOVER_STAGE_STRIDE - 1;
            clover_frame_advance(token, 1);
        }
        timing->layer[layer] += pipeline_clock() - mark;
        timing->calls[layer] += (unsigned)(count - first);
        stage = clover_stage_next(stage + CLOVER_STAGE_STRIDE - 1, 1);
    }
    return 1;
}

static unsigned long pipeline_read_bytes(void);
static unsigned long pipeline_major_faults(void);

/* Everything a request owns is created when it arrives and released when it finishes,
   so nothing is reserved for a request that never comes. */
static void *pipeline_request(void *argument)
{
    PipelineWork *work = argument;
    PipelineAdmission *admission = work->admission;
    double started = pipeline_clock();
    PipelineSession session; memset(&session, 0, sizeof session);
    PipelineTiming timing; memset(&timing, 0, sizeof timing);
    PipelineToken *tokens = calloc(PIPELINE_MAX_TOKENS, sizeof *tokens);
    ServerInput *input = NULL;
    char text[8192], path[4096];
    size_t text_length = 0, count = work->request.count, filled = 0;
    int ok = 0, delivered = 0;
    double normalize = 0, head = 0;
    unsigned long read_before = pipeline_read_bytes(), faults_before = pipeline_major_faults();

    timing.waited = started - work->submitted;

    if (!tokens || !pipeline_session_open(work->fleet, &session)) goto finish;
    if (!pipeline_path(path, sizeof path, work->root, "server/bin/dataset/inputs/seed.bin") ||
        !(input = server_input_open(path))) goto finish;
    work->fleet->layer_93_set_request(session.layer_93, work->index);

    for (size_t position = 0; position < count; position++) {
        clover_frame_init(&tokens[position], work->index);
        if (!server_input(input, work->request.ids[position], tokens[position].residual)) goto finish;
    }
    for (unsigned produced = 0; produced < work->request.wanted; produced++) {
        /* The last token is the one that ends the walk, and clover_stage_next says so:
           at the final stage with nothing more to produce it returns 0 rather than
           looping back to the server. That is the same bit layer 93 posts on. */
        int final = produced + 1 == work->request.wanted || count >= PIPELINE_MAX_TOKENS;
        if (!pipeline_advance(work->fleet, &session, work->index, tokens, filled, count, &timing)) goto finish;
        filled = count;
        uint32_t selected; TokenText word;
        double mark = pipeline_clock();
        if (!work->fleet->layer_93_process(work->fleet->tail.model, session.layer_93,
                tokens[count - 1].residual, &tokens[count - 1].snapshots[0][0],
                PIPELINE_SNAPSHOTS, final, work->request.callback, &selected, &word)) goto finish;
        timing.tail += pipeline_clock() - mark;
        timing.tail_calls++;
        if (text_length + word.length < sizeof text) {
            memcpy(text + text_length, word.bytes, word.length);
            text_length += word.length;
        }
        if (final) break;
        clover_frame_init(&tokens[count], work->index);
        if (!server_input(input, selected, tokens[count].residual)) goto finish;
        count++;
    }
    ok = 1;
finish:
    delivered = ok && session.layer_93 && work->request.callback[0]
        && work->fleet->layer_93_delivered(session.layer_93);
    if (session.layer_93)
        work->fleet->layer_93_timing(session.layer_93, &normalize, &head, &timing.deliver);
    pipeline_session_close(work->fleet, &session);
    server_input_close(input);
    free(tokens);
    pthread_mutex_lock(&admission->lock);
    printf("REQUEST %u %s %.3fs [", work->index, ok ? "OK" : "FAILED", pipeline_clock() - started);
    if (ok) fwrite(text, 1, text_length, stdout);
    fputs("]", stdout);
    if (work->request.callback[0]) fputs(delivered ? " DELIVERED\n" : " UNDELIVERED\n", stdout);
    else fputs("\n", stdout);
    if (getenv("CLOVER_TIMING")) {
        double total = pipeline_clock() - started;
        fprintf(stderr, "TIMING %u owner,name,calls,seconds\n", work->index);
        fprintf(stderr, "TIMING %u -,queue,1,%.6f\n", work->index, timing.waited);
        fprintf(stderr, "TIMING %u 0,server,%u,%.6f\n", work->index, timing.server_calls, timing.server);
        for (unsigned layer = 1; layer <= PIPELINE_LAYERS; layer++)
            fprintf(stderr, "TIMING %u %u,layer-%u,%u,%.6f\n",
                work->index, layer, layer, timing.calls[layer], timing.layer[layer]);
        fprintf(stderr, "TIMING %u 93,normalize,%u,%.6f\n", work->index, timing.tail_calls, normalize);
        fprintf(stderr, "TIMING %u 93,head,%u,%.6f\n", work->index, timing.tail_calls, head);
        fprintf(stderr, "TIMING %u 93,deliver,1,%.6f\n", work->index, timing.deliver);
        fprintf(stderr, "TIMING %u -,request-total,1,%.6f\n", work->index, total);
        fprintf(stderr, "IO %u read_bytes %lu major_faults %lu\n", work->index,
            pipeline_read_bytes() - read_before, pipeline_major_faults() - faults_before);
    }
    fflush(stdout);
    fflush(stderr);
    pthread_mutex_unlock(&admission->lock);
    free(work);
    return NULL;
}

/* A submission names where the answer should go. "SUBMIT <url> <wanted> <ids...>"
   is accepted and acknowledged at once; the bare "<wanted> <ids...>" form still
   reports on stdout, which is what the end-to-end harness reads. */
static int pipeline_parse(char *line, PipelineRequest *request)
{
    char *save = NULL, *field = strtok_r(line, " \t\r\n", &save);
    request->count = 0;
    request->callback[0] = 0;
    if (!field) return 0;
    if (!strcmp(field, "SUBMIT")) {
        field = strtok_r(NULL, " \t\r\n", &save);
        if (!field || strlen(field) >= sizeof request->callback) return 0;
        memcpy(request->callback, field, strlen(field) + 1);
        field = strtok_r(NULL, " \t\r\n", &save);
    }
    if (!field || !decimal_id((const unsigned char *)field, strlen(field), &request->wanted) ||
        !request->wanted || request->wanted > 64) return 0;
    while ((field = strtok_r(NULL, " \t\r\n", &save)) && request->count < 128)
        if (!decimal_id((const unsigned char *)field, strlen(field), &request->ids[request->count++])) return 0;
    return request->count > 0;
}

/* The limit is memory, not a chosen number of workers. Both figures are read from the
   kernel rather than assumed: what one request actually maps, and what the box has. */
static size_t pipeline_mapped(void)
{
    FILE *file = fopen("/proc/self/statm", "r");
    if (!file) return 0;
    unsigned long pages = 0;
    int read = fscanf(file, "%lu", &pages);
    fclose(file);
    return read == 1 ? (size_t)pages * (size_t)sysconf(_SC_PAGESIZE) : 0;
}

static size_t pipeline_available(void)
{
    FILE *file = fopen("/proc/meminfo", "r");
    if (!file) return 0;
    char label[64]; unsigned long value = 0; size_t available = 0;
    while (fscanf(file, "%63s %lu kB\n", label, &value) == 2)
        if (!strcmp(label, "MemAvailable:")) { available = (size_t)value * 1024; break; }
    fclose(file);
    return available;
}

/* Bytes this process actually pulled off storage, as opposed to found in cache. */
static unsigned long pipeline_read_bytes(void)
{
    FILE *file = fopen("/proc/self/io", "r");
    if (!file) return 0;
    char label[64]; unsigned long value = 0, bytes = 0;
    while (fscanf(file, "%63s %lu\n", label, &value) == 2)
        if (!strcmp(label, "read_bytes:")) { bytes = value; break; }
    fclose(file);
    return bytes;
}

/* Faults that had to wait for storage. Minor faults never touched a disk. */
static unsigned long pipeline_major_faults(void)
{
    FILE *file = fopen("/proc/self/stat", "r");
    if (!file) return 0;
    char line[4096];
    unsigned long faults = 0;
    if (fgets(line, sizeof line, file)) {
        char *after = strrchr(line, ')');
        if (after) sscanf(after + 1, " %*c %*d %*d %*d %*d %*d %*u %*u %*u %lu", &faults);
    }
    fclose(file);
    return faults;
}

static void *pipeline_worker(void *argument)
{
    PipelineQueue *queue = argument;
    for (;;) {
        PipelineWork *work;
        pthread_mutex_lock(&queue->lock);
        while (!queue->pending && !queue->stopping) pthread_cond_wait(&queue->arrived, &queue->lock);
        if (!queue->pending) { pthread_mutex_unlock(&queue->lock); break; }
        work = queue->entries[queue->head];
        queue->head = (queue->head + 1) % PIPELINE_QUEUE;
        queue->pending--;
        queue->active++;
        pthread_mutex_unlock(&queue->lock);

        pipeline_request(work);

        pthread_mutex_lock(&queue->lock);
        /* The ticket is held until the work is finished, not until it is dequeued, so
           a caller cannot resubmit the same prompt while the first one is still running. */
        queue->tickets[work->ticket].live = 0;
        queue->active--;
        if (!queue->pending && !queue->active) pthread_cond_broadcast(&queue->idle);
        pthread_mutex_unlock(&queue->lock);
    }
    return NULL;
}

static uint64_t pipeline_fold(uint64_t hash, const void *data, size_t length)
{
    const unsigned char *bytes = data;
    for (size_t index = 0; index < length; index++) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

typedef enum {
    PIPELINE_ADMITTED, PIPELINE_BUSY, PIPELINE_DUPLICATE, PIPELINE_FLOODING
} PipelineVerdict;

static PipelineVerdict pipeline_submit(PipelineQueue *queue, PipelineWork *work)
{
    const PipelineRequest *request = &work->request;
    uint64_t client = pipeline_fold(14695981039346656037ULL, request->callback, strlen(request->callback));
    uint64_t fingerprint = pipeline_fold(client, &request->wanted, sizeof request->wanted);
    size_t free_slot = PIPELINE_QUEUE, slot;
    unsigned held = 0;

    fingerprint = pipeline_fold(fingerprint, request->ids, request->count * sizeof request->ids[0]);

    pthread_mutex_lock(&queue->lock);
    if (queue->stopping || queue->pending == PIPELINE_QUEUE) {
        pthread_mutex_unlock(&queue->lock);
        return PIPELINE_BUSY;
    }
    for (slot = 0; slot < PIPELINE_QUEUE; slot++) {
        if (!queue->tickets[slot].live) { if (free_slot == PIPELINE_QUEUE) free_slot = slot; continue; }
        /* Identity is the callback address, so these only bind submissions that
           asked to be answered somewhere. The local stdin form carries none. */
        if (!request->callback[0]) continue;
        if (queue->tickets[slot].request == fingerprint) {
            pthread_mutex_unlock(&queue->lock);
            return PIPELINE_DUPLICATE;
        }
        if (queue->tickets[slot].client == client) held++;
    }
    if (request->callback[0] && held >= PIPELINE_PER_CLIENT) {
        pthread_mutex_unlock(&queue->lock);
        return PIPELINE_FLOODING;
    }
    if (free_slot == PIPELINE_QUEUE) {
        pthread_mutex_unlock(&queue->lock);
        return PIPELINE_BUSY;
    }

    queue->tickets[free_slot].request = fingerprint;
    queue->tickets[free_slot].client = client;
    queue->tickets[free_slot].live = 1;
    work->ticket = free_slot;
    queue->entries[queue->tail] = work;
    queue->tail = (queue->tail + 1) % PIPELINE_QUEUE;
    queue->pending++;
    pthread_cond_signal(&queue->arrived);
    pthread_mutex_unlock(&queue->lock);
    return PIPELINE_ADMITTED;
}

static int pipeline_serve(PipelineFleet *fleet, PipelineAdmission *admission,
    Tokenizer *tokenizer, const char *root)
{
    char line[8192];
    unsigned index = 0;
    PipelineQueue queue;
    pthread_t *pool;
    unsigned created = 0;

    memset(&queue, 0, sizeof queue);
    pthread_mutex_init(&queue.lock, NULL);
    pthread_cond_init(&queue.arrived, NULL);
    pthread_cond_init(&queue.idle, NULL);

    /* How many run at once is still the measured memory figure, not a chosen number.
       It now sizes the pool instead of blocking the caller at the door. */
    queue.workers = admission->per_request && admission->budget / admission->per_request
        ? (unsigned)(admission->budget / admission->per_request) : 1;
    pool = calloc(queue.workers, sizeof *pool);
    if (!pool) return 0;
    for (; created < queue.workers; created++)
        if (pthread_create(&pool[created], NULL, pipeline_worker, &queue)) break;
    if (!created) { free(pool); return 0; }
    queue.workers = created;

    /* poll() sees the descriptor, not stdio's buffer, so stdin must not hoard lines. */
    setvbuf(stdin, NULL, _IONBF, 0);
    fputs("READY\n", stdout); fflush(stdout);
    while (fgets(line, sizeof line, stdin)) {
        PipelineRequest request;
        if (!pipeline_parse(line, &request)) {
            fputs("BAD_REQUEST\n", stdout); fflush(stdout); continue;
        }
        PipelineWork *work = calloc(1, sizeof *work);
        if (!work) break;
        work->fleet = fleet; work->admission = admission; work->tokenizer = tokenizer;
        work->root = root; work->request = request; work->index = index;
        work->submitted = pipeline_clock();
        /* Refused submissions cost a line of output and nothing else. The caller is
           told which wall it hit so it knows whether to retry or stop. */
        switch (pipeline_submit(&queue, work)) {
        case PIPELINE_ADMITTED:
            index++;
            if (request.callback[0]) { printf("ACCEPTED %u\n", work->index); fflush(stdout); }
            continue;
        case PIPELINE_BUSY:      fputs("BUSY\n", stdout); break;
        case PIPELINE_DUPLICATE: fputs("DUPLICATE\n", stdout); break;
        case PIPELINE_FLOODING:  fputs("TOO_MANY\n", stdout); break;
        }
        fflush(stdout);
        free(work);
    }

    pthread_mutex_lock(&queue.lock);
    queue.stopping = 1;
    pthread_cond_broadcast(&queue.arrived);
    pthread_mutex_unlock(&queue.lock);
    for (unsigned worker = 0; worker < queue.workers; worker++) pthread_join(pool[worker], NULL);
    free(pool);
    fputs("END\n", stdout);
    fflush(stdout);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 3 || strcmp(argv[2], "serve")) { fputs("usage: pipeline CODE_ROOT serve\n", stderr); return 2; }
    const char *root = argv[1];
    char path[4096], vocab[4096], config[4096];
    Tokenizer *tokenizer = NULL;
    PipelineFleet fleet; memset(&fleet, 0, sizeof fleet);
    PipelineAdmission admission; memset(&admission, 0, sizeof admission);
    pthread_mutex_init(&admission.lock, NULL);
    pthread_cond_init(&admission.released, NULL);
    pthread_cond_init(&admission.drained, NULL);
    int success = 0;

    if (!pipeline_path(path, sizeof path, root, "server/bin/dataset/tiktoken.model") ||
        !pipeline_path(vocab, sizeof vocab, root, "server/bin/dataset/vocabulary.bin") ||
        !pipeline_path(config, sizeof config, root, "server/bin/configs/tokenizer_config.json") ||
        tokenizer_open(path, vocab, config, &tokenizer) != TOKENIZER_OK) goto finish;

    double opened = pipeline_clock();
    if (!pipeline_fleet_open(root, &fleet)) { fputs("fleet open failed\n", stderr); goto finish; }

    /* Price one request by building a session and watching what it maps, rather than
       trusting a constant that would rot the moment a layer changed shape. */
    {
        PipelineSession probe;
        size_t before = pipeline_mapped();
        if (!pipeline_session_open(&fleet, &probe)) { fputs("probe session failed\n", stderr); goto finish; }
        size_t after = pipeline_mapped();
        pipeline_session_close(&fleet, &probe);
        admission.per_request = (after > before ? after - before : 0)
            + PIPELINE_MAX_TOKENS * sizeof(PipelineToken);
    }
    admission.budget = pipeline_available() / 2;
    const char *limit = getenv("CLOVER_MEMORY_BUDGET_MB");
    if (limit && *limit) {
        char *end = NULL;
        unsigned long megabytes = strtoul(limit, &end, 10);
        if (end && !*end && megabytes) admission.budget = (size_t)megabytes * 1024 * 1024;
    }
    fprintf(stderr, "Fleet resident: server + %d layers in %.3fs; request costs %.0f MB, "
        "budget %.1f GB, so %zu run at once\n", PIPELINE_LAYERS, pipeline_clock() - opened,
        admission.per_request / 1048576.0, admission.budget / 1073741824.0,
        admission.per_request ? admission.budget / admission.per_request : 0);
    fflush(stderr);

    success = pipeline_serve(&fleet, &admission, tokenizer, root);
finish:
    pipeline_fleet_close(&fleet);
    tokenizer_close(tokenizer);
    if (!success) fputs("pipeline failed\n", stderr);
    return success ? 0 : 1;
}
