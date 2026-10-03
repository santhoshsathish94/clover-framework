#define main standalone_program_main
#include "clover-one.c"
#undef main
#include <assert.h>
#include <sys/wait.h>

int main(int argc, char **argv)
{
    assert(argc == 2);
    resident_configure(argv[1]);
    char path[4096];
    resident_path(path, sizeof(path), "outputs/fruit.bin");
    assert(!client_input_open(path));
    resident_path(path, sizeof(path), "inputs/seed.bin");
    assert(!client_output_open(path));
    ClientInput *input = client_input_open(path);
    assert(input);
    float vector[E];
    assert(!client_input(input, CLIENT_ROWS, vector));
    client_input_close(input);
    assert(!setenv("K3_RESULT_SET", "france", 1));
    recorded_open(1);
    int32_t expert;
    memcpy(&expert, recorded_roots, sizeof(expert));
    float unmatched[LAT];
    memcpy(unmatched, recorded_inputs, sizeof(unmatched));
    unmatched[0] = nextafterf(unmatched[0], INFINITY);
    for (unsigned record = 0; record < 80; record++)
        assert(memcmp(unmatched, recorded_inputs + (size_t)record * LAT * 4, sizeof(unmatched)));
    pid_t child = fork();
    assert(child >= 0);
    if (!child) {
        recorded_down(1, expert, unmatched);
        _exit(99);
    }
    int status;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 3);
    assert(!munmap((void *)recorded_roots, recorded_root_bytes));
    assert(!munmap((void *)recorded_inputs, recorded_input_bytes));
    puts("PASS: crossed containers, invalid token and unmatched expert input rejected; no fallback");
    return 0;
}