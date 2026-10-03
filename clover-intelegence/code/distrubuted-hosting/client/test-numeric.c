#define CLIENT_NO_MAIN
#include "client.c"
#include <assert.h>

int main(int argc,char **argv)
{
    if(argc!=3)return 2;
    ClientInput *input=client_input_open(argv[1]);
    ClientOutput *output=client_output_open(argv[2]);
    assert(input&&output);
    assert(!client_input_open(argv[2]));assert(!client_output_open(argv[1]));
    float vector[CLIENT_WIDTH];
    const unsigned ids[]={0,15,16,176,1008,163839};
    for(unsigned index=0;index<sizeof ids/sizeof *ids;index++)assert(client_input(input,ids[index],vector));
    assert(!client_input(input,163840,vector));
    assert(table_load(output->table,11));
    assert(table_u32(output->table->index+11*56+16)==5);
    vector[0]=NAN;uint32_t token=UINT32_MAX;
    assert(!client_output_project(output,vector,NULL,&token)&&token==UINT32_MAX);
    client_input_close(input);client_output_close(output);
    puts("PASS: input rows, distinct table identities, output bitplane block and invalid numeric inputs");
    return 0;
}