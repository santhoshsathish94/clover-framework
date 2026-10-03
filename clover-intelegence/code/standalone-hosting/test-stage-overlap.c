#define main standalone_program_main
#include "clover-one.c"
#undef main
#include <assert.h>

int main(int argc,char **argv)
{
    assert(argc==2);
    assert(generation_config_load("bin/configs/generation.json",&resident_config));
    resident_config.expert_result_cache_mib=0;
    resident_configure(argv[1]); result_options();
    load_index(getenv("K3_INDEX")); prepared_init(); resident_startup(0);
    assert(resident_pipeline.initialized && !resident_results.sets);
    const unsigned layers[]={1,3,92};
    float input[E],latent[LAT],base_gate[SI],base_up[SI],base_shared[E],
        gate[SI],up[SI],shared[E],base_down[LAT];
    for (unsigned coordinate=0;coordinate<E;coordinate++) input[coordinate]=(float)((int)(coordinate%19)-9)/19.0f;
    for (unsigned coordinate=0;coordinate<LAT;coordinate++) latent[coordinate]=(float)((int)(coordinate%13)-6)/13.0f;
    for (unsigned index=0;index<3;index++) {
        unsigned layer=layers[index]; prepared_open((int)layer); operator_open((int)layer);
        const unsigned char *first=slot_ptr((int)layer,S_SH1), *second=slot_ptr((int)layer,S_SH3), *third=slot_ptr((int)layer,S_SH2);
        float *base_gate_row=base_gate,*base_up_row=base_up,*base_shared_row=base_shared;
        const float *input_row=input;
        Qm(&base_gate_row,&input_row,1,first,E,SI); Qm(&base_up_row,&input_row,1,second,E,SI);
        situ(base_gate,base_gate,base_up,SI);
        const float *activated=base_gate;
        Qm(&base_shared_row,&activated,1,third,SI,E);
        memcpy(base_down,resident_expert(layer,0,latent),sizeof base_down);
        double wait=resident_pipeline.wait_seconds,started=now_s();
        assert(expert_pipeline_submit(&resident_pipeline,resident_roots[layer],0,0));
        resident_shared(gate,up,shared,input,first,second,third);
        assert(expert_pipeline_acquire(&resident_pipeline,resident_roots[layer],0,0));
        assert(!memcmp(base_down,resident_expert(layer,0,latent),sizeof base_down));
        expert_pipeline_release(resident_roots[layer]);
        assert(!memcmp(base_gate,gate,sizeof gate) && !memcmp(base_up,up,sizeof up) && !memcmp(base_shared,shared,sizeof shared));
        printf("PASS: layer%u shared-stage/read overlap bit-exact; wait=%.6fs total=%.6fs\n",layer,
            resident_pipeline.wait_seconds-wait,now_s()-started); fflush(stdout);
        for (unsigned slot=0;slot<37;slot++) {
            const PreparedRecord *record=prepared_slots[slot];
            if (operator_vectors[slot]) {
                assert(record && prepared_vector((int)layer,slot,record->count));
            } else if (operator_pointers[slot]) {
                assert(record);
                OperatorMatrix matrix=operator_matrix_view(slot_ptr((int)layer,(int)slot),record->columns,record->rows);
                assert(matrix.codes && matrix.scales && matrix.palette);
            }
        }
        operator_close(); prepared_close();
    }
    resident_shutdown();
    puts("PASS: fresh-input overlap uses no prior request or result cache");
    return 0;
}