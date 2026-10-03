#include "generation.h"
#include <assert.h>

typedef struct { unsigned calls, emitted, stop; int fail; } FakeModel;
static int fake_step(void *state, unsigned token, unsigned position, int project, unsigned *next)
{
    FakeModel *model = state;
    assert(position == model->calls++ && token < GENERATION_VOCAB);
    if (model->fail) return 0;
    if (project) *next = model->emitted == model->stop ? 163585 : 387;
    return 1;
}
static void fake_emit(void *state,unsigned token,unsigned index)
{
    FakeModel *model = state;
    assert(index == model->emitted++ && token < GENERATION_VOCAB);
}
int main(int argc,char **argv)
{
    assert(argc == 2);
    const unsigned char empty[] = "[]";
    JsonReader reader = {empty, empty + 2, 0};
    assert(json_skip(&reader) && reader.cursor == reader.end);
    unsigned char decoded[4]; size_t written;
    assert(decode_base64((const unsigned char *)"YQ==",4,decoded,sizeof decoded,&written) && written==1 && decoded[0]=='a');
    GenerationConfig config;
    assert(generation_config_load(argv[1],&config));
    assert(config.max_input_tokens == 128 && config.max_output_tokens == 128);
    const char *bad_configs[] = {"{}", "{\"max_input_tokens\":128,\"max_output_tokens\":129,\"eos_token_id\":163585}",
        "{\"max_input_tokens\":0,\"max_output_tokens\":128,\"eos_token_id\":163585}"};
    for (unsigned index=0; index<sizeof bad_configs/sizeof *bad_configs;index++)
        assert(!generation_config_parse((const unsigned char *)bad_configs[index],strlen(bad_configs[index]),&config));
    const char *bad_requests[] = {"{\"input_ids\":[]}","{\"input_ids\":[163840]}","{\"input_ids\":[-1]}",
        "{\"input_ids\":[1],\"max_new_tokens\":129}","{\"input_ids\":[1],\"max_new_tokens\":0}",
        "{\"input_ids\":[1],\"input_ids\":[2]}","{\"input_ids\":[1]} trailing"};
    GenerationRequest request;
    for (unsigned index=0;index<sizeof bad_requests/sizeof *bad_requests;index++)
        assert(!generation_request_parse((const unsigned char *)bad_requests[index],strlen(bad_requests[index]),&config,&request));
    char text[2048];
    for (unsigned count=1;count<=129;count++) {
        size_t used=(size_t)sprintf(text,"{\"input_ids\":[");
        for (unsigned position=0;position<count;position++) used+=(size_t)sprintf(text+used,"%s387",position?",":"");
        used+=(size_t)sprintf(text+used,"]}");
        int valid=generation_request_parse((unsigned char *)text,used,&config,&request);
        assert(valid == (count<=128));
        if (!valid) continue;
        FakeModel model={.stop=999}; unsigned produced; int eos;
        assert(generation_run(&config,&request,fake_step,fake_emit,&model,&produced,&eos));
        assert(produced==128 && !eos && model.emitted==128 && model.calls==count+127);
    }
    request=(GenerationRequest){.tokens={387},.count=1,.max_new_tokens=128};
    FakeModel model={.stop=1}; unsigned produced; int eos;
    assert(generation_run(&config,&request,fake_step,fake_emit,&model,&produced,&eos));
    assert(eos && produced==2 && model.calls==2);
    model=(FakeModel){.fail=1};
    assert(!generation_run(&config,&request,fake_step,fake_emit,&model,&produced,&eos) && produced==0);
    puts("PASS: input lengths 1..128, output limit128, EOS, overflow and failure controls; synthetic engine only");
    return 0;
}