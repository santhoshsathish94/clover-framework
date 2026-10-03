#include "sha256.h"
#include <assert.h>
#include <stdio.h>

static void check(const unsigned char *data, size_t count, const char *expected)
{
    unsigned char digest[32]; char hex[65];
    client_sha256(data,count,digest);
    for (unsigned index=0;index<32;index++) snprintf(hex+index*2,3,"%02x",digest[index]);
    assert(!strcmp(hex,expected));
}

int main(void)
{
    check((const unsigned char *)"",0,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    check((const unsigned char *)"abc",3,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    check((const unsigned char *)"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",56,
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    ClientHash hash; client_hash_init(&hash);
    for(unsigned index=0;index<1000000;index++) client_hash_update(&hash,(const unsigned char *)"a",1);
    unsigned char digest[32];char hex[65];client_hash_final(&hash,digest);
    for(unsigned index=0;index<32;index++)snprintf(hex+index*2,3,"%02x",digest[index]);
    assert(!strcmp(hex,"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
    puts("PASS: SHA256 empty, abc, multiblock and million-a incremental vectors");
}