#ifndef CLIENT_SHA256_H
#define CLIENT_SHA256_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct { uint32_t state[8]; uint64_t bytes; unsigned used; unsigned char block[64]; } ClientHash;

static uint32_t client_rotate(uint32_t value, unsigned count)
{
    return (value >> count) | (value << (32 - count));
}

static void client_hash_block(ClientHash *hash, const unsigned char *data)
{
    static const uint32_t constants[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
    uint32_t schedule[64], working[8];
    for (unsigned index = 0; index < 16; index++) schedule[index] =
        (uint32_t)data[index*4] << 24 | (uint32_t)data[index*4+1] << 16 |
        (uint32_t)data[index*4+2] << 8 | data[index*4+3];
    for (unsigned index = 16; index < 64; index++) {
        uint32_t earlier = schedule[index-15], recent = schedule[index-2];
        schedule[index] = schedule[index-16] + (client_rotate(earlier,7)^client_rotate(earlier,18)^(earlier>>3)) +
            schedule[index-7] + (client_rotate(recent,17)^client_rotate(recent,19)^(recent>>10));
    }
    memcpy(working,hash->state,sizeof working);
    for (unsigned index = 0; index < 64; index++) {
        uint32_t first = working[7] + (client_rotate(working[4],6)^client_rotate(working[4],11)^client_rotate(working[4],25)) +
            ((working[4]&working[5])^(~working[4]&working[6])) + constants[index] + schedule[index];
        uint32_t second = (client_rotate(working[0],2)^client_rotate(working[0],13)^client_rotate(working[0],22)) +
            ((working[0]&working[1])^(working[0]&working[2])^(working[1]&working[2]));
        for (unsigned offset = 7; offset > 0; offset--) working[offset] = working[offset-1];
        working[4] += first;
        working[0] = first + second;
    }
    for (unsigned index = 0; index < 8; index++) hash->state[index] += working[index];
}

static void client_hash_init(ClientHash *hash)
{
    const uint32_t initial[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memset(hash,0,sizeof *hash);
    memcpy(hash->state,initial,sizeof initial);
}

static void client_hash_update(ClientHash *hash, const unsigned char *data, size_t count)
{
    hash->bytes += count;
    while (count) {
        size_t take = 64 - hash->used;
        if (take > count) take = count;
        memcpy(hash->block+hash->used,data,take);
        hash->used += (unsigned)take; data += take; count -= take;
        if (hash->used == 64) { client_hash_block(hash,hash->block); hash->used = 0; }
    }
}

static void client_hash_final(ClientHash *hash, unsigned char digest[32])
{
    uint64_t bits = hash->bytes * 8;
    unsigned char padding[128] = {0x80};
    size_t count = hash->used < 56 ? 56-hash->used : 120-hash->used;
    for (unsigned index = 0; index < 8; index++) padding[count+index] = (unsigned char)(bits >> (56-index*8));
    client_hash_update(hash,padding,count+8);
    for (unsigned index = 0; index < 32; index++) digest[index] = (unsigned char)(hash->state[index/4] >> (24-(index%4)*8));
}

static void client_sha256(const unsigned char *data, size_t count, unsigned char digest[32])
{
    ClientHash hash; client_hash_init(&hash); client_hash_update(&hash,data,count); client_hash_final(&hash,digest);
}
#endif