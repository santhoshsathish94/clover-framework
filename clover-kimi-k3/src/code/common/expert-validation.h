#ifndef CLOVER_EXPERT_VALIDATION_H
#define CLOVER_EXPERT_VALIDATION_H
static int expert_store_validate_group(const ExpertStore *store,const unsigned char *codes,const unsigned char *map,uint32_t *checksum)
{
    uint32_t crc=*checksum;
    for (unsigned coordinate=0;coordinate<16;coordinate+=2) {
        unsigned packed0=codes[coordinate], packed1=codes[coordinate+1];
        unsigned value0=map[packed0&15], value1=map[packed0>>4];
        unsigned value2=map[packed1&15], value3=map[packed1>>4];
        if (value0>=store->palette_count || value1>=store->palette_count ||
            value2>=store->palette_count || value3>=store->palette_count) return 0;
        uint32_t first=expert_store_u16(store->constants+value0*2)|(uint32_t)expert_store_u16(store->constants+value1*2)<<16;
        uint32_t second=expert_store_u16(store->constants+value2*2)|(uint32_t)expert_store_u16(store->constants+value3*2)<<16;
        first^=crc;
        crc=store->crc_wide[6][first&255]^store->crc_wide[5][(first>>8)&255]^
            store->crc_wide[4][(first>>16)&255]^store->crc_wide[3][first>>24]^
            store->crc_wide[2][second&255]^store->crc_wide[1][(second>>8)&255]^
            store->crc_wide[0][(second>>16)&255]^store->crc_table[second>>24];
    }
    *checksum=crc;
    return 1;
}
#endif