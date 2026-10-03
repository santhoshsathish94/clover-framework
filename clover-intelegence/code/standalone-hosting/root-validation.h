#ifndef CLOVER_ROOT_VALIDATION_H
#define CLOVER_ROOT_VALIDATION_H
static int root_validate_group(const Root *root,const unsigned char *codes,const unsigned char *map,uint32_t *checksum)
{
    uint32_t crc=*checksum;
    for (unsigned coordinate=0;coordinate<16;coordinate+=2) {
        unsigned packed0=codes[coordinate], packed1=codes[coordinate+1];
        unsigned value0=map[packed0&15], value1=map[packed0>>4];
        unsigned value2=map[packed1&15], value3=map[packed1>>4];
        if (value0>=root->palette_count || value1>=root->palette_count ||
            value2>=root->palette_count || value3>=root->palette_count) return 0;
        uint32_t first=root_u16(root->constants+value0*2)|(uint32_t)root_u16(root->constants+value1*2)<<16;
        uint32_t second=root_u16(root->constants+value2*2)|(uint32_t)root_u16(root->constants+value3*2)<<16;
        first^=crc;
        crc=root->crc_wide[6][first&255]^root->crc_wide[5][(first>>8)&255]^
            root->crc_wide[4][(first>>16)&255]^root->crc_wide[3][first>>24]^
            root->crc_wide[2][second&255]^root->crc_wide[1][(second>>8)&255]^
            root->crc_wide[0][(second>>16)&255]^root->crc_table[second>>24];
    }
    *checksum=crc;
    return 1;
}
#endif