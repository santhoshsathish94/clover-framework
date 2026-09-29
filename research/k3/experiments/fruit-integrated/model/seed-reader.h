#include <errno.h>
#include <openssl/sha.h>
#include <zlib.h>

typedef struct {
    FILE *file;
    uint32_t rows, width, block_rows, dictionary_count, bits, blocks;
    unsigned char *metadata, *dictionary, *index, *raw;
    size_t metadata_bytes, raw_capacity;
    uint32_t cached_block;
    uint64_t blocks_decoded, payload_bytes_read, rows_requested;
    unsigned char source_sha[32];
} SeedInput;

static uint32_t seed_u32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

static uint64_t seed_u64(const unsigned char *bytes)
{
    return (uint64_t)seed_u32(bytes) | (uint64_t)seed_u32(bytes + 4) << 32;
}

static unsigned seed_field(const unsigned char *bytes, size_t offset, unsigned bits)
{
    unsigned value = 0;
    for (unsigned bit = 0; bit < bits; bit++)
        value |= ((bytes[(offset + bit) / 8] >> ((offset + bit) % 8)) & 1U) << bit;
    return value;
}

static void seed_exact_read(FILE *file, unsigned char *bytes, size_t size)
{
    if (fread(bytes, 1, size, file) != size) die("seed short read");
}

static void seed_open(SeedInput *seed, const char *path)
{
    memset(seed, 0, sizeof(*seed));
    seed->cached_block = UINT32_MAX;
    seed->file = fopen(path, "rb");
    if (!seed->file) die("seed open failed");
    struct stat statbuf;
    if (fstat(fileno(seed->file), &statbuf) || statbuf.st_size < 128) die("seed file size invalid");
    unsigned char header[128];
    seed_exact_read(seed->file, header, sizeof(header));
    if (memcmp(header, "K3SEED1\0", 8) || seed_u32(header + 8) != 1 || seed_u32(header + 36))
        die("seed header version invalid");
    seed->rows = seed_u32(header + 12);
    seed->width = seed_u32(header + 16);
    seed->block_rows = seed_u32(header + 20);
    seed->dictionary_count = seed_u32(header + 24);
    seed->bits = seed_u32(header + 28);
    seed->blocks = seed_u32(header + 32);
    if (!seed->rows || seed->rows > 163840 || !seed->width || seed->width > 7168 ||
        !seed->block_rows || seed->block_rows > 256 || !seed->dictionary_count || seed->dictionary_count > 65536)
        die("seed dimensions unsupported");
    unsigned expected_bits = 0;
    for (unsigned remaining = seed->dictionary_count - 1; remaining; remaining >>= 1) expected_bits++;
    if (seed->bits != expected_bits || seed->blocks != (seed->rows + seed->block_rows - 1) / seed->block_rows)
        die("seed dictionary/index count invalid");
    const uint64_t index_offset = 128 + (uint64_t)seed->dictionary_count * 2;
    const uint64_t payload_offset = index_offset + (uint64_t)seed->blocks * 56;
    if (seed_u64(header + 40) != 128 || seed_u64(header + 48) != index_offset ||
        seed_u64(header + 56) != payload_offset || payload_offset > (uint64_t)statbuf.st_size)
        die("seed metadata offsets invalid");
    seed->metadata_bytes = (size_t)payload_offset - 128;
    unsigned char *authenticated = malloc(96 + seed->metadata_bytes);
    if (!authenticated) die("seed metadata allocation failed");
    memcpy(authenticated, header, 96);
    seed_exact_read(seed->file, authenticated + 96, seed->metadata_bytes);
    unsigned char hash[32];
    SHA256(authenticated, 96 + seed->metadata_bytes, hash);
    if (memcmp(hash, header + 96, 32)) die("seed metadata checksum differs");
    seed->metadata = malloc(seed->metadata_bytes);
    if (!seed->metadata) die("seed index allocation failed");
    memcpy(seed->metadata, authenticated + 96, seed->metadata_bytes);
    free(authenticated);
    seed->dictionary = seed->metadata;
    seed->index = seed->metadata + seed->dictionary_count * 2;
    memcpy(seed->source_sha, header + 64, 32);
    unsigned char seen[65536] = {0};
    for (unsigned entry = 0; entry < seed->dictionary_count; entry++) {
        const unsigned word = seed_field(seed->dictionary, (size_t)entry * 16, 16);
        if (seen[word]) die("seed dictionary duplicate");
        seen[word] = 1;
    }
    uint64_t cursor = payload_offset;
    for (unsigned block = 0; block < seed->blocks; block++) {
        const unsigned char *entry = seed->index + (size_t)block * 56;
        const unsigned remaining = seed->rows - block * seed->block_rows;
        const unsigned rows = remaining < seed->block_rows ? remaining : seed->block_rows;
        const uint32_t length = seed_u32(entry + 8);
        if (seed_u64(entry) != cursor || seed_u32(entry + 12) != rows || seed_u32(entry + 16) >= 6 ||
            length > (uint64_t)rows * seed->width * 2) die("seed block index invalid");
        cursor += length;
        if (cursor > (uint64_t)statbuf.st_size) die("seed block outside file");
    }
    if (cursor != (uint64_t)statbuf.st_size) die("seed file trailing bytes");
    seed->raw_capacity = (size_t)seed->block_rows * seed->width * 2;
    seed->raw = malloc(seed->raw_capacity);
    if (!seed->raw) die("seed raw block allocation failed");
}

static void seed_load_block(SeedInput *seed, unsigned block)
{
    if (block >= seed->blocks) die("seed block out of range");
    if (block == seed->cached_block) return;
    const unsigned char *entry = seed->index + (size_t)block * 56;
    const unsigned length = seed_u32(entry + 8), rows = seed_u32(entry + 12), codec = seed_u32(entry + 16);
    const size_t count = (size_t)rows * seed->width, raw_size = count * 2;
    const size_t decoded_size = codec <= 2 ? (count * seed->bits + 7) / 8 : raw_size;
    unsigned char *payload = malloc(length ? length : 1), *decoded = malloc(decoded_size + 1);
    if (!payload || !decoded) die("seed block allocation failed");
    if (fseeko(seed->file, (off_t)seed_u64(entry), SEEK_SET)) die("seed block seek failed");
    seed_exact_read(seed->file, payload, length);
    if ((uint32_t)crc32(0L, payload, length) != seed_u32(entry + 20)) die("seed payload checksum differs");
    if (codec == 0) {
        if (length != decoded_size) die("seed packed block length invalid");
        memcpy(decoded, payload, decoded_size);
    } else {
        z_stream decoder = {0};
        decoder.next_in = payload;
        decoder.avail_in = length;
        decoder.next_out = decoded;
        decoder.avail_out = (uInt)(decoded_size + 1);
        if (inflateInit(&decoder) != Z_OK) die("seed inflate init failed");
        const int status = inflate(&decoder, Z_FINISH);
        const int valid = status == Z_STREAM_END && decoder.total_out == decoded_size && decoder.avail_in == 0;
        inflateEnd(&decoder);
        if (!valid) die("seed compressed block framing invalid");
    }
    if (codec <= 2) {
        const size_t total_bits = count * seed->bits;
        if ((total_bits % 8) && (decoded[decoded_size - 1] >> (total_bits % 8))) die("seed field padding nonzero");
        for (size_t position = 0; position < count; position++) {
            unsigned identifier = 0;
            if (codec == 2) {
                for (unsigned bit = 0; bit < seed->bits; bit++)
                    identifier |= seed_field(decoded, (size_t)bit * count + position, 1) << bit;
            } else identifier = seed_field(decoded, position * seed->bits, seed->bits);
            if (identifier >= seed->dictionary_count) die("seed dictionary ID invalid");
            memcpy(seed->raw + position * 2, seed->dictionary + identifier * 2, 2);
        }
    } else if (codec == 3) memcpy(seed->raw, decoded, raw_size);
    else if (codec == 4) {
        for (size_t position = 0; position < count; position++) {
            seed->raw[position * 2] = decoded[position];
            seed->raw[position * 2 + 1] = decoded[count + position];
        }
    } else {
        for (size_t position = 0; position < count; position++) {
            unsigned word = 0;
            for (unsigned bit = 0; bit < 16; bit++) word |= seed_field(decoded, (size_t)bit * count + position, 1) << bit;
            seed->raw[position * 2] = word;
            seed->raw[position * 2 + 1] = word >> 8;
        }
    }
    unsigned char hash[32];
    SHA256(seed->raw, raw_size, hash);
    if (memcmp(hash, entry + 24, 32)) die("seed reconstructed block checksum differs");
    free(decoded);
    free(payload);
    seed->cached_block = block;
    seed->blocks_decoded++;
    seed->payload_bytes_read += length;
}

static void seed_read_row(SeedInput *seed, unsigned row, uint16_t *output)
{
    if (row >= seed->rows) die("seed row out of range");
    seed_load_block(seed, row / seed->block_rows);
    const unsigned char *bytes = seed->raw + (size_t)(row % seed->block_rows) * seed->width * 2;
    for (unsigned coordinate = 0; coordinate < seed->width; coordinate++)
        output[coordinate] = bytes[coordinate * 2] | (unsigned)bytes[coordinate * 2 + 1] << 8;
    seed->rows_requested++;
}

static void seed_close(SeedInput *seed)
{
    if (fclose(seed->file)) die("seed close failed");
    free(seed->metadata);
    free(seed->raw);
    seed->file = NULL;
}