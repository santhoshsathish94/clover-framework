import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,existsSync} from 'node:fs';
const original=readFileSync(new URL('../distrubuted-hosting/tansformers/transformer-1/root.h',import.meta.url),'utf8').replaceAll('\r\n','\n');
let source=original;
function replace(before,after){before=before.replaceAll('\t','    ');after=after.replaceAll('\t','    ');assert.equal(source.split(before).length,2,before);source=source.replace(before,after);}
replace('#include <stdlib.h>','#include <stdlib.h>\n#include <fcntl.h>\n#include <unistd.h>\n#include <sys/mman.h>\n#include <omp.h>\n#if defined(__AVX2__)\n#include <immintrin.h>\n#endif\n#include "root-metadata.h"');
replace('    ROOT_CONSTANT_BYTES = 15880, ROOT_PALETTE = 66, ROOT_MAPS = 465,\n    ROOT_TEMPLATES = 515, ROOT_REFS = 3638, ROOT_RAW = 121856, ROOT_COMPRESSED = 229376 };','    ROOT_RAW = 121856, ROOT_COMPRESSED = 229376, ROOT_DOWN_RAW = 104448, ROOT_EXPERT_RAW = 17547264 };');
replace('    unsigned char constants[ROOT_CONSTANT_BYTES];','    unsigned char *constants;\n    unsigned constant_bytes, palette_count, map_count, template_count, reference_count;');
replace('    double values[ROOT_PALETTE];','    double *values;');
replace('    free(root->index);','    if (root->direct) munmap((void *)root->direct, root->direct_bytes);\n    free(root->pairs);\n    free(root->constants);\n    free(root->values);\n    free(root->index);');
replace('static Root *root_open(const char *directory)','static Root *root_open(const char *directory, unsigned layer)');
replace('    if (!root) return NULL;','    if (!root) return NULL;\n    unsigned counts[4];\n    int metadata_length = snprintf(path, sizeof path, "%s/maps.json", directory);\n    if (metadata_length < 0 || (size_t)metadata_length >= sizeof path || !root_metadata(path, layer, counts)) { root_close(root); return NULL; }\n    root->palette_count = counts[0]; root->map_count = counts[1];\n    root->template_count = counts[2]; root->reference_count = counts[3];\n    root->constant_bytes = counts[0]*2 + counts[1]*16 + (counts[2]+1)*2 + counts[3]*2;\n    root->constants = malloc(root->constant_bytes);\n    root->values = malloc(root->palette_count * sizeof *root->values);\n    if (!root->constants || !root->values) { root_close(root); return NULL; }');
for(const [name,member] of Object.entries({ROOT_CONSTANT_BYTES:'constant_bytes',ROOT_PALETTE:'palette_count',ROOT_MAPS:'map_count',ROOT_TEMPLATES:'template_count',ROOT_REFS:'reference_count'}))source=source.replaceAll(name,'root->'+member);
replace('    if (!root_seek(root->file, root->offsets[ordinal]) ||\n        fread(scratch->compressed, 1, bytes, root->file) != bytes ||','    if (!root_read_at(root->file, scratch->compressed, bytes, root->offsets[ordinal]) ||');
replace('    for (unsigned block = 0; block < rows / 64; block++) {\n        if (!root_block(root, scratch, expert, matrix, block)) return 0;','    int valid = 1;\n    RootScratch *pool = scratch;\n    double converted[3584];\n    for (unsigned coordinate = 0; coordinate < width; coordinate++) converted[coordinate] = (double)input[coordinate];\n#pragma omp parallel for schedule(static) reduction(&:valid)\n    for (unsigned block = 0; block < rows / 64; block++) {\n        RootScratch *scratch = pool + omp_get_thread_num();\n        if (!root_block(root, scratch, expert, matrix, block)) { valid = 0; continue; }');
replace('    return 1;\n}\n#endif','    return valid;\n}\n#endif');
replace('    unsigned counts[4];','    unsigned counts[4] = {0};\n    struct stat constants_info;\n    int constants_length = snprintf(path, sizeof path, "%s/constants.bin", directory);\n    if (constants_length < 0 || (size_t)constants_length >= sizeof path || stat(path, &constants_info) || constants_info.st_size <= 0) { root_close(root); return NULL; }');
replace('!root_metadata(path, layer, counts)','!root_metadata(path, layer, (size_t)constants_info.st_size, counts)');
replace('#include "root-metadata.h"','#include "root-metadata.h"\n\n#ifndef ROOT_PHASE_BEGIN\n#define ROOT_PHASE_BEGIN(name)\n#define ROOT_PHASE_END(name,phase)\n#endif');
replace('    double *values;','    double *values;\n    float (*pairs)[256][2];\n    const unsigned char *direct;\n    size_t direct_bytes;\n    const unsigned char *prefetched;\n    size_t prefetched_bytes;\n    unsigned prefetched_expert;');
replace('    uint32_t crc_table[256];','    uint32_t crc_table[256];\n    uint32_t crc_wide[7][256];');
replace('        root->crc_table[byte] = crc;\n    }','        root->crc_table[byte] = crc;\n    }\n    for (unsigned byte=0;byte<256;byte++) {\n        uint32_t crc=root->crc_table[byte];\n        for (unsigned depth=0;depth<7;depth++) {\n            crc=root->crc_table[crc&255]^(crc>>8);\n            root->crc_wide[depth][byte]=crc;\n        }\n    }');
replace('static int root_block(', '#include "root-validation.h"\n\nstatic int root_block(');
replace('    if (!root_read_at(root->file, scratch->compressed, bytes, root->offsets[ordinal]) ||\n        !decode_zlib(scratch->compressed, bytes, scratch->raw, 64 * width / 2 + 64 * width / 32)) return 0;',
`    ROOT_PHASE_BEGIN(read_started);
	const unsigned char *compressed=scratch->compressed;
	if (root->prefetched && root->prefetched_expert==expert) {
		uint64_t offset=root->offsets[ordinal]-root->offsets[expert*152];
		if (offset>root->prefetched_bytes || bytes>root->prefetched_bytes-offset) return 0;
		compressed=root->prefetched+(size_t)offset;
	} else if (!root_read_at(root->file, scratch->compressed, bytes, root->offsets[ordinal])) return 0;
	ROOT_PHASE_END(read_started,0);
	ROOT_PHASE_BEGIN(decode_started);
	if (!decode_zlib(compressed, bytes, scratch->raw, 64 * width / 2 + 64 * width / 32)) return 0;
	ROOT_PHASE_END(decode_started,1);
	ROOT_PHASE_BEGIN(check_started);`);
replace(`        for (unsigned coordinate = 0; coordinate < 32; coordinate++) {
			unsigned code = (scratch->raw[group * 16 + coordinate / 2] >> ((coordinate & 1) * 4)) & 15;
			unsigned value_id = map[code];
			if (value_id >= root->palette_count) return 0;
			for (unsigned byte = 0; byte < 2; byte++)
				crc = root->crc_table[(crc ^ root->constants[value_id * 2 + byte]) & 255] ^ (crc >> 8);
		}`, '        if (!root_validate_group(root,scratch->raw+group*16,map,&crc)) return 0;');
replace('    return (crc ^ UINT32_MAX) == root_u32(root->index + ROOT_MATRICES * 2 + ordinal * 8 + 4);',
	'    int valid=(crc ^ UINT32_MAX) == root_u32(root->index + ROOT_MATRICES * 2 + ordinal * 8 + 4);\n    ROOT_PHASE_END(check_started,2);\n    return valid;');
replace('        if (!root_block(root, scratch, expert, matrix, block)) { valid = 0; continue; }',
	'        const unsigned char *data;\n        if (root->direct) data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +\n            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW\n                        : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);\n        else if (root_block(root, scratch, expert, matrix, block)) data = scratch->raw;\n        else { valid = 0; continue; }\n        ROOT_PHASE_BEGIN(math_started);');
replace('            const unsigned char *codes = scratch->raw + row * width / 2;\n            const unsigned char *selectors = scratch->raw + 64 * width / 2 + row * width / 32;',
	'            const unsigned char *codes = data + row * width / 2;\n            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;');
replace('    if (!valid || root->offsets[ROOT_BLOCKS] - root->offsets[0] != root_u64(header + 24) ||\n        !root_length(root->file, root->offsets[ROOT_BLOCKS])) { root_close(root); return NULL; }\n    return root;',
	'    int direct_length = snprintf(path, sizeof path, "%s/experts.direct", directory);\n    if (direct_length > 0 && (size_t)direct_length < sizeof path) {\n        int handle = open(path, O_RDONLY);\n        if (handle >= 0) {\n            struct stat direct_info;\n            size_t bytes = (size_t)896 * ROOT_EXPERT_RAW;\n            if (!fstat(handle, &direct_info) && (size_t)direct_info.st_size == bytes) {\n                void *mapped = mmap(NULL, bytes, PROT_READ, MAP_SHARED, handle, 0);\n                if (mapped != MAP_FAILED) { root->direct = mapped; root->direct_bytes = bytes; }\n            }\n            close(handle);\n        }\n    }\n    if (!valid || root->offsets[ROOT_BLOCKS] - root->offsets[0] != root_u64(header + 24) ||\n        (!root->direct && !root_length(root->file, root->offsets[ROOT_BLOCKS]))) { root_close(root); return NULL; }\n    return root;');
replace('    }\n    return valid;\n}\n#endif','        ROOT_PHASE_END(math_started,3);\n    }\n    return valid;\n}\n#endif');
/* One byte holds two codes, so a single lookup yields both the even and odd column value. */
replace('    for (unsigned index = 0; index < root->reference_count; index++) if (root_u16(refs + index * 2) >= root->map_count) valid = 0;',
`    for (unsigned index = 0; index < root->reference_count; index++) if (root_u16(refs + index * 2) >= root->map_count) valid = 0;
    root->pairs = malloc((size_t)root->map_count * sizeof *root->pairs);
    if (!root->pairs) { root_close(root); return NULL; }
    for (unsigned entry = 0; entry < root->map_count; entry++) {
        const unsigned char *map = maps + entry * 16;
        for (unsigned byte = 0; byte < 256; byte++) {
            unsigned low = map[byte & 15], high = map[byte >> 4];
            root->pairs[entry][byte][0] = low < root->palette_count ? (float)root->values[low] : 0.0f;
            root->pairs[entry][byte][1] = high < root->palette_count ? (float)root->values[high] : 0.0f;
        }
    }`);
replace(`            double lanes[16] = {0};
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                unsigned map_id = root_u16(refs + (begin + selectors[coordinate / 32]) * 2);
                const unsigned char *map = maps + map_id * 16;
                for (unsigned lane = 0; lane < 16; lane++) {
                    unsigned column = coordinate + lane;
                    unsigned code = (codes[column / 2] >> ((column & 1) * 4)) & 15;
                    lanes[lane] = lanes[lane] + root->values[map[code]] * (double)input[column];
                }
            }
            double sums[4];
            for (unsigned lane = 0; lane < 4; lane++)
                sums[lane] = (lanes[lane] + lanes[lane + 8]) + (lanes[lane + 4] + lanes[lane + 12]);
            output[block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));`,
`            double sums[4];
#if defined(__AVX2__)
            __m256d lane0 = _mm256_setzero_pd(), lane4 = _mm256_setzero_pd();
            __m256d lane8 = _mm256_setzero_pd(), lane12 = _mm256_setzero_pd();
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                const unsigned char *packed = codes + coordinate / 2;
                const double *source = converted + coordinate;
                __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]]))));
                __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]]))));
                __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]]))));
                __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]]))));
                lane0 = _mm256_add_pd(lane0, _mm256_mul_pd(w0, _mm256_loadu_pd(source)));
                lane4 = _mm256_add_pd(lane4, _mm256_mul_pd(w1, _mm256_loadu_pd(source + 4)));
                lane8 = _mm256_add_pd(lane8, _mm256_mul_pd(w2, _mm256_loadu_pd(source + 8)));
                lane12 = _mm256_add_pd(lane12, _mm256_mul_pd(w3, _mm256_loadu_pd(source + 12)));
            }
            _mm256_storeu_pd(sums, _mm256_add_pd(_mm256_add_pd(lane0, lane8), _mm256_add_pd(lane4, lane12)));
#else
            double lanes[16] = {0};
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                for (unsigned lane = 0; lane < 16; lane++) {
                    unsigned column = coordinate + lane;
                    lanes[lane] = lanes[lane] + (double)pair[codes[column / 2]][column & 1] * converted[column];
                }
            }
            for (unsigned lane = 0; lane < 4; lane++)
                sums[lane] = (lanes[lane] + lanes[lane + 8]) + (lanes[lane + 4] + lanes[lane + 12]);
#endif
            output[block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));`);
const writeTarget=new URL('live-root.h',import.meta.url);
if(existsSync(writeTarget))assert.equal(readFileSync(writeTarget,'utf8'),source,'Refusing to overwrite a changed generated reader');
else writeFileSync(writeTarget,source);
console.log('PASS: derived generic reader from original decoder/math; runtime metadata counts and cursor-free parallel reads');