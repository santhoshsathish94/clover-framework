# Clover AI Studio — Mobile Edition (`clover-ai-system/mobile/`)

Complete on-device 7-layer Clover AI System and Android APK (`ai.clover.studio`) engineered for the **Samsung Galaxy S23 (`Qualcomm Snapdragon 8 Gen 2 for Galaxy`, `SM8550-AC`)** unified SoC:

- **`android-app/CloverAIStudio.apk`**: Signed Android application (`APK Signature Scheme v2 & v3`) with official 5-leaf Clover adaptive launcher icons (`res/mipmap-anydpi-v26/`) and the botanical light/dark mobile Studio UI (`assets/index.html`) matching the Laptop LFM3-Omni Clover AI Studio.
- **`lfm3_cpu_engine.c`**: Native 24-layer C/OpenMP/NEON/OpenCL engine (`liblfm3_cpu_engine.so`) featuring:
  - **Heterogeneous Adreno 740 GPU + Kryo ARMv9 CPU Co-Execution**: Zero-copy OpenCL 3.0 `CL_MEM_USE_HOST_PTR` + `cl_khr_integer_dot_product` (`dot_acc_sat`) running SwiGLU rows `0..3839` on the Adreno 740 GPU while the Kryo CPU cores simultaneously compute rows `3840..7679` via `SDOT` (`7680 / 7680` bit-identical parity, `max_abs_diff = 0.0`).
  - **ARMv8.6-A `i8mm` (`SMMLA` `vmmlaq_s32`)**: 2×2 matrix tile broadcast coalescing (`32` Int8 MACs/instruction) for `T >= 2`.
  - **1-Pass `FlashAttention-2` Online Softmax**: Register-resident `(m_i, l_i, o_acc[0..127])` across all 11 GQA layers (`0` `scores[]` allocations, `0` strided V-cache misses).
  - **`LFM3-SoC-Q4` Mobile Format (`894.4 MB` Working Set)**: Two-stage 4-bit `lm_head.4bit.bin` (`129.39 MiB`) with Top-64 exact BF16 refinement (`23.65 ms`, `100.00%` top-1 match) + 4-bit planar group-64 `operator.4bit.bin` (`164.59 MiB`) + `ffn.4bit.bin` (`586.1 MB`).
  - **Direction B 4-Tier Mobile SoC Ledger (`lfm3_dm_extended_stats`)**: Tracks `bytes_tier0_reg`, `bytes_tier1_sram_slc`, `bytes_tier2_lpddr5x`, `bytes_saved_q4_and_broadcast`, `smmla_i8mm_tile_calls`, `fa2_online_softmax_calls`, `adreno_opencl_launches`, and Linux `/proc/self/stat` `minflt`/`majflt` page-fault deltas.
- **`s23_on_device_clover.py`**: Complete 7-layer on-device coordinator (`LFM3-Omni-2B` + Stage-3 `HybridLink2048` + `11-Level NCM` + `Memory Graph` + `Ontology GraphRAG` + `Python Generative Compiler` + `SpecMD Policy 3 madvise` + all **11 Foundational Papers** `1943–2017`).
- **`start_trusted_cdsp_shell.sh`**: Enters the trusted `u:r:shell:s0` (`shell_33_0`) SELinux domain (`vendor_hal_dspmanager_client`) via Termux local `adb` (`127.0.0.1`) to stage Hexagon V73 cDSP skel/shell files without root.
- **`lfm3_omni_s23_live_phone_report.json`**: Full live Samsung Galaxy S23 verification report across Layers 1–7, the 11 foundational papers, and the 4-mode Mobile SoC hardware/format benchmark.
