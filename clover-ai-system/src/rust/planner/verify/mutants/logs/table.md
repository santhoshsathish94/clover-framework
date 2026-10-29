| M00 | M00: control: unmodified copy | d719148fa7f6299409041b0e1f2f3f07 | NOT CAUGHT | 0/2236 | P1=0/500 P2=0/500 P3=0/36 P4=0/500 P5=0/600 P6=0/100 | - |
| M01 | M01: M01 sweep: no re-look after swap (i += 1 after rel swap_remove) | d7e7dff8688a62e4daac612e9b75c44a | CAUGHT | 222/2236 | P1=0/500 P2=148/500 P3=8/36 P4=42/500 P5=9/600 P6=15/100 | P2 4 |
| M02 | M02: M02 pend_cap return keeps planned_to = i + 1 | 20581db7e472d3bc7ea32dca1c87d715 | CAUGHT | 1110/2236 | P1=339/500 P2=203/500 P3=21/36 P4=268/500 P5=279/600 P6=0/100 | P1 3 |
| M03 | M03: M03 skipped not counted | a5ed973b63fb28c970cb6a12e2825fce | CAUGHT | 1056/2236 | P1=419/500 P2=217/500 P3=21/36 P4=234/500 P5=165/600 P6=0/100 | P1 2 |
| M04 | M04: M04 write-back threshold 9/10 | dc34a74b36a3f6448da6466df2e42d82 | CAUGHT | 7/2236 | P1=0/500 P2=4/500 P3=0/36 P4=3/500 P5=0/600 P6=0/100 | P2 78 |
| M05 | M05: M05 full[card] never set | f7ce76b07f886a5732fc9966c85645dc | CAUGHT | 453/2236 | P1=0/500 P2=265/500 P3=17/36 P4=117/500 P5=52/600 P6=2/100 | P2 3 |
| M06 | M06: M06 pend_take after remove in vq_issue | 81baa6c8e6076f892a63a2734dbe234d | NOT CAUGHT | 0/2236 | P1=0/500 P2=0/500 P3=0/36 P4=0/500 P5=0/600 P6=0/100 | - |
| M07 | M07: M07 vq prot without layer (raw expert numbers) | d50a2cd4f6c3336a8dac466c964faaba | CAUGHT | 610/2236 | P1=0/500 P2=357/500 P3=18/36 P4=142/500 P5=92/600 P6=1/100 | P2 1 |
| M08 | M08: M08 replaced pending freed at once (no graveyard) | 8b55cc53c5c094f6f3590b4d2d3b6c00 | NOT CAUGHT | 0/2236 | P1=0/500 P2=0/500 P3=0/36 P4=0/500 P5=0/600 P6=0/100 | - |
| M09 | M09: M09 out counts every arrival | 93a01e2613f950bef478b1d761b68103 | CAUGHT | 123/2236 | P1=0/500 P2=52/500 P3=16/36 P4=33/500 P5=22/600 P6=0/100 | P2 12 |
| M10 | M10: M10 cand always rt_next_use | 547eb3117183e5de8ca37489f545da68 | CAUGHT | 597/2236 | P1=0/500 P2=350/500 P3=18/36 P4=138/500 P5=91/600 P6=0/100 | P2 1 |
| M11 | M11: M11 pos incremented before the plan | b2623bdf65a148ccbb384c1c1816fa46 | CAUGHT | 2209/2236 | P1=500/500 P2=500/500 P3=36/36 P4=500/500 P5=573/600 P6=100/100 | P1 1 |
| M12 | M12: M12 read-ahead heat 2.0 | 74e32b698afcf4607d1b4bfb73e49a12 | CAUGHT | 1807/2236 | P1=500/500 P2=461/500 P3=36/36 P4=476/500 P5=334/600 P6=0/100 | P1 1 |
| M13 | M13: M13 last = pos + lookahead (no +1) | 57296c1cfab9cdc4e4c8d3ceebbb7c02 | CAUGHT | 1105/2236 | P1=229/500 P2=304/500 P3=29/36 P4=242/500 P5=301/600 P6=0/100 | P1 2 |
| M14 | M14: M14 pend_take counts used only when it waited | f90b8057fe1d7cc1827c2543cde22821 | CAUGHT | 1637/2236 | P1=487/500 P2=444/500 P3=36/36 P4=456/500 P5=214/600 P6=0/100 | P1 1 |
| M15 | M15: M15 sp_config field order swapped (cards/lookahead) | d2d49367d3277439cd967a459f3ff258 | CAUGHT | 1662/2236 | P1=391/500 P2=459/500 P3=31/36 P4=387/500 P5=392/600 P6=2/100 | P1 1 |
| M16 | M16: M16 sweep wb: do not erase pending entry when done | c9aebead929bd29c2a667220ce9f1f28 | CAUGHT | 210/2236 | P1=0/500 P2=147/500 P3=6/36 P4=23/500 P5=12/600 P6=22/100 | P2 4 |
| M17 | M17: M17 host worth checked before pend_cap | cee6d0f7c464a79e6180eb7a9b0caf10 | CAUGHT | 1110/2236 | P1=339/500 P2=203/500 P3=21/36 P4=268/500 P5=279/600 P6=0/100 | P1 3 |
| M18 | M18: M18 arr_wait does not remove the entry first (get, then remove after the spin) | 59f39d9a7928d76598f9724b482d2b79 | CAUGHT | 612/2236 | P1=0/500 P2=326/500 P3=16/36 P4=123/500 P5=88/600 P6=59/100 | P2 1 |
| M19 | M19: M19 mismatch does not unset next-use | 95aaffb967736dc8998cf7b1aee31492 | CAUGHT | 794/2236 | P1=0/500 P2=0/500 P3=0/36 P4=499/500 P5=295/600 P6=0/100 | P4 1 |
| M20 | M20: M20 vq victim wb inserted into wb_keys twice-order: push before acquire | d06fa25267088850897feb613951834f | CAUGHT | 188/2236 | P1=0/500 P2=147/500 P3=6/36 P4=23/500 P5=12/600 P6=0/100 | P2 4 |
| M21 | M21: M21 t_plan also counts mismatch path (timer around everything) | fe563cd20fc471548d9eaac789644b88 | CAUGHT | 4/2236 | P1=0/500 P2=0/500 P3=0/36 P4=4/500 P5=0/600 P6=0/100 | P4 70 |
| MA | MA: vq_issue: try_acquire gets (cand_nu, key) instead of (key, cand_nu) | 0ab9b312ebc5e93b9e4f248fe68a670b | CAUGHT | 670/2236 | P1=0/500 P2=358/500 P3=18/36 P4=143/500 P5=92/600 P6=59/100 | P2 1 |
| MB | MB: pend_take counts pf_used also when there is no pending entry | c7d46745c14f070f08cfe5f49263921b | CAUGHT | 915/2236 | P1=262/500 P2=216/500 P3=36/36 P4=290/500 P5=111/600 P6=0/100 | P1 2 |
| MC | MC: card-pool prot list drops its last element | e1ae35af9d17fb72ee22bc8ef281c275 | CAUGHT | 603/2236 | P1=0/500 P2=353/500 P3=18/36 P4=140/500 P5=92/600 P6=0/100 | P2 1 |
| MD | MD: job off = e * expert_raw as a 32-bit multiply (both job builders) | 7f161ff4b123c9597b1d206bdedb5b3e | CAUGHT | 1868/2236 | P1=500/500 P2=500/500 P3=36/36 P4=496/500 P5=236/600 P6=100/100 | P1 1 |
| MD1 | MD1: job off as a 32-bit multiply, read-ahead job (kind 0) only | 86426133305b35086b50a6738dc1bd68 | CAUGHT | 1780/2236 | P1=500/500 P2=460/500 P3=36/36 P4=476/500 P5=214/600 P6=94/100 | P1 1 |
| MD2 | MD2: job off as a 32-bit multiply, card-queue job (kind 1/2) only | b14cea7c5a8776042b5563d60558fafe | CAUGHT | 573/2236 | P1=0/500 P2=328/500 P3=10/36 P4=123/500 P5=53/600 P6=59/100 | P2 1 |
