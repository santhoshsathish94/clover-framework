# Binding-direction intervention hypothesis

Status: discovery-based mathematical hypothesis, not a verified improvement.

Discovery comparison uses 56 MCQs with correct free answers and explicit numeric-binding references: 31 failed, 25 successful.
Direction = mean(arithmetic-reference | failure) minus mean(arithmetic-reference | success), normalized, then aligned with the stored post-MLP norm gains. This subtracts a shared prompt-condition contrast, but does not remove every possible confound.
Column attribution combines actual pre-down-projection activation differences and each column's write along that direction. Rank uses failure-minus-success contribution and penalizes successful-case contribution magnitude.

For exactly one MLP output tensor $W$ and its chosen columns $S$, test $W'_{:,S}=W_{:,S}-\alpha u(u^T W_{:,S})$; all other columns/tensors remain original.
The post-MLP RMS denominator is nonlinear and is not modeled by this approximation. Actual generated answers and regression tests decide whether the proposal works.
Fixed strengths: 0.5, 1.0, 2.0. Test the top two discovery-ranked layers separately, never combined. No norm, embedding or output-head edits; no gradients or optimizer.
Expected benefit: fewer default-position answers when arithmetic must be mapped to a label. Risks: removal of task-relevant representations, worse arithmetic/free response, selection or multimodal behavior.
Reversal: restore the exact original target-tensor bytes in the copy and verify the whole checkpoint SHA-256. Original source remains read-only.

## Layer 22

Target `model.language_model.layers.22.feed_forward.w2.weight`; shape [2048, 7680]; score 0.77690514.
Failure contribution 1.1320775; successful contribution 0.18339604.
Exact columns: [170, 190, 453, 511, 846, 1077, 1078, 1195, 1318, 1772, 1993, 2050, 2053, 2078, 2091, 2212, 2376, 2404, 2767, 2796, 2820, 2823, 2983, 3001, 3073, 3709, 4130, 4199, 4371, 4432, 4439, 4443, 4451, 4479, 4543, 4671, 4731, 4745, 4808, 4816, 4904, 4918, 4921, 4928, 4987, 5055, 5074, 5542, 5603, 5653, 6079, 6191, 6339, 6368, 6620, 6686, 6920, 6970, 6987, 7140, 7238, 7292, 7328, 7531]

## Layer 23

Target `model.language_model.layers.23.feed_forward.w2.weight`; shape [2048, 7680]; score 0.60007373.
Failure contribution 0.6589902; successful contribution -0.023149272.
Exact columns: [127, 228, 286, 335, 383, 478, 514, 664, 708, 717, 793, 941, 1025, 1537, 1676, 1685, 1815, 1998, 2262, 2397, 2472, 2496, 2595, 2611, 2671, 2761, 2789, 2844, 2848, 2964, 3237, 3272, 3620, 3715, 3948, 3952, 4059, 4078, 4114, 4180, 4269, 4285, 4417, 5012, 5086, 5116, 5537, 5682, 5779, 5947, 5965, 6190, 6218, 6307, 6430, 6545, 6630, 6660, 6733, 6776, 7016, 7118, 7167, 7257]
