"""NCM runtime pieces the LFM3 renderer needs, copied from training scripts so that none of their
K3 / training imports come along. Source file and line of each copy are noted; repo commit 63c29a8."""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

L_WMT = 64
P_BPE = 48
P_RH = 48


# eval_both_tracks.py:22
def words(t):
    return re.findall(r"[a-z']+", t.lower())


# eval_both_tracks.py:26
def jacc(x, y):
    sx, sy = set(words(x)), set(words(y))
    return len(sx & sy) / max(len(sx | sy), 1)


# eval_both_tracks.py:70
@torch.no_grad()
def generate_from_zpref(net, tk, z_prefix, pad_mask_bool, temps_per_level, top_k=0, top_p=1.0):
    """Generate 12 levels of codes given z_prefix [B, P, 384] and pad_mask_bool [B, P]."""
    dev = z_prefix.device
    B, P, _ = z_prefix.shape
    vq = tk.vqvae.vq
    levels = list(vq.levels)
    z_hat = torch.zeros(B, levels[-1], tk.vqvae.embed_dim, device=dev)

    prefix_emb = net.transformer.wte(z_prefix)
    if not net.use_rope:
        prefix_emb = prefix_emb + net.wpe(net.wp_prefix)
    level_pos_embed = net.transformer.wle(net.lvl_1L) if net.config.use_level_encoding else torch.zeros(1, net.L - levels[0], net.config.n_embd, device=dev)
    if not net.use_rope:
        level_pos_embed = level_pos_embed + net.wpe(net.wp_1L)

    _pbias = torch.zeros(B, 1, 1, P, device=dev)
    _pbias[:, :, :, :P] = torch.where(pad_mask_bool[:, None, None, :], torch.tensor(float("-inf"), device=dev), torch.tensor(0.0, device=dev))

    rope_p = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
    h = prefix_emb
    pfx_mask = torch.zeros(1, 1, P, P, device=dev) + _pbias
    for block in net.transformer.blocks:
        h = block(h, attn_mask=pfx_mask, rope_cos_sin=rope_p)
    h = net.transformer.ln_f(h)
    l0_logits = net.lm_head_l0(net._level0_slice(h, pad_mask_bool))
    idx = net._sample_level_raw(l0_logits, temps_per_level[0], top_k, top_p)

    generated_idx = [idx]
    idx_next = idx
    level_acum = 0
    for level_idx in range(len(levels) - 1):
        l_next = levels[level_idx + 1]
        z_q_level = vq.embeddings[level_idx](idx_next)
        z_hat, x = vq.get_next_autoregressive_input(level_idx, z_hat, z_q_level)
        x = net.transformer.wte(x) + level_pos_embed[:, level_acum:level_acum + l_next, :]
        logits = net._decode_level_logits(x, prefix_emb, level_idx, _pbias, hidx=level_idx)
        idx_next = net._sample_level_raw(logits, temps_per_level[level_idx + 1], top_k, top_p)
        generated_idx.append(idx_next)
        level_acum += l_next
    return torch.cat(generated_idx, dim=1)


# train_combo_prefix.py:59
def pack_combo_prefix(z_rh, msk_rh_bool, z_bpe, msk_bpe_bool):
    """Packs [valid R(h) tokens, valid BPE tokens, PAD...] contiguously so (~pad_mask).sum(1)-1 is always the last valid token."""
    valid_rh = ~msk_rh_bool
    valid_bpe = ~msk_bpe_bool
    z_cat = torch.cat([z_rh, z_bpe], dim=1)
    valid_cat = torch.cat([valid_rh, valid_bpe], dim=1)
    perm = (~valid_cat).int().argsort(dim=1, stable=True)
    z_combo = z_cat.gather(1, perm.unsqueeze(-1).expand(-1, -1, z_cat.shape[-1]))
    pad_mask_combo = ~valid_cat.gather(1, perm)
    return z_combo, pad_mask_combo


# train_k3_copy_decoder.py:42
class CrossBlock(nn.Module):
    def __init__(self, d=384, n_head=6, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d = d
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.cx_q = nn.Linear(d, d)
        self.cx_kv = nn.Linear(d, 2 * d)
        self.cx_proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, mem_pad_mask=None):
        B, T, C = x.shape
        nh = self.n_head
        hd = C // nh
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=-1)
        q = q.view(B, T, nh, hd).transpose(1, 2)
        k = k.view(B, T, nh, hd).transpose(1, 2)
        v = v.view(B, T, nh, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))

        hq = self.ln_q(x)
        hkv = self.ln_kv(mem)
        cq = self.cx_q(hq).view(B, T, nh, hd).transpose(1, 2)
        ck, cv = self.cx_kv(hkv).split(C, dim=-1)
        M = mem.shape[1]
        ck = ck.view(B, M, nh, hd).transpose(1, 2)
        cv = cv.view(B, M, nh, hd).transpose(1, 2)
        attn_bias = None
        if mem_pad_mask is not None:
            attn_bias = torch.zeros(B, 1, 1, M, device=x.device, dtype=cq.dtype)
            attn_bias.masked_fill_(mem_pad_mask[:, None, None, :], float("-inf"))
        cy = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=attn_bias, is_causal=False, dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.cx_proj(cy.transpose(1, 2).contiguous().view(B, T, C)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


# train_e7_clean_chat.py:62 (K3LexicalPointerDecoder); constructor only: the LFM3 renderer runs these
# submodules itself with LFM3's pad id, so the K3-specific forward() is not needed.
class LexicalPointerDecoder(nn.Module):
    def __init__(self, d=384, n_layer=3, n_head=6):
        super().__init__()
        self.d = d
        self.tgt_pos = nn.Embedding(L_WMT, d)
        self.plan_pos = nn.Embedding(P_BPE, d)
        self.rh_pos = nn.Embedding(P_RH, d)
        self.blocks = nn.ModuleList([CrossBlock(d, n_head, 0.1) for _ in range(n_layer)])
        self.ln_out = nn.LayerNorm(d)
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.copy_q = nn.Linear(d, d)
        self.copy_k = nn.Linear(d, d)
        self.copy_gate = nn.Linear(d * 2, 1)
        self.copy_scale = nn.Parameter(torch.tensor(18.0))


# conversational-chat-engine.py:519
def format_readable_response(raw_joined: str) -> str:
    saved_fenced = []

    def _stash_fenced(m):
        idx_f = len(saved_fenced)
        saved_fenced.append(m.group(0).strip())
        return f"\n__FENCED_CODE_{idx_f}__\n"

    working = re.sub(r"```[a-zA-Z0-9_+-]*\n[\s\S]*?```", _stash_fenced, raw_joined)
    lines = [ln.strip() for ln in working.splitlines() if ln.strip()]
    if not lines:
        return raw_joined.strip()

    formatted_blocks = []
    for idx, ln in enumerate(lines):
        m_fc = re.match(r"^__FENCED_CODE_(\d+)__$", ln)
        if m_fc:
            fc_i = int(m_fc.group(1))
            if 0 <= fc_i < len(saved_fenced):
                formatted_blocks.append(saved_fenced[fc_i])
            continue
        ln_clean = re.sub(r"^[•\-*]\s*", "", ln).strip()
        if ln_clean.endswith(":") and len(ln_clean.split()) <= 10 and idx > 0:
            formatted_blocks.append(ln_clean)
            continue

        m_item = re.match(r"^([^:]{2,95}):\s+(.+)$", ln_clean)
        if m_item and (idx > 0 or len(lines) > 1):
            label, body = m_item.group(1).strip(), m_item.group(2).strip()
            if len(label.split()) <= 16 and not label.lower().startswith(("http", "https", "here is", "here's", "note")):
                formatted_blocks.append(f"• {label}: {body}")
                continue

        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", ln_clean) if s.strip()]
        if len(sents) >= 3:
            for s_i in range(0, len(sents), 2):
                formatted_blocks.append(" ".join(sents[s_i : s_i + 2]))
        else:
            formatted_blocks.append(ln_clean)

    out_str = ""
    for i, blk in enumerate(formatted_blocks):
        if i == 0:
            out_str = blk
        else:
            prev_is_bullet = formatted_blocks[i - 1].startswith("• ")
            cur_is_bullet = blk.startswith("• ")
            if prev_is_bullet and cur_is_bullet:
                out_str += "\n" + blk
            else:
                out_str += "\n\n" + blk
    return out_str


# conversational-chat-engine.py:582
def join_rendered_windows(txts, plans):
    out = []
    for i, t in enumerate(txts):
        t_clean = t.strip()
        if not t_clean:
            continue
        if i == 0:
            out.append(t_clean)
        else:
            p_orig = plans[i] if i < len(plans) else ""
            is_newline = (
                p_orig.startswith("\n")
                or any(
                    t_clean.startswith(pfx)
                    for pfx in ("Block ", "Session ", "Step ", "Morning", "Afternoon", "Evening", "Night", "Tip:", "Note:")
                )
                or bool(re.match(r"^(?:\d{1,2}:\d{2}|\d+\.|[A-Z][A-Za-z0-9\s()\-]{2,55}:\s)", t_clean))
            )
            out.append(("\n" if is_newline else " ") + t_clean)
    return format_readable_response("".join(out))


# conversational-chat-engine.py:606
GRAMMAR_CONNECTOR_WORDS = [
    "the", "The", "a", "A", "an", "An", "in", "In", "on", "On", "at", "At", "to", "To", "for", "For",
    "of", "by", "with", "from", "as", "into", "over", "after", "before", "during", "between", "through",
    "and", "or", "but", "so", "while", "when", "where", "which", "that", "who", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "it", "its", "this", "these",
    "they", "them", "their", "we", "our", "you", "your", "I", "my", "me",
    ".", ",", ";", ":", "!", "?", "'s", "'", "-", "(", ")", "/", "//", "://", "_", "=", "&", "#", "[", "]",
    "https", "http", "www", "com", "org", "io", "ai",
]
