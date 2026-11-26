"""HybridSFT objective on top of HybridAgentModel: the backbone
loss (cross-entropy at text positions and anchors) and the HybridLink loss (cosine + MSE
regression at latent positions), in the teacher-forced activation form and in the
two-pass self-conditioned form.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hybrid_agent.model import HybridAgentModel

FORMAT_ACTIVATION = "format_activation"
HYBRIDLINK_ACTIVATION = "hybridlink_activation"
SELF_CONDITIONED = "self_conditioned"
STAGES = (FORMAT_ACTIVATION, HYBRIDLINK_ACTIVATION, SELF_CONDITIONED)

# Latent regression weights (lambda_cos, lambda_mse) and the extra weight of the exit
# anchor, whose regression target is Emb(</latent>).
LAMBDA_COS = 1.0
LAMBDA_MSE = 0.2
EXIT_ANCHOR_HYBRIDLINK_WEIGHT = 1.0

# Vocabulary chunking for the CE loss keeps the full-sequence logits off memory.
CE_CHUNK_SIZE = 2048


class HybridSFTModel(HybridAgentModel):

    def set_trainable(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
        train_backbone = stage in (FORMAT_ACTIVATION, SELF_CONDITIONED)
        train_link = stage in (HYBRIDLINK_ACTIVATION, SELF_CONDITIONED)
        for p in self.backbone.parameters():
            p.requires_grad = train_backbone
        for p in self.hybridlink.parameters():
            p.requires_grad = train_link

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return (values * mask).sum() / denom

    def _chunked_ce(
        self, hidden: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted mean next-token CE; hidden[:, t] predicts input_ids[:, t+1]."""
        output_head = self.backbone.get_output_embeddings()
        src = hidden[:, :-1]
        tgt = input_ids[:, 1:]
        msk = loss_mask[:, 1:].float()
        denom = msk.sum().clamp(min=1.0)

        def _chunk_ce(h: torch.Tensor, t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            logits = output_head(h).float()
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), t.reshape(-1), reduction="none")
            return (ce * m.reshape(-1)).sum()

        # zero-valued anchor keeps the loss in the graph when every chunk is masked
        total = src.sum() * 0.0
        use_ckpt = torch.is_grad_enabled() and src.requires_grad
        for start in range(0, src.shape[1], CE_CHUNK_SIZE):
            end = min(start + CE_CHUNK_SIZE, src.shape[1])
            m = msk[:, start:end]
            if m.sum() == 0:
                continue
            h, t = src[:, start:end], tgt[:, start:end]
            if use_ckpt:
                total = total + checkpoint(_chunk_ce, h, t, m, use_reentrant=False)
            else:
                total = total + _chunk_ce(h, t, m)
        return total.float() / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        latent_mask: Optional[torch.Tensor] = None,
        stage: str = SELF_CONDITIONED,
    ) -> Dict[str, torch.Tensor]:
        if loss_mask is None:
            loss_mask = attention_mask
        if latent_mask is None:
            latent_mask = torch.zeros_like(attention_mask, dtype=torch.float32, device=input_ids.device)

        embed_layer = self.backbone.get_input_embeddings()
        input_embeds = embed_layer(input_ids)
        decoder = self.backbone.get_decoder()

        if stage == SELF_CONDITIONED:
            # pass 1: teacher forced; HybridLink predicts the next latent inputs one step ahead
            with torch.no_grad():
                teacher_embeds = embed_layer(input_ids)
            hidden = decoder(
                inputs_embeds=input_embeds, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            ).last_hidden_state
            input_embeds_mod = teacher_embeds.clone()
            pred_next = self.run_hybridlink(hidden[:, :-1], output_dtype=teacher_embeds.dtype)
            mask = latent_mask[:, :-1].unsqueeze(-1).expand_as(pred_next).bool()
            input_embeds_mod[:, 1:, :] = torch.where(mask, pred_next, input_embeds_mod[:, 1:, :])
            # pass 2: the model's own latent inputs replace the ground-truth embeddings
            hidden = decoder(
                inputs_embeds=input_embeds_mod, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            ).last_hidden_state
            target_embeds = teacher_embeds
        else:
            hidden = decoder(
                inputs_embeds=input_embeds, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            ).last_hidden_state
            target_embeds = input_embeds

        backbone_loss = self._chunked_ce(hidden, input_ids, loss_mask)

        # latent positions regress HybridLink(hidden[t]) onto Emb(token[t+1]); the last
        # position of each span targets the exit anchor
        latent_src_mask = latent_mask[:, :-1] * attention_mask[:, :-1]
        next_is_latent = latent_mask[:, 1:]
        exit_mask = latent_src_mask * (1.0 - next_is_latent)

        if stage == FORMAT_ACTIVATION:
            hybridlink_loss = torch.tensor(0.0, device=input_ids.device)
            cos_loss = hybridlink_loss
            mse_loss = hybridlink_loss
        else:
            preds = self.run_hybridlink(hidden[:, :-1], output_dtype=target_embeds.dtype)
            preds_f = preds.float()
            targets_f = target_embeds[:, 1:].float()
            cosine = 1.0 - F.cosine_similarity(preds_f, targets_f, dim=-1)
            mse = (preds_f - targets_f).pow(2).mean(dim=-1)

            cos_loss = self._masked_mean(cosine, latent_src_mask)
            mse_loss = self._masked_mean(mse, latent_src_mask)
            hybridlink_loss = LAMBDA_COS * cos_loss + LAMBDA_MSE * mse_loss

            exit_cos = self._masked_mean(cosine, exit_mask)
            exit_mse = self._masked_mean(mse, exit_mask)
            hybridlink_loss = hybridlink_loss + EXIT_ANCHOR_HYBRIDLINK_WEIGHT * (
                LAMBDA_COS * exit_cos + LAMBDA_MSE * exit_mse
            )

        if stage == FORMAT_ACTIVATION:
            total_loss = backbone_loss
        else:
            total_loss = backbone_loss + hybridlink_loss

        return {
            "loss": total_loss,
            "backbone_loss": backbone_loss.detach(),
            "hybridlink_loss": hybridlink_loss.detach(),
            "cos_loss": cos_loss.detach(),
            "mse_loss": mse_loss.detach(),
        }
