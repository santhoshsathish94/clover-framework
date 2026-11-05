"""Layer 2 planner on the LFM3 C engine (laptop).

The base LFM3 planner runs on the same memory-mapped weights as Layer 1, using the engine's cache (lfm3_cache_*), so
no second copy of the model is held in RAM. It stands in for the parts of a Hugging Face causal LM that Clover's
agent loop (clover_loop.run_agent_planning) and the VS Code step use:
    backbone(input_ids=..., past_key_values=..., use_cache=True, output_hidden_states=..., return_dict=True)
        -> .logits [1, 1, V] (last position), .past_key_values (handle), .hidden_states ([1, 1, D],)
    backbone.get_input_embeddings().weight
A call without past_key_values starts a new sequence and keeps the longest prefix already in the cache. The engine
holds one sequence: passing an older handle raises instead of silently computing on the wrong state.
Not supported: inputs_embeds (HybridLink latent steps). A trained LFM3 HybridAgent checkpoint uses the torch planner.
"""
import ctypes
from types import SimpleNamespace

import numpy as np
import torch

D_LFM3, V_LFM3 = 2048, 128000
_PF = ctypes.POINTER(ctypes.c_float)


class CacheHandle:
    __slots__ = ("length",)

    def __init__(self, length):
        self.length = length


class CEngineBackbone:
    def __init__(self, lib, embed_weight, max_tokens=8192):
        self.lib = lib
        lib.lfm3_cache_init.argtypes = [ctypes.c_int]
        lib.lfm3_cache_init.restype = ctypes.c_int
        lib.lfm3_cache_feed.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, _PF, _PF]
        lib.lfm3_cache_feed.restype = ctypes.c_int
        lib.lfm3_cache_truncate.argtypes = [ctypes.c_int]
        lib.lfm3_cache_truncate.restype = ctypes.c_int
        rc = lib.lfm3_cache_init(int(max_tokens))
        if rc != 0:
            raise RuntimeError(f"lfm3_cache_init({max_tokens}) failed with rc={rc}")
        self.max_tokens = int(max_tokens)
        self.ids = []
        self.live = None
        self._emb = embed_weight
        self._hid = np.zeros(D_LFM3, dtype=np.float32)
        self._lg = np.zeros(V_LFM3, dtype=np.float32)
        self.stats = {"calls": 0, "fed_tokens": 0, "reused_tokens": 0}

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self._emb)

    def __call__(self, input_ids=None, inputs_embeds=None, past_key_values=None, use_cache=True,
                 output_hidden_states=False, return_dict=True, **_):
        if inputs_embeds is not None:
            raise NotImplementedError("HybridLink latent steps need the torch planner (trained checkpoint)")
        new = [int(x) for x in (input_ids.reshape(-1).tolist() if hasattr(input_ids, "reshape") else input_ids)]
        if not new:
            raise ValueError("empty input_ids")
        if past_key_values is None:
            lcp, m = 0, min(len(self.ids), len(new) - 1)
            while lcp < m and self.ids[lcp] == new[lcp]:
                lcp += 1
            if self.lib.lfm3_cache_truncate(lcp) != 0:
                raise RuntimeError("lfm3_cache_truncate failed")
            self.ids = self.ids[:lcp]
            feed = new[lcp:]
            self.stats["reused_tokens"] += lcp
        else:
            if past_key_values is not self.live:
                raise RuntimeError("stale cache handle: the C engine keeps one sequence at a time")
            feed = new
        if len(self.ids) + len(feed) > self.max_tokens:
            raise ValueError(f"sequence of {len(self.ids) + len(feed)} tokens exceeds the engine cache ({self.max_tokens})")
        arr = (ctypes.c_int * len(feed))(*feed)
        rc = self.lib.lfm3_cache_feed(arr, len(feed), self._hid.ctypes.data_as(_PF), self._lg.ctypes.data_as(_PF))
        if rc < 0:
            raise RuntimeError(f"lfm3_cache_feed failed with rc={rc}")
        self.ids += feed
        self.stats["calls"] += 1
        self.stats["fed_tokens"] += len(feed)
        self.live = CacheHandle(len(self.ids))
        logits = torch.from_numpy(self._lg.copy()).view(1, 1, -1)
        hidden = (torch.from_numpy(self._hid.copy()).view(1, 1, -1),) if output_hidden_states else None
        return SimpleNamespace(logits=logits, past_key_values=self.live, hidden_states=hidden)


def _no_latent(*_a, **_k):
    raise NotImplementedError("HybridLink latent steps need the torch planner (trained checkpoint)")


def c_engine_planner(lib, embed_weight, max_tokens=8192):
    """Object with the `.backbone` / `.run_hybridlink` attributes clover_loop reads from `eng.ha_model`."""
    return SimpleNamespace(backbone=CEngineBackbone(lib, embed_weight, max_tokens), run_hybridlink=_no_latent,
                           hybridlink=None)
