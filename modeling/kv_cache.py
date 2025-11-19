import torch
import torch.nn as nn
from vllm.attention import Attention
from modeling.segmenter import Rule

from typing import Optional

class LukaKVCache(nn.Module):
    def __init__(self,
                 base_attn: Attention,
                 boundary_rule: Rule = None):
        super().__init__()
        """
        Constructs the LukaKVCache. Note that one LukaKVCache is created per
        layer, as the heads are parallelized. See the Attention class usage.
        """
        self.base_attn = base_attn
        self.boudary_rule = boundary_rule
        self.has_prefilled = False
    
    @torch.no_grad()
    def prefill(
        self,
        q: torch.Tensor,  # [B, T, H, D]
        k: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, H, D]
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill path. For now, just calls the underlying Attention.

        Args:
            q, k, v: same shapes as Qwen3Attention currently passes to self.attn
            positions: optional, unused for now

        Returns:
            attn_output: same as base_attn(q, k, v)
        """
        attn_output = self.base_attn(q, k, v)
        attn_probs = self._compute_causal_attention_probs(q, k) # [T, T]
        self._print_boundaries_from_probs(attn_probs)

    @torch.no_grad()
    def decode(
        self,
        q: torch.Tensor,  # [B, T, H, D]
        k: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, H, D]
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode path. For now, just calls the underlying Attention.

        Args:
            q, k, v: same shapes as prefill()
            positions: optional, unused for now

        Returns:
            attn_output: same as base_attn(q, k, v)
        """
        return self.base_attn(q, k, v)

    def forward(
        self,
        q: torch.Tensor,  # [B, T, H, D]
        k: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, H, D]
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Default call path used by Qwen3Attention today: self.attn(q, k, v).

        Signature is backward-compatible:
          - Qwen3Attention currently calls self.attn(q, k, v) with no
            positions/is_prefill.
          - Those extra args are optional and ignored for now.

        Later you can:
          - detect prefill/decode here,
          - or have Qwen3Attention call prefill()/decode() explicitly.
        """
        # For now we completely ignore positions/is_prefill and just behave
        # exactly like the original Attention.
        if not self.has_prefilled:
            self.has_prefilled = True
            return self.prefill(q, k, v, positions)
        else:
            return self.decode(q, k, v, positions)
    
    def _compute_causal_attention_probs(
        self,
        q: torch.Tensor,  # arbitrary shape, last dim = feature dim
        k: torch.Tensor,  # same batch shape as q
    ) -> torch.Tensor:
        """
        Compute a causal attention probability tensor from q,k without assuming
        a particular (B, T, H, D) layout.

        Strategy:
          - Flatten all leading dims into a single "token" axis N.
          - Compute [N, N] dot-product attention.
          - Apply causal mask.
          - Softmax to get probabilities.
          - Return as [1, 1, N, N] so _print_boundaries_from_probs can
            treat it as a single (batch=0, head=0) matrix.
        """
        # q, k: [..., D]
        if q.shape != k.shape:
            raise ValueError(
                f"LukaKVCache._compute_causal_attention_probs: q and k shape mismatch: "
                f"q={q.shape}, k={k.shape}"
            )

        *prefix_dims, D = q.shape
        # Flatten all prefix dims into a single token axis.
        # If q is 2D, prefix_dims = [N]; if 3D, prefix_dims = [B, T]; etc.
        N = 1
        for d in prefix_dims:
            N *= d

        if N == 0:
            raise ValueError("LukaKVCache._compute_causal_attention_probs: N == 0")

        q_flat = q.reshape(N, D)  # [N, D]
        k_flat = k.reshape(N, D)  # [N, D]

        # [N, N]
        scores = torch.matmul(q_flat, k_flat.transpose(0, 1))
        scores = scores / torch.sqrt(D)

        # Causal mask over the flattened token axis.
        device = q.device
        causal_mask = torch.ones(N, N, device=device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        attn_probs_2d = torch.softmax(scores, dim=-1)  # [N, N]

        # Wrap as [B=1, H=1, T=N, T=N] so existing printing code can iterate
        # over (batch, head) and call the Rule on a [T, T] slice.
        attn_probs = attn_probs_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        return attn_probs

    def _print_boundaries_from_probs(
        self,
        attn_probs: torch.Tensor,  # [B, H, T, T] or [1, 1, N, N] from above
    ) -> None:
        if self.rule is None:
            return

        if attn_probs.ndim != 4:
            raise ValueError(
                f"LukaKVCache._print_boundaries_from_probs expects 4D tensor, "
                f"got shape={attn_probs.shape}"
            )

        B, H, T, _ = attn_probs.shape

        for b in range(B):
            for h in range(H):
                attn_2d = attn_probs[b, h]  # [T, T]
                scores_2d = self.rule.process(attn_2d)  # [T, T]

                # Collapse to a row-wise score; adjust aggregation if you prefer.
                row_scores = scores_2d.mean(dim=-1)  # [T]

                k = min(self.max_print_boundaries, T)
                topk_vals, topk_idx = torch.topk(row_scores, k=k, largest=True)

                # Sort indices just for readability
                sorted_indices, _ = torch.sort(topk_idx)

                print(
                    f"[LukaKVCache] batch={b}, head={h}, "
                    f"boundary_positions={sorted_indices.tolist()}, "
                    f"row_scores={row_scores[sorted_indices].tolist()}"
                )