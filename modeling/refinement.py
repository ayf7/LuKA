"""
Refinement rules for LuKA top-down attention.

These rules determine which summary tokens should be refined (expanded to raw tokens)
during attention computation. Different rules provide different trade-offs between
accuracy and computational cost.
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple

import torch


class RefinementRule(ABC):
    """Base class for refinement selection rules.

    A refinement rule decides which summary tokens need to be "refined" by
    re-computing attention over their underlying raw tokens. This is the
    core mechanism for trading off compression ratio vs accuracy in LuKA.

    Args:
        always_refine_first_n: Always refine the first N pages regardless of
            attention patterns. Similar to StreamingLLM attention sinks.
            Set to 0 to disable (default).
    """

    def __init__(self, always_refine_first_n: int = 0):
        self.always_refine_first_n = always_refine_first_n

    @abstractmethod
    def _select_impl(self, attn_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subclass implementation of selection logic.

        Args:
            attn_probs: [N_sum, H, L] attention probabilities to summary positions.

        Returns:
            summary_mask: [N_sum] boolean mask of which summaries to refine
            detail_mask: [N_sum, H, L] boolean mask for per-(summary, head, query)
        """
        pass

    def select(
        self,
        attn_probs: torch.Tensor,
        page_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select which summaries to refine based on attention probabilities.

        Args:
            attn_probs: [N_sum, H, L] attention probabilities to summary positions,
                where N_sum is the number of summary tokens, H is the number of
                attention heads, and L is the number of query positions.
            page_ids: [N_sum] page index for each summary position. Required if
                always_refine_first_n > 0.

        Returns:
            summary_mask: [N_sum] boolean mask of which summaries to refine
            detail_mask: [N_sum, H, L] boolean mask for per-(summary, head, query)
                refinement. This allows fine-grained control over which head/query
                combinations get refined for each summary.
        """
        summary_mask, detail_mask = self._select_impl(attn_probs)

        # Apply "always refine first N pages" (attention sinks)
        if self.always_refine_first_n > 0 and page_ids is not None:
            first_n_mask = page_ids < self.always_refine_first_n
            summary_mask = summary_mask | first_n_mask
            detail_mask = detail_mask | first_n_mask.view(-1, 1, 1).expand_as(detail_mask)

        return summary_mask, detail_mask

    def __repr__(self):
        base = f"{self.__class__.__name__}()"
        if self.always_refine_first_n > 0:
            base = base[:-1] + f", always_refine_first_n={self.always_refine_first_n})"
        return base


class NoRefinementRule(RefinementRule):
    """Disable refinement entirely - use summary attention as-is.

    This is the fastest option but may lose accuracy for summaries
    that receive high attention.

    Note: If always_refine_first_n > 0, those pages will still be refined.
    """

    def __init__(self, always_refine_first_n: int = 0):
        super().__init__(always_refine_first_n)

    def _select_impl(self, attn_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N_sum = attn_probs.shape[0]
        device = attn_probs.device
        return (
            torch.zeros(N_sum, dtype=torch.bool, device=device),
            torch.zeros_like(attn_probs, dtype=torch.bool)
        )


class FixedThresholdRule(RefinementRule):
    """Refine summaries where attention exceeds a fixed threshold.

    This is the original LuKA refinement strategy. A summary is refined
    if ANY (head, query) pair has attention probability > threshold.
    Only those specific (head, query) pairs are refined.

    Args:
        threshold: Attention probability threshold (0.0 to 1.0).
            Lower values = more refinement = slower but more accurate.
            Higher values = less refinement = faster but may lose accuracy.
        always_refine_first_n: Always refine first N pages (attention sinks).
    """

    def __init__(self, threshold: float = 0.2, always_refine_first_n: int = 0):
        super().__init__(always_refine_first_n)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def _select_impl(self, attn_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-(summary, head, query) mask
        detail_mask = attn_probs > self.threshold
        # Summary is refined if any head/query exceeds threshold
        summary_mask = detail_mask.any(dim=(1, 2))
        return summary_mask, detail_mask

    def __repr__(self):
        r = f"FixedThresholdRule(threshold={self.threshold}"
        if self.always_refine_first_n > 0:
            r += f", always_refine_first_n={self.always_refine_first_n}"
        return r + ")"


class TopKRule(RefinementRule):
    """Refine top K summaries by maximum attention.

    Selects the K summaries with highest attention (across all heads/queries)
    and refines all heads/queries for those summaries.

    Args:
        k: Maximum number of summaries to refine per forward pass.
        always_refine_first_n: Always refine first N pages (attention sinks).
    """

    def __init__(self, k: int = 3, always_refine_first_n: int = 0):
        super().__init__(always_refine_first_n)
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self.k = k

    def _select_impl(self, attn_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N_sum = attn_probs.shape[0]
        device = attn_probs.device

        if N_sum == 0 or self.k == 0:
            return (
                torch.zeros(N_sum, dtype=torch.bool, device=device),
                torch.zeros_like(attn_probs, dtype=torch.bool)
            )

        # Max attention across heads and queries for each summary
        max_attn = attn_probs.amax(dim=(1, 2))  # [N_sum]

        k = min(self.k, N_sum)
        _, top_indices = max_attn.topk(k)

        summary_mask = torch.zeros(N_sum, dtype=torch.bool, device=device)
        summary_mask[top_indices] = True

        # Refine all heads/queries for selected summaries
        detail_mask = summary_mask.view(N_sum, 1, 1).expand_as(attn_probs).clone()

        return summary_mask, detail_mask

    def __repr__(self):
        r = f"TopKRule(k={self.k}"
        if self.always_refine_first_n > 0:
            r += f", always_refine_first_n={self.always_refine_first_n}"
        return r + ")"


class TopPRule(RefinementRule):
    """Refine summaries until cumulative attention reaches P.

    Sorts summaries by total attention and includes summaries until
    their cumulative attention exceeds P fraction of total attention
    to summaries.

    Args:
        p: Cumulative attention threshold (0.0 to 1.0).
            E.g., p=0.9 means refine summaries that account for 90%
            of total attention to summaries.
        always_refine_first_n: Always refine first N pages (attention sinks).
    """

    def __init__(self, p: float = 0.9, always_refine_first_n: int = 0):
        super().__init__(always_refine_first_n)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = p

    def _select_impl(self, attn_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N_sum = attn_probs.shape[0]
        device = attn_probs.device

        if N_sum == 0:
            return (
                torch.zeros(N_sum, dtype=torch.bool, device=device),
                torch.zeros_like(attn_probs, dtype=torch.bool)
            )

        # Sum attention across heads and queries
        sum_attn = attn_probs.sum(dim=(1, 2))  # [N_sum]
        total = sum_attn.sum()

        if total == 0:
            return (
                torch.zeros(N_sum, dtype=torch.bool, device=device),
                torch.zeros_like(attn_probs, dtype=torch.bool)
            )

        # Sort by descending attention
        sorted_attn, sorted_indices = sum_attn.sort(descending=True)
        cumsum = sorted_attn.cumsum(dim=0)

        # Find cutoff: include summaries until cumulative > p * total
        # We want the smallest set that captures at least p of the attention
        cutoff_mask = cumsum <= self.p * total
        num_selected = cutoff_mask.sum() + 1  # Include first one that exceeds
        num_selected = min(num_selected.item(), N_sum)

        summary_mask = torch.zeros(N_sum, dtype=torch.bool, device=device)
        summary_mask[sorted_indices[:num_selected]] = True

        detail_mask = summary_mask.view(N_sum, 1, 1).expand_as(attn_probs).clone()

        return summary_mask, detail_mask

    def __repr__(self):
        r = f"TopPRule(p={self.p}"
        if self.always_refine_first_n > 0:
            r += f", always_refine_first_n={self.always_refine_first_n}"
        return r + ")"


class TopFracRule(RefinementRule):
    """Refine a fixed fraction of all summaries.

    Selects the top ceil(frac * N_summaries) summaries by attention score.
    Unlike TopPRule which uses cumulative attention mass, this uses page count.

    Args:
        frac: Fraction of summaries to refine (0.0 to 1.0).
            E.g., frac=0.5 means refine the top 50% of pages by attention.
        always_refine_first_n: Always refine first N pages (attention sinks).
    """

    def __init__(self, frac: float = 0.5, always_refine_first_n: int = 0):
        super().__init__(always_refine_first_n)
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"frac must be in [0, 1], got {frac}")
        self.frac = frac

    def _select_impl(self, attn_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N_sum = attn_probs.shape[0]
        device = attn_probs.device

        if N_sum == 0:
            return (
                torch.zeros(N_sum, dtype=torch.bool, device=device),
                torch.zeros_like(attn_probs, dtype=torch.bool)
            )

        # Number of pages to select: ceil(frac * N_sum)
        k = math.ceil(self.frac * N_sum)
        k = max(0, min(k, N_sum))

        if k == 0:
            return (
                torch.zeros(N_sum, dtype=torch.bool, device=device),
                torch.zeros_like(attn_probs, dtype=torch.bool)
            )

        # Sort by total attention (sum across heads and queries)
        sum_attn = attn_probs.sum(dim=(1, 2))  # [N_sum]
        _, top_indices = sum_attn.topk(k)

        summary_mask = torch.zeros(N_sum, dtype=torch.bool, device=device)
        summary_mask[top_indices] = True

        detail_mask = summary_mask.view(N_sum, 1, 1).expand_as(attn_probs).clone()

        return summary_mask, detail_mask

    def __repr__(self):
        r = f"TopFracRule(frac={self.frac}"
        if self.always_refine_first_n > 0:
            r += f", always_refine_first_n={self.always_refine_first_n}"
        return r + ")"


class TopKTopPRule(RefinementRule):
    """Refine at most K summaries, constrained by cumulative attention P.

    Combines TopK and TopP: selects summaries by descending attention
    until either K summaries are selected OR cumulative attention exceeds P,
    whichever comes first.

    This provides a budget-constrained refinement strategy that adapts to
    the attention distribution.

    Args:
        k: Maximum number of summaries to refine.
        p: Cumulative attention threshold.
        always_refine_first_n: Always refine first N pages (attention sinks).
    """

    def __init__(self, k: int = 3, p: float = 0.9, always_refine_first_n: int = 0):
        super().__init__(always_refine_first_n)
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.k = k
        self.p = p

    def _select_impl(self, attn_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N_sum = attn_probs.shape[0]
        device = attn_probs.device

        if N_sum == 0 or self.k == 0:
            return (
                torch.zeros(N_sum, dtype=torch.bool, device=device),
                torch.zeros_like(attn_probs, dtype=torch.bool)
            )

        # Sum attention across heads and queries
        sum_attn = attn_probs.sum(dim=(1, 2))  # [N_sum]
        total = sum_attn.sum()

        if total == 0:
            return (
                torch.zeros(N_sum, dtype=torch.bool, device=device),
                torch.zeros_like(attn_probs, dtype=torch.bool)
            )

        # Sort by descending attention
        sorted_attn, sorted_indices = sum_attn.sort(descending=True)
        cumsum = sorted_attn.cumsum(dim=0)

        # Top P cutoff
        p_cutoff_mask = cumsum <= self.p * total
        num_p = (p_cutoff_mask.sum() + 1).item()

        # Take minimum of K and P selections
        num_selected = min(self.k, num_p, N_sum)

        summary_mask = torch.zeros(N_sum, dtype=torch.bool, device=device)
        summary_mask[sorted_indices[:num_selected]] = True

        detail_mask = summary_mask.view(N_sum, 1, 1).expand_as(attn_probs).clone()

        return summary_mask, detail_mask

    def __repr__(self):
        r = f"TopKTopPRule(k={self.k}, p={self.p}"
        if self.always_refine_first_n > 0:
            r += f", always_refine_first_n={self.always_refine_first_n}"
        return r + ")"


# Registry for string-based instantiation
REFINEMENT_RULES = {
    "none": NoRefinementRule,
    "threshold": FixedThresholdRule,
    "top_k": TopKRule,
    "top_p": TopPRule,
    "top_frac": TopFracRule,
    "top_k_top_p": TopKTopPRule,
}


def get_refinement_rule(rule) -> RefinementRule:
    """Get a refinement rule instance from various input types.

    Args:
        rule: Can be:
            - RefinementRule instance: returned as-is
            - str: looked up in REFINEMENT_RULES registry (uses defaults)
            - float: interpreted as threshold for FixedThresholdRule
            - None: returns NoRefinementRule

    Returns:
        RefinementRule instance
    """
    if rule is None:
        return NoRefinementRule()

    if isinstance(rule, RefinementRule):
        return rule

    if isinstance(rule, (int, float)):
        if rule < 0:
            return NoRefinementRule()
        return FixedThresholdRule(threshold=float(rule))

    if isinstance(rule, str):
        rule_lower = rule.lower()
        if rule_lower not in REFINEMENT_RULES:
            raise ValueError(
                f"Unknown refinement rule: {rule}. "
                f"Available: {list(REFINEMENT_RULES.keys())}"
            )
        return REFINEMENT_RULES[rule_lower]()

    raise TypeError(
        f"Cannot create RefinementRule from {type(rule).__name__}. "
        f"Expected RefinementRule, str, float, or None."
    )
