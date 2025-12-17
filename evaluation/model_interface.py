"""
Model Interface Abstraction for LuKA Evaluation Pipeline

Provides a clean interface for swapping between different model backends:
- Local HuggingFace models
- API endpoints
- Pre-loaded Qwen-LuKA models
- Any other generation backend
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 0.0  # Deterministic by default for evaluation
    top_p: float = 1.0
    do_sample: bool = False
    stop_strings: Optional[list] = None


class ModelInterface(ABC):
    """Abstract interface that all model implementations must follow."""

    @abstractmethod
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """
        Generate text given a prompt.

        Args:
            prompt: Input text prompt
            config: Generation configuration (uses defaults if None)

        Returns:
            Generated text (without the prompt)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return information about the model.

        Returns:
            Dict with keys like 'name', 'type', 'parameters', etc.
        """
        pass

    def reset(self):
        """
        Optional: Reset model state (e.g., clear KV cache).
        Only needed for stateful models.
        """
        pass

    def batch_generate(
        self, prompts: List[str], config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts in a batch.
        Default implementation falls back to sequential generation.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of generated texts
        """
        return [self.generate(p, config) for p in prompts]


class BaselineModelInterface(ModelInterface):
    """Interface specifically for baseline (non-LuKA) models."""
    pass


class LuKAModelInterface(ModelInterface):
    """
    Interface for LuKA-enabled models.
    Extends base interface with compression metrics.
    """

    @abstractmethod
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics from the last generation.

        Returns:
            Dict with keys like:
            - 'original_tokens': Number of tokens in original KV cache
            - 'compressed_tokens': Number of tokens after compression
            - 'summary_tokens': Number of summary page tokens
            - 'compression_ratio': Ratio of compression
            - 'boundaries': List of detected page boundaries
        """
        pass

    @abstractmethod
    def get_decompressed_segments(self) -> list:
        """
        Get which segments were decompressed during last generation.

        Returns:
            List of segment IDs that were decompressed
        """
        pass
