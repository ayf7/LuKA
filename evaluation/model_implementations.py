"""
Example Model Implementations

Provides concrete implementations of ModelInterface for:
1. Local HuggingFace models
2. API-based models (stub)
3. LuKA-enabled models (stub - to be implemented with actual LuKA integration)
"""

import torch
from typing import Optional, Dict, Any
from evaluation.model_interface import (
    ModelInterface,
    BaselineModelInterface,
    LuKAModelInterface,
    GenerationConfig
)


class HuggingFaceModel(BaselineModelInterface):
    """
    Wrapper for local HuggingFace models.
    Works with any AutoModelForCausalLM model.
    """

    def __init__(self, model_name: str, device: str = "auto"):
        """
        Load a HuggingFace model.

        Args:
            model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-3B-Instruct")
            device: Device to load model on ("cuda", "cpu", or "auto")
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device
        )
        self.model.eval()
        print(f"Model loaded on {self.model.device}")

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate text from prompt."""
        if config is None:
            config = GenerationConfig()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the new tokens (exclude prompt)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            'name': self.model_name,
            'type': 'huggingface_baseline',
            'device': str(self.model.device),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }

    def reset(self):
        """Clear any cached state."""
        # For stateless models, this is a no-op
        # KV cache is automatically cleared between generate() calls
        pass


class APIModel(BaselineModelInterface):
    """
    Wrapper for API-based models.
    Supports various API providers (OpenAI format, Anthropic, etc.)
    """

    def __init__(
        self,
        api_type: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize API model.

        Args:
            api_type: Type of API ("openai", "dashscope", "anthropic", etc.)
            model_name: Model identifier for the API
            api_key: API key (or set via environment variable)
            base_url: Optional base URL for custom endpoints
        """
        self.api_type = api_type
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

        # Initialize API client based on type
        if api_type == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif api_type == "dashscope":
            # Alibaba DashScope for Qwen models
            import dashscope
            if api_key:
                dashscope.api_key = api_key
            self.client = dashscope
        else:
            raise ValueError(f"Unsupported API type: {api_type}")

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate text from prompt via API."""
        if config is None:
            config = GenerationConfig()

        if self.api_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p
            )
            return response.choices[0].message.content

        elif self.api_type == "dashscope":
            from dashscope import Generation
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p
            )
            return response.output.text

        else:
            raise NotImplementedError(f"Generation not implemented for {self.api_type}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            'name': self.model_name,
            'type': f'api_{self.api_type}',
            'base_url': self.base_url
        }


class LuKAQwenModel(LuKAModelInterface):
    """
    Wrapper for LuKA-enabled Qwen models.

    This is a STUB implementation. To be completed when integrating with
    actual LuKA compression from modeling/qwen/luka_qwen3.py
    """

    def __init__(self, model_name: str, device: str = "auto", luka_config: Optional[Dict] = None):
        """
        Load a LuKA-enabled Qwen model.

        Args:
            model_name: HuggingFace model ID
            device: Device to load model on
            luka_config: LuKA configuration dict with segmenter params, etc.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # TODO: Import LuKA-specific components
        # from modeling.qwen.luka_qwen3 import set_luka_segmenter, set_luka_kv_params

        print(f"Loading LuKA-enabled model: {model_name}")
        self.model_name = model_name
        self.device = device
        self.luka_config = luka_config or {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device
        )
        self.model.eval()

        # TODO: Configure LuKA compression
        # set_luka_segmenter(...)
        # set_luka_kv_params(...)

        # Track compression stats
        self.last_compression_stats = None
        self.last_decompressed_segments = None

        print(f"LuKA model loaded on {self.model.device}")

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate text with LuKA compression."""
        if config is None:
            config = GenerationConfig()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate with LuKA compression active
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only new tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # TODO: Extract compression stats from LuKA cache
        # self.last_compression_stats = self._extract_compression_stats()
        # self.last_decompressed_segments = self._extract_decompressed_segments()

        return generated_text

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics from last generation."""
        # TODO: Implement actual stats extraction from LuKA cache
        if self.last_compression_stats is None:
            return {
                'original_tokens': 0,
                'compressed_tokens': 0,
                'summary_tokens': 0,
                'compression_ratio': 1.0,
                'boundaries': []
            }
        return self.last_compression_stats

    def get_decompressed_segments(self) -> list:
        """Get which segments were decompressed."""
        # TODO: Implement actual segment tracking
        return self.last_decompressed_segments or []

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        return {
            'name': f"{self.model_name}_luka",
            'type': 'luka_qwen',
            'device': str(self.model.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'luka_config': self.luka_config
        }

    def reset(self):
        """Reset compression state."""
        # TODO: Clear LuKA cache state
        self.last_compression_stats = None
        self.last_decompressed_segments = None
