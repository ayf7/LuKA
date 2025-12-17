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

    Integrates with modeling/qwen/luka_qwen3.py for hierarchical KV cache compression.
    Supports three attention modes:
    - top_down: Traditional LuKA with page summaries and refinement
    - lined: H2O-style heavy-hitter grid on all layers
    - mix: Lined on early/late layers, top-down on middle layers
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        attention_mode: str = "top_down",
        lined_layers: Optional[list] = None,
        luka_config: Optional[Dict] = None
    ):
        """
        Load a LuKA-enabled Qwen model.

        Args:
            model_name: HuggingFace model ID (e.g., "Qwen/Qwen3-1.7B-Base")
            device: Device to load model on ("cuda", "cpu", or "auto")
            attention_mode: One of "top_down", "lined", or "mix"
            lined_layers: Optional list of layer indices for lined attention
            luka_config: LuKA configuration dict with keys:
                - compressor: "mean", "attention_weighted", etc.
                - segmenter: "dummy", "kl", "gaussian"
                - segment_interval: int (for dummy segmenter)
                - refine_threshold: float
        """
        from transformers import AutoTokenizer
        from modeling.qwen.luka_qwen3 import load_luka_model, set_luka_kv_params

        print(f"Loading LuKA-enabled model: {model_name}")
        self.model_name = model_name
        self.device = device
        self.attention_mode = attention_mode
        self.lined_layers = lined_layers
        self.luka_config = luka_config or {}

        # Configure LuKA KV params before loading
        if attention_mode == "baseline":
            # Vanilla attention - no compression
            kv_params = {
                "use_exact_attention": True,
                "create_pages_in_generation": False,
                "compressor": "mean",
                "segmenter": "dummy",
                "production_mode": True,
            }
        else:
            # Build compressor kwargs if using attention_weighted
            compressor = self.luka_config.get("compressor", "attention_weighted")
            compressor_kwargs = {"temperature": 7.0} if compressor == "attention_weighted" else {}

            kv_params = {
                "compressor": compressor,
                "compressor_kwargs": compressor_kwargs,
                "segmenter": self.luka_config.get("segmenter", "dummy"),
                "segment_interval": self.luka_config.get("segment_interval", 16),
                "refinement_rule": self.luka_config.get("refinement_rule", "top_k"),
                "refinement_rule_kwargs": self.luka_config.get("refinement_rule_kwargs", {"k": 3}),
                "log_bias_mode": self.luka_config.get("log_bias_mode", "adaptive_k"),
                "production_mode": True,
            }
        set_luka_kv_params(**kv_params)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LuKA model
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        device_map = "auto" if device == "cuda" else None

        self.model = load_luka_model(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        if device == "cpu":
            self.model.to("cpu")

        self.model.eval()

        # Configure attention mode
        self._configure_attention_mode()

        # Track compression stats
        self.last_compression_stats = None
        self.last_decompressed_segments = None

        print(f"LuKA model loaded with {attention_mode} attention mode")

    def _configure_attention_mode(self):
        """Configure the LuKA controller's attention mode."""
        if self.attention_mode == "baseline":
            # Baseline uses exact attention - no controller configuration needed
            return

        if not (hasattr(self.model, "model") and hasattr(self.model.model, "luka_kv_controller")):
            print("Warning: Model does not have luka_kv_controller")
            return

        controller = self.model.model.luka_kv_controller
        num_layers = controller.num_layers

        if self.attention_mode == "top_down":
            controller.use_lined_attention = False
            controller.lined_layers = set()

        elif self.attention_mode == "lined":
            controller.use_lined_attention = True
            if self.lined_layers is not None:
                controller.lined_layers = set(self.lined_layers)
            else:
                controller.lined_layers = set(range(num_layers))

        elif self.attention_mode == "mix":
            controller.use_lined_attention = True
            if self.lined_layers is not None:
                controller.lined_layers = set(self.lined_layers)
            else:
                # Default mix: first 6 layers + last 5 layers use lined attention
                controller.lined_layers = set(range(0, 6)) | set(range(num_layers - 5, num_layers))

        else:
            raise ValueError(f"Unknown attention mode: {self.attention_mode}")

    def _get_controller(self):
        """Get the LuKA KV controller if available."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "luka_kv_controller"):
            return self.model.model.luka_kv_controller
        return None

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate text with LuKA compression."""
        if config is None:
            config = GenerationConfig()

        # Build prompt with chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context. Answer in a few words or a short phrase. Do not include explanations."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # For base models without chat template, use simple completion format
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt", max_length=4096, truncation=True
        ).to(self.model.device)

        # Generate with LuKA compression active
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else 1.0,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode only new tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Extract compression stats
        self.last_compression_stats = self._extract_compression_stats(inputs['input_ids'].shape[1])

        return generated_text

    def _extract_compression_stats(self, input_len: int = 0) -> Dict[str, Any]:
        """Extract compression statistics from LuKA controller."""
        controller = self._get_controller()
        if controller is None:
            return self._empty_stats()

        stats = {
            'original_tokens': input_len,
            'compressed_tokens': 0,
            'summary_tokens': 0,
            'compression_ratio': 1.0,
            'boundaries': [],
            'num_pages': 0,
        }

        try:
            raw_k, _, seq_start, raw_seq_start = controller.raw_cache.get_layer(0, with_offsets=True)
            if raw_k is not None:
                total_raw = raw_k.shape[2]
                tail_len = total_raw - raw_seq_start[0].item() if raw_seq_start is not None else total_raw
                stats['compressed_tokens'] = tail_len

            summary_cache = controller.summary_cache[0]
            if summary_cache.keys is not None:
                num_pages = summary_cache.page_lens[0].item() if summary_cache.page_lens.numel() > 0 else 0
                stats['summary_tokens'] = num_pages
                stats['num_pages'] = num_pages
                if num_pages > 0:
                    stats['boundaries'] = summary_cache.page_end[0, :num_pages].tolist()

            total_after = stats['compressed_tokens'] + stats['summary_tokens']
            if total_after > 0 and stats['original_tokens'] > 0:
                stats['compression_ratio'] = stats['original_tokens'] / total_after

        except Exception as e:
            print(f"Warning: Failed to extract compression stats: {e}")

        return stats

    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty compression stats."""
        return {
            'original_tokens': 0, 'compressed_tokens': 0, 'summary_tokens': 0,
            'compression_ratio': 1.0, 'boundaries': [], 'num_pages': 0,
        }

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics from last generation."""
        return self.last_compression_stats if self.last_compression_stats else self._empty_stats()

    def get_decompressed_segments(self) -> list:
        """Get which segments were decompressed during refinement."""
        controller = self._get_controller()
        if controller is None:
            return []
        decompressed = []
        try:
            for layer_idx in range(controller.num_layers):
                stats = controller.attn_buffer[layer_idx].get_stats()
                if stats and stats.get('total_refinements_made', 0) > 0:
                    decompressed.append({'layer': layer_idx, 'refinements': stats['total_refinements_made']})
        except Exception:
            pass
        return decompressed

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""
        controller = self._get_controller()
        controller_info = {}
        if controller is not None:
            controller_info = {
                'use_lined_attention': controller.use_lined_attention,
                'lined_layers': sorted(controller.lined_layers) if controller.lined_layers else [],
                'num_layers': controller.num_layers,
            }
        return {
            'name': f"{self.model_name}_luka",
            'type': 'luka_qwen',
            'attention_mode': self.attention_mode,
            'device': str(self.model.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'luka_config': self.luka_config,
            'controller': controller_info,
        }

    def reset(self):
        """Reset LuKA compression state between examples."""
        self.last_compression_stats = None
        self.last_decompressed_segments = None
        controller = self._get_controller()
        if controller is not None:
            try:
                controller.reset()  # Full reset of all cache state
            except Exception:
                pass
