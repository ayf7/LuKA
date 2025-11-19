from vllm import LLM, SamplingParams
from modeling.qwen.luka_qwen3 import LukaQwenForCausalLM, initialize_luka_hook

initialize_luka_hook() # any Qwen3Model we use will now use our backend.

llm = LLM(
    model="Qwen/Qwen3-1.7B-Base",
    max_model_len=4096,
    enforce_eager=True
)


sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128,
    n=1,          # number of completions per prompt
    seed=42,      # deterministic sampling for debugging
)


prompts = [
    "You are LuKA, a model with a custom attention pattern. "
    "In 2 sentences, describe what makes your attention different.",
]


outputs = llm.generate(prompts, sampling_params=sampling_params)


for i, request_output in enumerate(outputs):
    print("=" * 80)
    print(f"Prompt {i}: {prompts[i]}")
    for j, candidate in enumerate(request_output.outputs):
        print(f"\n[Candidate {j}]")
        print(candidate.text)