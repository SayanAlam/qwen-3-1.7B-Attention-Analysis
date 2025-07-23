import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Qwen3-1.7B model and tokenizer (assumes already downloaded)
local_model_path = "/path/to/Qwen3-1.7B"  # Update if needed

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)
model.eval().to("cuda")  # Use GPU

# Define prompt sets
instruction_prompts = [
    "Translate this sentence to Spanish: 'He is reading a book.'",
    "Summarize the following paragraph in one sentence.",
    "Paraphrase this question: 'What are the benefits of exercise?'",
    "Generate a title for this article about climate change.",
    "Rewrite this sentence to make it more formal."
]

reasoning_prompts = [
    "If Alice is older than Bob and Bob is older than Carol, who is the oldest?",
    "Despite the warning, he went out. What might be the consequence?",
    "If a train leaves at 2 PM and travels at 60 km/h, how far will it go in 3 hours?",
    "Why is the sky blue during the day but red during sunset?",
    "If x is greater than y and y is greater than z, is x greater than z?"
]

def compute_attention_focus(prompt_list):
    all_focus = []

    for prompt in prompt_list:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ids = inputs.input_ids
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        attentions = outputs.attentions  # List of (batch, heads, seq_len, seq_len)

        # Compute average attention across heads and tokens
        focus_by_layer = []
        for layer_attn in attentions:
            avg_attn = layer_attn[0].mean(dim=0)  # (seq_len, seq_len)
            focus = avg_attn.sum(dim=1).mean().item() / avg_attn.size(0)
            focus_by_layer.append(focus)
        all_focus.append(focus_by_layer)

    # Average across prompts
    avg_focus = torch.tensor(all_focus).mean(dim=0).tolist()
    return avg_focus

instruction_focus = compute_attention_focus(instruction_prompts)
reasoning_focus = compute_attention_focus(reasoning_prompts)

# Display results
print("\nAttention Focus per Layer (Averaged Over Prompts)")
print("--------------------------------------------------")
print("Layer | Instruction Focus | Reasoning Focus")
print("--------------------------------------------------")
for i, (inst, reas) in enumerate(zip(instruction_focus, reasoning_focus)):
    print(f"{i:5} |       {inst:.6f}     |     {reas:.6f}")
