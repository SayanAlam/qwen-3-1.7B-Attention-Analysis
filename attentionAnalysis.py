import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Set local model path (update this to your path)
local_model_path = "/path/to/Qwen3-1.7B"  # Replace with your actual path

# Automatically detect GPU or fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    attn_implementation="eager"
)
model.eval().to(device)

# List of prompts with both instruction and reasoning elements
prompts = [
    "Translate the following sentence into French: 'If Tom is taller than Jerry and Jerry is taller than Sam, who is the tallest?'",
    "Summarize this: 'Although Alice wanted to go, she stayed home because she was sick. What does this suggest about her priorities?'",
    "Paraphrase: 'Despite the rain, the team continued their match with enthusiasm. Who showed more resilience?'",
    "Generate a question: 'John didn't bring an umbrella, yet he didn't get wet. Why might that be?'",
    "Rewrite this sentence formally: 'Hey, what's up with the late delivery?' What might be a professional reason for the delay?"
]

# Configuration for token classification
class TokenClassifier:
    def __init__(self):
        # Expanded keyword sets
        self.instruction_keywords = [
            "translate", "summarize", "paraphrase", "generate", "rewrite",
            "write", "explain", "answer", "respond", "convert", "rephrase",
            "analyze", "compare", "evaluate", "describe", "list", "define",
            "calculate", "solve", "predict", "recommend", "outline", "create"
        ]
        
        self.reasoning_keywords = [
            "if", "then", "why", "who", "what", "how", "because", "although", "despite",
            "therefore", "however", "but", "so", "since", "unless", "implies",
            "more", "less", "greater", "smaller", "before", "after", "cause", "effect",
            "consequently", "furthermore", "meanwhile", "nevertheless", "whereas",
            "while", "thus", "hence", "given that", "assuming", "considering"
        ]
        
        # Compile regex patterns for faster matching
        self.instruction_pattern = self._compile_pattern(self.instruction_keywords)
        self.reasoning_pattern = self._compile_pattern(self.reasoning_keywords)
        
    def _compile_pattern(self, keywords):
        # Create regex pattern that matches whole words only
        pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def classify_tokens(self, tokens, prompt):
        """Classify tokens using both keyword matching and positional context"""
        instruction_indices = []
        reasoning_indices = []
        
        # Convert tokens to lowercase for case-insensitive matching
        tokens_lower = [t.lower() for t in tokens]
        prompt_lower = prompt.lower()
        
        # First pass: direct keyword matching
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            # Check if token contains any instruction keywords
            if any(kw in token_lower for kw in self.instruction_keywords):
                instruction_indices.append(i)
                
            # Check if token contains any reasoning keywords
            if any(kw in token_lower for kw in self.reasoning_keywords):
                reasoning_indices.append(i)
        
        # Second pass: context-based classification using the original prompt
        # Find instruction segments (typically at the beginning of prompts)
        instruction_matches = list(self.instruction_pattern.finditer(prompt_lower))
        reasoning_matches = list(self.reasoning_pattern.finditer(prompt_lower))
        
        # Add tokens that appear in the context of instruction/reasoning phrases
        for i, token in enumerate(tokens):
            # Skip tokens already classified
            if i in instruction_indices or i in reasoning_indices:
                continue
                
            token_text = token.replace("▁", " ").strip().lower()
            if not token_text:
                continue
                
            # Check if token is part of a larger instruction phrase
            for match in instruction_matches:
                if token_text in prompt_lower[match.start():match.end()+10]:
                    instruction_indices.append(i)
                    break
                    
            # Check if token is part of a reasoning phrase
            for match in reasoning_matches:
                if token_text in prompt_lower[match.start()-5:match.end()+5]:
                    reasoning_indices.append(i)
                    break
        
        return instruction_indices, reasoning_indices

# Print token matches for each prompt (set to False to disable)
verbose = True

# Function to compute attention focus per layer
def compute_focus(attentions, indices):
    focus_by_layer = []
    for layer_attn in attentions:
        avg_attn = layer_attn[0].mean(dim=0)
        if indices:
            focus = avg_attn[:, indices].sum().item() / (avg_attn.shape[0] * len(indices))
        else:
            focus = 0.0
        focus_by_layer.append(focus)
    return focus_by_layer

# Initialize token classifier
token_classifier = TokenClassifier()

# Accumulate focus values across prompts
num_layers = None
instruction_focus_sum = []
reasoning_focus_sum = []
counted_prompts = 0

for idx, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Use the token classifier to identify instruction and reasoning tokens
    instruction_indices, reasoning_indices = token_classifier.classify_tokens(tokens, prompt)

    if not instruction_indices and not reasoning_indices:
        continue

    if verbose:
        print(f"\nPrompt {idx+1}: {prompt}")
        print(f"Instruction tokens: {[tokens[i] for i in instruction_indices]}")
        print(f"Reasoning tokens:   {[tokens[i] for i in reasoning_indices]}")

    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    attentions = outputs.attentions

    inst_focus = compute_focus(attentions, instruction_indices)
    reas_focus = compute_focus(attentions, reasoning_indices)

    if num_layers is None:
        num_layers = len(inst_focus)
        instruction_focus_sum = [0.0] * num_layers
        reasoning_focus_sum = [0.0] * num_layers

    for i in range(num_layers):
        instruction_focus_sum[i] += inst_focus[i]
        reasoning_focus_sum[i] += reas_focus[i]

    counted_prompts += 1

if counted_prompts == 0:
    print("No valid prompts with matched tokens.")
else:
    instruction_focus_avg = [x / counted_prompts for x in instruction_focus_sum]
    reasoning_focus_avg = [x / counted_prompts for x in reasoning_focus_sum]

    print("\nAverage Attention Focus by Layer")
    print("Layer | Instruction Focus | Reasoning Focus")
    print("--------------------------------------------")
    for i, (inst, reas) in enumerate(zip(instruction_focus_avg, reasoning_focus_avg)):
        print(f"{i:5} |       {inst:.6f}     |     {reas:.6f}")