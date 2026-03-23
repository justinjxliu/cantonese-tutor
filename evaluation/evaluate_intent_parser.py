from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import yaml
from pathlib import Path

base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
base_model.to("mps")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Absolute path to root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Prompt path
prompt_file = PROJECT_ROOT / "prompts/intent_parser.yaml"

with open(prompt_file) as f:
    SYSTEM_PROMPT = yaml.safe_load(f)["system_prompt"]
prompts = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": "Could you suggest a cool song?"
    }
]
tokens = tokenizer.apply_chat_template(
    prompts,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

# Put model and inputs on the same device
tokens = {k: v.to("mps") for k, v in tokens.items()}

generation_config = GenerationConfig(
    max_new_tokens=256
)
def generate_and_print(model, tokens, tokenizer, description):
    output = model.generate(**tokens, generation_config=generation_config)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n--- {description} --- \n{text}\n")

# Baseline performance
generate_and_print(base_model, tokens, tokenizer, "Base Model")

peft_model = PeftModel.from_pretrained(base_model, "../models/llama_intent_parser")
peft_model.to("mps")

# Finetuned performance
generate_and_print(peft_model, tokens, tokenizer, "LoRA-Finetuned Model")
