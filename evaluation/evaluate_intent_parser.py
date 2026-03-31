from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import json
import yaml
from pathlib import Path

# Model identifiers
base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
adapter_path = "../models/llama_intent_parser"

# Load models and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="mps")
base_model_for_peft = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="mps")
peft_model = PeftModel.from_pretrained(base_model_for_peft, adapter_path, device_map="mps")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Absolute path to root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Prompt path
prompt_file = PROJECT_ROOT / "prompts/intent_parser.yaml"
with open(prompt_file) as f:
    SYSTEM_PROMPT = yaml.safe_load(f)["system_prompt"]


# # ============================================================
# #                        MANUAL EVALUATION
# # ============================================================

# prompts = [
#     {
#         "role": "system",
#         "content": SYSTEM_PROMPT
#     },
#     {
#         "role": "user",
#         "content": "Give me the Cantonese for 'that's awesome' with a vocabulary list and grammar explanation."
#     }
# ]
# tokens = tokenizer.apply_chat_template(
#     prompts,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_tensors="pt"
# )
# tokens = {k: v.to("mps") for k, v in tokens.items()} # Put model and inputs on the same device

# generation_config = GenerationConfig(
#     max_new_tokens=128,
#     pad_token_id=tokenizer.eos_token_id
# )
# def generate_and_print(model, tokens, tokenizer, description):
#     prompt_length = tokens["input_ids"].shape[1]
#     output = model.generate(**tokens, generation_config=generation_config)
#     text = tokenizer.decode(output[0][prompt_length], skip_special_tokens=True)
#     print(f"\n--- {description} --- \n{text}\n")

# # Baseline performance
# generate_and_print(base_model, tokens, tokenizer, "Base Model")

# # Finetuned performance
# generate_and_print(peft_model, tokens, tokenizer, "LoRA-Finetuned Model")


# ============================================================
#                       AUTOMATED EVALUATION
# ============================================================

# Test path
test_file = PROJECT_ROOT / "data/processed/test_examples.jsonl"
test_ds = load_dataset("json", data_files=str(test_file), split="train")

generation_config = GenerationConfig(
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False
)

invalid_base, invalid_peft = 0, 0
# snippet_len = 100

for i, example in enumerate(test_ds):
    system_message, user_message, assistant_message = example["messages"]
    prompts = [system_message, user_message]

    tokens = tokenizer.apply_chat_template(
        prompts,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    tokens = {k: v.to("mps") for k, v in tokens.items()}
    prompt_length = tokens["input_ids"].shape[1]

    # Generate base model output
    base_generation = base_model.generate(**tokens, generation_config=generation_config)
    base_output = tokenizer.decode(base_generation[0][prompt_length:], skip_special_tokens=True)

    # Generate LoRA finetuned output
    peft_generation = peft_model.generate(**tokens, generation_config=generation_config)
    peft_output = tokenizer.decode(peft_generation[0][prompt_length:], skip_special_tokens=True)

    # Check validity of JSON
    try:
        json.loads(base_output)
    except json.JSONDecodeError:
        print(f"[Example {i}] Base output is invalid: {base_output}")
        invalid_base += 1
    
    try:
        json.loads(peft_output)
    except json.JSONDecodeError:
        print(f"[Example {i}] LoRA output is invalid: {peft_output}")
        invalid_peft += 1

print(f"Base model invalid outputs: {invalid_base}/{len(test_ds)}")
print(f"LoRA model invalid outputs: {invalid_peft}/{len(test_ds)}")
