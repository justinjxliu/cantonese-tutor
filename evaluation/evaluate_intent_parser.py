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
#     max_new_tokens=64,
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
    max_new_tokens=64,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=False
)

invalid_json_base, invalid_json_peft = 0, 0
incorrect_intent_base, incorrect_intent_peft = 0, 0

# Translate specific
incorrect_phrase_base, incorrect_phrase_peft = 0, 0
incorrect_options_base, incorrect_options_peft = 0, 0

# Redirect specific
incorrect_style_base, incorrect_style_peft = 0, 0

total, num_translate_examples, num_redirect_examples = len(test_ds), 0, 0

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

    reference_json = json.loads(assistant_message["content"])
    reference_intent = reference_json["intent"]
    if reference_intent == "translate":
        num_translate_examples += 1
    else:
        num_redirect_examples += 1

    # Check validity of JSON
    try:
        base_json = json.loads(base_output)

        # Check intent classification correct
        base_intent = base_json.get("intent")
        if base_intent != reference_intent:
            print(f"[Example {i}] Base intent is incorrect: {base_intent}")
            incorrect_intent_base += 1

            if reference_intent == "translate":
                incorrect_phrase_base += 1
                incorrect_options_base += 1
            else:
                incorrect_style_base += 1

        else:
            if reference_intent == "translate":
                if base_json.get("phrase") != reference_json["phrase"]:
                    print(f"[Example {i}] Base phrase is incorrect: {base_json.get("phrase")}")

                if (
                    "options" not in base_json 
                    or sorted(base_json.get("options")) != sorted(reference_json["options"])
                ):
                    print(f"[Example {i}] Base options are incorrect: {base_json.get("options")}")
            else:
                if base_json.get("style") != reference_json["style"]:
                    print(f"[Example {i}] Base style is incorrect: {base_json.get("style")}")
    except json.JSONDecodeError:
        print(f"[Example {i}] Base output is invalid: {base_output}")
        invalid_json_base += 1
    
    try:
        peft_json = json.loads(peft_output)

        peft_intent = peft_json.get("intent")
        if peft_intent != reference_intent:
            print(f"[Example {i}] LoRA intent is incorrect: {peft_intent}")
            incorrect_intent_peft += 1

            if reference_intent == "translate":
                incorrect_phrase_peft += 1
                incorrect_options_peft += 1
            else:
                incorrect_style_peft += 1
        else:
            if reference_intent == "translate":
                if peft_json.get("phrase") != reference_json["phrase"]:
                    print(f"[Example {i}] LoRA phrase is incorrect: {peft_json.get("phrase")}")

                if (
                    "options" not in peft_json 
                    or sorted(peft_json.get("options")) != sorted(reference_json["options"])
                ):
                    print(f"[Example {i}] LoRA options are incorrect: {peft_json.get("options")}")
            else:
                if peft_json.get("style") != reference_json["style"]:
                    print(f"[Example {i}] LoRA style is incorrect: {peft_json.get("style")}")
    except json.JSONDecodeError:
        print(f"[Example {i}] LoRA output is invalid: {peft_output}")
        invalid_json_peft += 1

print(f"Base model JSON validity: {100*(1 - invalid_json_base/total):.2f}%")
print(f"LoRA model JSON validity: {100*(1 - invalid_json_peft/total):.2f}%")

print(f"Base model intent accuracy: {100*(1 - incorrect_intent_base/total):2f}%")
print(f"LoRA model intent accuracy: {100*(1 - incorrect_intent_peft/total):2f}%")

# Translate specific
print(f"Base model phrase accuracy: {100*(1 - incorrect_phrase_base/num_translate_examples):2f}%")
print(f"LoRA model phrase accuracy: {100*(1 - incorrect_phrase_peft/num_translate_examples):2f}%")
print(f"Base model options accuracy: {100*(1 - incorrect_options_base/num_translate_examples):2f}%")
print(f"LoRA model options accuracy: {100*(1 - incorrect_options_peft/num_translate_examples):2f}%")

# Redirect specific
print(f"Base model style accuracy: {100*(1 - incorrect_style_base/num_redirect_examples):2f}%")
print(f"LoRA model style accuracy: {100*(1 - incorrect_style_peft/num_redirect_examples):2f}%")
