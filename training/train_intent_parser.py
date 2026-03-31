from datasets import concatenate_datasets, load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import torch
import yaml
import json
from pathlib import Path

base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="mps")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.chat_template = """
{{- bos_token }}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{{- system_message }}
{{- "<|eot_id|>" }}

{%- for message in messages %}
{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' -}}
{% if message['role'] == 'assistant' %}
{% generation %}{{ message['content'] | trim }}{% endgeneration %}{{ '<|eot_id|>' -}}
{% else %}
{{ message['content'] | trim }}{{ '<|eot_id|>' -}}
{% endif %}
{% endfor %}
{%- if add_generation_prompt %}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""

# Absolute path to root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
translate_file = PROJECT_ROOT / "data/processed/translate_examples.jsonl"
redirect_file = PROJECT_ROOT / "data/processed/redirect_examples.jsonl"

# Prompt path
prompt_file = PROJECT_ROOT / "prompts/intent_parser.yaml"

translate_ds = load_dataset("json", data_files=str(translate_file), split="train")
redirect_ds = load_dataset("json", data_files=str(redirect_file), split="train")
combined_ds = concatenate_datasets([translate_ds, redirect_ds])

with open(prompt_file) as f:
    SYSTEM_PROMPT = yaml.safe_load(f)["system_prompt"]
def format_example(batch):
    """
    Convert batch prompts/completions into structured messages.
    Returns: {"messages": [[{role, content}, ...], ...]}
    """
    messages = []
    for prompt, completion in zip(batch["prompt"], batch["completion"]):
        messages.append([
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": completion
            }
        ])
    return {"messages": messages}
formatted_ds = combined_ds.map(format_example, batched=True, remove_columns=["prompt", "completion"])

# Split data
train_eval_split = formatted_ds.train_test_split(test_size=0.3, shuffle=True, seed=42)
train_dataset = train_eval_split["train"]
eval_test_split = train_eval_split["test"].train_test_split(test_size=0.5, shuffle=True, seed=42)
eval_dataset = eval_test_split["train"]
test_dataset = eval_test_split["test"]

args = SFTConfig(
    warmup_steps=0.1,
    learning_rate=2e-5,
    max_length=512,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    logging_steps=5,
    eval_strategy="steps",
    save_steps=10,
    save_total_limit=2,
    output_dir="../outputs/llama_intent_parser",
    assistant_only_loss=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM"
)

trainer = SFTTrainer(
    model=base_model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=args,
    peft_config=lora_config
)
trainer.train()

# Save model
model_path = Path("../models/llama_intent_parser")
model_path.mkdir(parents=True, exist_ok=True)
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# Save test data set
test_path = Path("../data/processed/test_examples.jsonl")
test_path.parent.mkdir(parents=True, exist_ok=True)
with open(test_path, 'w', encoding="utf-8") as test_file:
    for example in test_dataset:
        json_line = json.dumps({"messages": example["messages"]}, ensure_ascii=False)
        test_file.write(json_line + '\n')
