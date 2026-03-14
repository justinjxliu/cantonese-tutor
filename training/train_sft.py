from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
import torch
from pathlib import Path

base_model_name = "Qwen/Qwen3-1.7B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

ds = load_dataset(
    "json",
    data_files="../data/processed/sft_examples.jsonl",
    split="train"
)

ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
def format_example(batch):
    input_ids = []
    labels = []
    for prompt, completion in zip(batch["prompt"], batch["completion"]):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False
        )
        completion_tokens = tokenizer(completion + tokenizer.eos_token)
        input_ids.append(tokens["input_ids"])
        labels.append([-100]*(len(tokens["input_ids"]) - len(completion_tokens["input_ids"])) + completion_tokens["input_ids"])
    return {
        "input_ids": input_ids,
        "labels": labels
    }
ds_formatted = ds.map(format_example, batched=True, remove_columns=["prompt", "completion"])

args = SFTConfig(
    warmup_ratio=0.1,
    learning_rate=2e-5,
    max_length=512,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    output_dir="../outputs/qwen_cantonese_sft"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.1,
    bias="none"
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=ds_formatted["train"],
    eval_dataset=ds_formatted["test"],
    args=args,
    peft_config=lora_config
)
trainer.train()

model_path = Path("../models/qwen_cantonese_lora")
model_path.mkdir(parents=True, exist_ok=True)
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
