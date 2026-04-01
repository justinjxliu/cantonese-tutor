def max_seq_length(dataset, tokenizer):
    """
    Compute and print the maximum sequence length in a dataset.

    Args:
        dataset: Hugging Face Dataset containing a "messages" list.
        tokenizer: Your tokenizer with apply_chat_template method.

    Returns:
        int: Maximum sequence length.
    """
    tokens_list = [
        tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=False
        )
        for message in dataset["messages"]
    ]
    max_len = max(len(tokens["input_ids"]) for tokens in tokens_list)
    print("Max sequence length:", max_len)
    return max_len

def inspect_first_example(trainer, tokenizer):
    """
    Inspect the first example in the SFT training dataloader.
    Prints input_ids, labels, and decoded completion text.
    """
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    print("=== Input IDs ===")
    print(input_ids)
    print("\n=== Input Decoded ===")
    print(tokenizer.decode(input_ids))

    print("\n=== Labels ===")
    print(labels)
    
    # Decode only completion portion
    completion_labels = [label for label in labels if label != -100]
    print("\n=== Completion Decoded ===")
    print(tokenizer.decode(completion_labels))

def get_class_proportions(dataset):
    """
    Calculate and print the proportion of each category in a dataset.

    The function counts how many examples fall into each category 
    (translate_clean, translate_messy, redirect_clean, redirect_messy)
    and prints their proportion relative to the total dataset size.

    Args:
        dataset: Hugging Face Dataset with a 'category' ClassLabel column.
    """
    from collections import Counter
    counts = Counter(example["category"] for example in dataset)
    total = len(dataset)

    classlabel = dataset.features["category"]
    for label_id in sorted(counts.keys()):
        label_name = classlabel.int2str(label_id)
        print(f"{label_name}: {counts[label_id]/total:.2f}")
