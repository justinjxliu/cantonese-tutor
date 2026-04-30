from pathlib import Path
import json
from pydantic import ValidationError
import sys
from models import Example

translate_examples_path = Path("../data/processed/translate_examples.jsonl")

valid = invalid = 0

with open(translate_examples_path, 'r') as translate_examples_file:
    for i, line in enumerate(translate_examples_file, start=1):
        translate_example = json.loads(line)
        
        try:
            Example.model_validate(translate_example)
            valid += 1
        except ValidationError as e:
            invalid += 1
            print(f"\nValidation error on line {i}:")
            print(f"Error: {e}")
            print(f"Example: {translate_example}")

print(f"\nDone: {valid} valid, {invalid} invalid")

if invalid > 0:
    sys.exit(1)
