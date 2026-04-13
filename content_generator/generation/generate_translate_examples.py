from pathlib import Path
import json
import random

def expand_vocabulary(lesson_primitive, vocab_primitives):
    return [
        {
            "chinese": vocab_primitives[component]["chinese"],
            "definition": vocab_primitives[component]["definition"]
        }
        for component in lesson_primitive["vocabulary"]
    ]

def expand_pronunciation(lesson_primitive, vocab_primitives):
    return [
        vocab_primitives[component]["pronunciation"]
        for component in lesson_primitive["pronunciation"]
    ]

def expand_practice(lesson_primitive):
    return [
        {
            "phrase": sentence["phrase"],
            "characters": sentence["characters"]
        }
        for sentence in random.sample(lesson_primitive["practice"], k=3)
    ]

def expand_lesson(lesson_primitive, vocab_primitives):
    expanded_lesson = dict(lesson_primitive)
    expanded_lesson["vocabulary"] = expand_vocabulary(lesson_primitive, vocab_primitives)
    expanded_lesson["pronunciation"] = expand_pronunciation(lesson_primitive, vocab_primitives)
    return expanded_lesson

raw_data_path = Path("../data/raw")

translate_primitives_path = raw_data_path / "translate_primitives.json"
lesson_primitives_path = raw_data_path / "lesson_primitives.json"
vocab_primitives_path = raw_data_path / "vocab_primitives.json"

with open(translate_primitives_path, 'r') as translate_primitives_file:
    translate_primitives = json.load(translate_primitives_file)
with open(lesson_primitives_path, 'r') as lesson_primitives_file:
    lesson_primitives = json.load(lesson_primitives_file)
with open(vocab_primitives_path, 'r') as vocab_primitives_file:
    vocab_primitives = json.load(vocab_primitives_file)

translate_examples = []

# Generate examples
for task in translate_primitives.keys():
    primitives_for_translate = translate_primitives[task]
    
    requests = primitives_for_translate["requests"]
    for lesson_primitive in lesson_primitives:
        expanded_lesson = expand_lesson(lesson_primitive, vocab_primitives)

        for combination in requests.keys():
            request_template = requests[combination]
            request = {
                "intent": request_template["intent"],
                "phrase": expanded_lesson["phrase"],
                "options": request_template["options"]
            }

            response = {}
            response["phrase"] = request["phrase"]
            for option in request["options"]:
                if option == "practice":
                    response["practice"] = expand_practice(lesson_primitive)
                else:
                    response[option] = expanded_lesson[option]
            
            prompt = json.dumps(request, ensure_ascii=False)
            completion = json.dumps(response, ensure_ascii=False)
            translate_examples.append({
                "prompt": prompt,
                "completion": completion
            })

# Save examples
examples_path = Path("../data/processed/translate_examples.jsonl")
examples_path.parent.mkdir(parents=True, exist_ok=True)
with open(examples_path, 'w') as examples_file:
    for example in translate_examples:
        json_line = json.dumps(example, ensure_ascii=False)
        examples_file.write(json_line + '\n')
