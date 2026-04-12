from pathlib import Path
import json

raw_data_path = Path("../data/raw")
translate_primitives_path = raw_data_path / "translate_primitives.json"

with open(translate_primitives_path, 'r') as translate_primitives_file:
    translate_primitives = json.load(translate_primitives_file)

translate_examples = []

# Generate examples
for task in translate_primitives.keys():
    primitives_for_translate = translate_primitives[task]

    fillers = primitives_for_translate["fillers"]
    templates = primitives_for_translate["templates"]
    responses = primitives_for_translate["responses"]

    for filler in fillers:
        for template_type, template_list in templates.items():
            response = responses[template_type]
            for template in template_list:
                prompt = template["template"].format(phrase=filler)
                completion = json.dumps({
                    "intent": response["intent"],
                    "phrase": filler,
                    "options": response["options"]
                })
                translate_examples.append({
                    "prompt": prompt,
                    "completion": completion,
                    "cleanliness": template["cleanliness"]
                })

# Save examples
examples_path = Path("../data/processed/translate_examples.jsonl")
examples_path.parent.mkdir(parents=True, exist_ok=True)
with open(examples_path, 'w') as examples_file:
    for example in translate_examples:
        json_line = json.dumps(example, ensure_ascii=False)
        examples_file.write(json_line + '\n')
