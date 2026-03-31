from pathlib import Path
import json

raw_data_path = Path("../data/raw")
redirect_primitives_path = raw_data_path / "redirect_primitives.json"

with open(redirect_primitives_path, 'r') as redirect_primitives_file:
    redirect_primitives = json.load(redirect_primitives_file)

redirect_examples = []

# Generate examples
for style in redirect_primitives.keys():
    primitives_for_style = redirect_primitives[style]

    categories = primitives_for_style["categories"]
    fillers = primitives_for_style["fillers"]
    templates = primitives_for_style["templates"]
    responses = primitives_for_style["responses"]

    for category, placeholder in categories.items():
        for filler in fillers[category]:
            format_dict = {placeholder: filler}
            for template in templates[category]:
                for response in responses[category]:
                    prompt = template.format(**format_dict)
                    completion = json.dumps({
                        "intent": "redirect",
                        "style": style,
                        "message": response.format(**format_dict)
                    })
                    redirect_examples.append({"prompt": prompt, "completion": completion})

# Save examples
examples_path = Path("../data/processed/redirect_examples.jsonl")
examples_path.parent.mkdir(parents=True, exist_ok=True)
with open(examples_path, 'w') as examples_file:
    for example in redirect_examples:
        json_line = json.dumps(example, ensure_ascii=False)
        examples_file.write(json_line + '\n')
