from pathlib import Path
import json
import random
import copy

def expand_vocabulary(lesson_primitive, vocab_primitives):
    return [
        {
            "characters": vocab_primitives[component]["characters"],
            "definition": vocab_primitives[component]["definition"]
        }
        for component in lesson_primitive["vocabulary"]
    ]

def expand_pronunciation(lesson_primitive, vocab_primitives):
    return [
        {
            "character": vocab_primitives[component]["characters"],
            **vocab_primitives[component]["pronunciation"]
        }
        for component in lesson_primitive["pronunciation"]
    ]

def expand_practice(lesson_primitive):
    k = min(3, len(lesson_primitive["practice"]))
    return [
        {
            "phrase": sentence["phrase"],
            "characters": sentence["characters"]
        }
        for sentence in random.sample(lesson_primitive["practice"], k=k)
    ]

def expand_lesson(lesson_primitive, vocab_primitives):
    expanded_lesson = copy.deepcopy(lesson_primitive)
    expanded_lesson["vocabulary"] = expand_vocabulary(lesson_primitive, vocab_primitives)
    expanded_lesson["pronunciation"] = expand_pronunciation(lesson_primitive, vocab_primitives)
    return expanded_lesson

def inflect(english, tag):
    verb = english_inflections["verbs"].get(english.replace(" ", "_"))
    if not verb:
        return english
    
    match tag:
        case "happened":
            return verb["past_simple"]
        case "experienced":
            return verb["past_participle"]
        case "planned":
            return verb["future_simple"]
        case _:
            return english

raw_data_path = Path("../data/raw")

translate_primitives_path = raw_data_path / "translate_primitives.json"
lesson_primitives_path = raw_data_path / "lesson_primitives.json"
vocab_primitives_path = raw_data_path / "vocab_primitives.json"
english_inflections_path = raw_data_path / "english_inflections.json"

with open(translate_primitives_path, 'r') as translate_primitives_file:
    translate_primitives = json.load(translate_primitives_file)
with open(lesson_primitives_path, 'r') as lesson_primitives_file:
    lesson_primitives = json.load(lesson_primitives_file)
with open(vocab_primitives_path, 'r') as vocab_primitives_file:
    vocab_primitives = json.load(vocab_primitives_file)
with open(english_inflections_path, 'r') as english_inflections_file:
    english_inflections = json.load(english_inflections_file)

translate_examples = []

# Generate examples
for task in translate_primitives.keys():
    primitives_for_translate = translate_primitives[task]
    
    request_templates = primitives_for_translate["requests"]

    lesson_templates = lesson_primitives["templates"]
    lesson_fillers = lesson_primitives["fillers"]

    for lesson_filler in lesson_fillers:
        for lesson_tag in lesson_filler["tags"]:
            lesson_template = copy.deepcopy(lesson_templates[lesson_tag])

            inflected_english = inflect(lesson_filler["english"], lesson_tag)

            lesson_template["phrase"] = lesson_template["phrase"].format(english=inflected_english)
            lesson_template["characters"] = lesson_template["characters"].format(cantonese=lesson_filler["cantonese"])
            match lesson_template["mode"]:
                case "append":
                    lesson_template["vocabulary"].extend(lesson_filler["vocabulary"])
                    lesson_template["pronunciation"].extend(lesson_filler["pronunciation"])
                case "replace":
                    new_vocab = []
                    for item in lesson_template["vocabulary"]:
                        if item == "VERB":
                            new_vocab.extend(lesson_filler["vocabulary"])
                        else:
                            new_vocab.append(item)
                    lesson_template["vocabulary"] = new_vocab

                    new_pron = []
                    for item in lesson_template["pronunciation"]:
                        if item == "VERB":
                            new_pron.extend(lesson_filler["pronunciation"])
                        else:
                            new_pron.append(item)
                    lesson_template["pronunciation"] = new_pron
                case "prepend":
                    lesson_template["vocabulary"] = lesson_filler["vocabulary"] + lesson_template["vocabulary"]
                    lesson_template["pronunciation"] = lesson_filler["pronunciation"] + lesson_template["pronunciation"]
            lesson_template["grammar"] = lesson_template["grammar"].format(
                english=inflected_english,
                cantonese=lesson_filler["cantonese"]
            )
            for p in lesson_template["practice"]:
                p["phrase"] = p["phrase"].format(english=inflected_english)
                p["characters"] = p["characters"].format(cantonese=lesson_filler["cantonese"])

            expanded_lesson = expand_lesson(lesson_template, vocab_primitives)

            for combination in request_templates.keys():
                request_template = request_templates[combination]
                request = {
                    "intent": request_template["intent"],
                    "phrase": expanded_lesson["phrase"],
                    "options": request_template["options"]
                }

                response = {}
                response["phrase"] = request["phrase"]
                for option in request["options"]:
                    if option == "practice":
                        response["practice"] = expand_practice(expanded_lesson)
                    else:
                        response[option] = expanded_lesson[option]
                
                translate_examples.append({
                    "prompt": request,
                    "completion": response
                })

# Save examples
examples_path = Path("../data/processed/translate_examples.jsonl")
examples_path.parent.mkdir(parents=True, exist_ok=True)
with open(examples_path, 'w') as examples_file:
    for example in translate_examples:
        json_line = json.dumps(example, ensure_ascii=False)
        examples_file.write(json_line + '\n')
