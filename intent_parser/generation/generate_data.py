from pathlib import Path
import json

raw_data_path = Path("../data/raw")

lessons_path = raw_data_path / "lessons.json"
vocab_path = raw_data_path / "vocab.json"

with open(lessons_path, 'r', encoding="utf-8") as lessons_file:
    lessons = json.load(lessons_file)
with open(vocab_path, 'r', encoding="utf-8") as vocab_file:
    vocab = json.load(vocab_file)

sft_examples = []

# Inject diversity
introductions = [
    "Absolutely! Let's break this down.",
    "Sure thing, that's a great question! ",
    "Happy to help! Let's tackle this step by step."
]
chinese_transitions = [
    "In traditional Chinese that would read as",
    "The Chinese characters for that would be",
    "A natural way to say that in Chinese is"
]
jyutping_transitions = [
    "The associated Jyutping would be",
    "In Jyutping that reads as",
    "A direct translation to Jyutping reads like"
]
conclusions = [
    "If you have any further questions, feel free to ask!",
    "Make sure to practise via repetition!",
    "Now that you know, give it a try yourself!"
]

# Generate lessons
for lesson in lessons:
    vocab_entries = [vocab[entry] for entry in lesson["vocabulary"]]
    lesson_vocab = [(vocab_entry["chinese"], vocab_entry["jyutping"], vocab_entry["definition"]) for vocab_entry in vocab_entries]
    vocab_lines = []
    for item in lesson_vocab:
        vocab_lines.append(f"{item[0]} = {item[1]} = {item[2]}")
    vocab_string = '\n'.join(vocab_lines)

    pronunciation_entries = [vocab[entry] for entry in lesson["pronunciation"]]
    lesson_pronunciation = [(pronunciation_entry["jyutping"], pronunciation_entry["pronunciation"]) for pronunciation_entry in pronunciation_entries]
    pronunciation_lines = []
    for item in lesson_pronunciation:
        pronunciation_lines.append(item[0])
        pronunciation_lines.append(f"Sounds like: {item[1]['sound']}.")
        pronunciation_lines.append(f"How to say it: {item[1]['description']}.")
        pronunciation_lines.append(f"Tone: {item[1]['tone']}.")
        pronunciation_lines.append("")
    pronunciation_string = '\n'.join(pronunciation_lines)

    practice_lines = []
    for item in lesson["practice"]:
        practice_lines.append(item["chinese"])
        practice_lines.append(item["jyutping"])
        practice_lines.append(f"({item['english']})")
        practice_lines.append("")
    practice_string = '\n'.join(practice_lines)

    prompt_full = f"How would you say \"{lesson['english']}\" in Cantonese? Could you provide a full explanation of the translation along with some practice examples?"
    for introduction in introductions:
        for chinese_transition in chinese_transitions:
            for jyutping_transition in jyutping_transitions:
                for conclusion in conclusions:
                    completion_full_lines = []
                    completion_full_lines.append(introduction)
                    completion_full_lines.append(f"Chinese:\n{chinese_transition} {lesson['chinese']}.")
                    completion_full_lines.append(f"Jyutping:\n{jyutping_transition} {lesson['jyutping']}.")
                    completion_full_lines.append(f"Vocabulary:\n{vocab_string.rstrip()}")
                    completion_full_lines.append(f"Pronunciation:\n{pronunciation_string.rstrip()}")
                    completion_full_lines.append(f"Grammar Note:\n{lesson['grammar']}")
                    completion_full_lines.append(f"Practice Sentences:\n{practice_string.rstrip()}")
                    completion_full_lines.append(conclusion)
                    completion_full = "\n\n".join(completion_full_lines)
                    sft_examples.append({"prompt": prompt_full, "completion": completion_full})
    
    prompt_breakdown = f"Could you break down \"{lesson['jyutping']}\" for me? Just give me the meaning and pronunciation of each part please."
    for introduction in introductions:
        for conclusion in conclusions:
            completion_breakdown_lines = []
            completion_breakdown_lines.append(introduction)
            completion_breakdown_lines.append(f"Vocabulary:\n{vocab_string.rstrip()}")
            completion_breakdown_lines.append(f"Pronunciation:\n{pronunciation_string.rstrip()}")
            completion_breakdown_lines.append(conclusion)
            completion_breakdown = "\n\n".join(completion_breakdown_lines)
            sft_examples.append({"prompt": prompt_breakdown, "completion": completion_breakdown})
    
    prompt_translation = f"What is \"{lesson['english']}\" in Cantonese? I only want the translation please, nothing extra."
    for introduction in introductions:
        for chinese_transition in chinese_transitions:
            for jyutping_transition in jyutping_transitions:
                for conclusion in conclusions:
                    completion_translation_lines = []
                    completion_translation_lines.append(introduction)
                    completion_translation_lines.append(f"Chinese:\n{chinese_transition} {lesson['chinese']}.")
                    completion_translation_lines.append(f"Jyutping:\n{jyutping_transition} {lesson['jyutping']}.")
                    completion_translation_lines.append(conclusion)
                    completion_translation = "\n\n".join(completion_translation_lines)
                    sft_examples.append({"prompt": prompt_translation, "completion": completion_translation})

# Inject diversity
offtopic_fillers = {
    "hobbies": ("movie", "song", "sport", "video game"),
    "queries": ("quantum physics", "machine learning")
}
offtopic_templates = {
    "hobbies": (
        "What's your favourite {hobby}?",
        "Could you suggest me a cool {hobby}?",
        "Are you interested in {hobby}?"
    ),
    "queries": (
        "Explain {query} to me.",
        "Can you explain {query}?",
        "Tell me how {query} works."
    )
}
offtopic_completions = {
    "hobbies": (
        "That's a fun question, but let's get back to learning Cantonese. What would you like to translate?",
        "I enjoy talking about that, but let's stay focused on Cantonese. The more you practise, the quicker you'll get better!",
        "I'd love to help you with that, but I'm equipped only to teach you Cantonese. Please ask me something about that instead!",
        "That sounds fun, but let's keep our attention on learning Cantonese today. Could you try and translate some of that into Cantonese?" 
    ),
    "queries": (
        "That's a fun question, but let's get back to learning Cantonese. What would you like to translate?",
        "I enjoy talking about that, but let's stay focused on Cantonese. The more you practise, the quicker you'll get better!",
        "I'd love to help you with that, but I'm equipped only to teach you Cantonese. Please ask me something about that instead!",
        "That sounds fun, but let's keep our attention on learning Cantonese today. Could you try and translate some of that into Cantonese?" 
    )
}

smalltalk_fillers = {
    "greetings": ("Hi", "Hello", "Hey", "What's up"),
    "feelings": ("good", "great", "awesome", "amazing")
}
smalltalk_templates = {
    "greetings": (
        "{interjection}, how are you?",
        "{interjection}, how are you doing?",
        "{interjection}, how are you feeling?"
    ),
    "feelings": (
        "I've been feeling {feeling} recently!",
        "It has been a {feeling} week for me."
    )
}
smalltalk_completions = {
    "greetings": (
        "I'm doing well! Since we're learning Cantonese, do you know how to say 'How are you?' in Cantonese?",
        "I'm great thanks! I'm really happy to be helping you pick up Cantonese!",
        "Happy to be here and to be of help! Let's keep practising!"
    ),
    "feelings": (
        "That's great to hear! Let's keep channelling that energy into learning Cantonese!",
        "I'm glad to hear that! Hopefully your Cantonese studies are going just as well!",
        "That sounds awesome! I'm glad you've been feeling {feeling} lately!"
    )
}

encouragement_fillers = {
    "difficulty": ("hard", "tough", "frustrating", "difficult"),
    "understanding": ("get", "grasp", "understand")
}
encouragement_templates = {
    "difficulty": (
        "This is too {difficulty}.",
        "Why is this so {difficulty}?"
    ),
    "understanding": (
        "I don't {understanding} it.",
        "This is impossible to {understanding}."
    )
}
encouragement_completions = {
    "difficulty": (
        "Learning Cantonese can take time, but you're doing great. I'm here to help you along the way so let's keep practicing!",
        "Feeling a bit confused is just part of the process. If there's anything in particular you don't understand, feel free to ask me!",
        "Don't feel down, frustration is a natural part of learning. We're taking this step by step together so don't hesitate to ask for help.",
        "Yeah, learning something {difficulty} can be tricky, but you'll get it step by step!",
    ),
    "understanding": (
        "Learning Cantonese can take time, but you're doing great. I'm here to help you along the way so let's keep practicing!",
        "Feeling a bit confused is just part of the process. If there's anything in particular you don't understand, feel free to ask me!",
        "Don't feel down, frustration is a natural part of learning. We're taking this step by step together so don't hesitate to ask for help.",
        "I know it can be tough to {understanding} at first, but keep going — you're making progress!"
    )
}

# Generate redirections
def generate_redirections(categories, fillers, templates, completions):
    for category, placeholder in categories.items():
        for filler in fillers[category]:
            for template in templates[category]:
                for completion in completions[category]:
                    sft_examples.append({
                        "prompt": template.format(**{placeholder: filler}),
                        "completion": completion.format(**{placeholder: filler})
                    })

offtopic_categories = {
    "hobbies": "hobby", 
    "queries": "query"
}
generate_redirections(offtopic_categories, offtopic_fillers, offtopic_templates, offtopic_completions)

smalltalk_categories = {
    "greetings": "interjection",
    "feelings": "feeling"
}
generate_redirections(smalltalk_categories, smalltalk_fillers, smalltalk_templates, smalltalk_completions)

encouragement_categories = {
    "difficulty": "difficulty",
    "understanding": "understanding"
}
generate_redirections(encouragement_categories, encouragement_fillers, encouragement_templates, encouragement_completions)

# Save examples
examples_path = Path("../data/processed/sft_examples.jsonl")
examples_path.parent.mkdir(parents=True, exist_ok=True)
with open(examples_path, "w", encoding="utf-8") as examples_file:
    for example in sft_examples:
        json_string = json.dumps(example, ensure_ascii=False)
        examples_file.write(json_string + '\n')
