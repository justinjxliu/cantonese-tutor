from pathlib import Path
import json
import yaml
from google import genai
from dotenv import load_dotenv
import os
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) # For exponential backoff
from models import Response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(contents):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": Response.model_json_schema()
        }
    )

    if response.text is None:
        raise ValueError("Empty response")

    # Force retry if JSON is invalid
    try:
        return Response.model_validate_json(response.text).model_dump()
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}\nRAW RESPONSE:\n{response.text}")

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Input and output paths
translate_examples_path = Path("../data/processed/translate_examples.jsonl")
results_path = Path("../data/processed/judgements.jsonl")

# Load judge prompt
judge_prompt_path = Path("../prompts/example_judge.yaml")
with open(judge_prompt_path) as judge_prompt_file:
    JUDGE_PROMPT = yaml.safe_load(judge_prompt_file)["prompt"]

def judge(example: dict) -> dict:
    # Format input cleanly
    contents = f"""
{JUDGE_PROMPT}

-------------------------
EXAMPLE TO EVALUATE
-------------------------
{json.dumps(example, ensure_ascii=False)}
"""
    return completion_with_backoff(contents)

with open(translate_examples_path, 'r') as translate_examples_file, open(results_path, 'w') as results_file:
    for i, line in enumerate(translate_examples_file, start=1):
        # Only evaluate full combinations
        if i % 10 != 0:
            continue

        translate_example = json.loads(line)

        try:
            judgement = judge(translate_example)
            result = {
                "example":translate_example,
                "judgement": judgement
            }
        except Exception as e:
            result = {
                "example": translate_example,
                "error": str(e)
            }
        
        # Save results
        results_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Processed line {i}")
