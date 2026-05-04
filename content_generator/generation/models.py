from pydantic import BaseModel, model_validator, Field
from typing import Self

class Prompt(BaseModel):
    intent: str
    phrase: str
    options: list[str]

class Vocabulary(BaseModel):
    characters: str
    definition: str

class Pronunciation(BaseModel):
    character: str
    sound: str
    description: str
    tone: str

class Practice(BaseModel):
    phrase: str
    characters: str

class Completion(BaseModel):
    phrase: str
    characters: str
    vocabulary: list[Vocabulary] | None = None
    pronunciation: list[Pronunciation] | None = None
    grammar: str | None = None
    practice: list[Practice] | None = None

    @model_validator(mode="after")
    def check_alignment(self) -> Self:
        self._check_vocabulary_alignment()
        self._check_pronunciation_alignment()
        return self

    def _check_vocabulary_alignment(self) -> None:
        if self.vocabulary is None:
            return
        
        characters = self.characters
        vocabulary = self.vocabulary
        cursor = 0
        for v in vocabulary:
            length = len(v.characters)
            if characters[cursor:cursor + length] != v.characters:
                raise ValueError(
                    f"Character mismatch in vocabulary at position {cursor}\n"
                    f"Full sentence: {characters}\n"
                    f"Expected: {characters[cursor:cursor + length]}\n"
                    f"Got: {v.characters}"
                )
            cursor += length
        if cursor != len(characters):
            raise ValueError(
                "Vocabulary does not fully cover characters\n"
                f"Stopped at index {cursor}, length is {len(characters)}"
            )

    def _check_pronunciation_alignment(self) -> None:
        if self.pronunciation is None:
            return

        characters = self.characters
        pronunciation = self.pronunciation
        if len(pronunciation) != len(characters):
            raise ValueError(
                f"Length mismatch in pronunciation\n"
                f"Characters: {characters}\n"
                f"Pronunciation: {pronunciation}"
            )
        for i, p in enumerate(pronunciation):
            if p.character != characters[i]:
                raise ValueError(
                    f"Character mismatch in pronunciation at position {i}\n"
                    f"Expected: {characters[i]}\n"
                    f"Got: {p.character}"
                )
    
def check_no_braces(obj):
    if isinstance(obj, str):
        if "{" in obj or "}" in obj:
            raise ValueError(f"Unresolved template braces found: {obj}")
    elif isinstance(obj, list):
        for item in obj:
            check_no_braces(item)
    elif isinstance(obj, dict):
        for v in obj.values():
            check_no_braces(v)

class Example(BaseModel):
    prompt: Prompt
    completion: Completion

    @model_validator(mode="after")
    def check_options_templates(self) -> Self:
        self._check_options_match()
        self._check_templates_resolved()
        return self

    def _check_options_match(self) -> None:
        prompt_options = set(self.prompt.options)
        completion_options = set(self.completion.model_dump(exclude_none=True).keys())
        completion_options.discard("phrase")
        if prompt_options != completion_options:
            raise ValueError(
                f"Options mismatch\n"
                f"Prompt: {prompt_options}\n"
                f"Completion: {completion_options}"
            )
    
    def _check_templates_resolved(self) -> None:
        check_no_braces(self.model_dump())

class Response(BaseModel):
    translation_correct: bool
    vocabulary_correct: bool
    pronunciation_correct: bool
    practice_quality: int = Field(gt=1, lt=5)
    naturalness: int = Field(gt=1, lt=5)
    issues: list[str]
