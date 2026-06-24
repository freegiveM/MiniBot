from __future__ import annotations


class FakeModelClient:
    def __init__(self, outputs=None, model: str = "fake"):
        self.outputs = list(outputs or [])
        self.prompts: list[str] = []
        self.model = model
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        del max_new_tokens, kwargs
        self.prompts.append(prompt)
        self.last_completion_metadata = {
            "model": self.model,
            "input_chars": len(prompt),
            "prompt_cache_supported": False,
        }
        if self.outputs:
            return self.outputs.pop(0)
        return "<final>Done.</final>"

