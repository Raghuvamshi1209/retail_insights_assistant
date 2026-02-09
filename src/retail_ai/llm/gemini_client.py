from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import google.generativeai as genai


def _flatten(messages: List[Dict[str, str]]):
    system = []
    convo = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            system.append(content)
        elif role == "user":
            convo.append(f"User: {content}")
        else:
            convo.append(f"Assistant: {content}")
    out = ""
    if system:
        out += "\n".join(system).strip() + "\n\n"
    out += "\n".join(convo).strip()
    return out


@dataclass
class GeminiChat:
    model: str
    temperature: float = 0.1

    def __post_init__(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(self.model)

    def complete(self, messages: List[Dict[str, str]], max_output_tokens: int = 900):
        prompt = _flatten(messages)
        resp = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": float(self.temperature),
                "max_output_tokens": int(max_output_tokens),
            },
        )
        return (getattr(resp, "text", "") or "").strip()
