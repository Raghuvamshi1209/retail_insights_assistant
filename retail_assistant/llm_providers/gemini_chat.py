from __future__ import annotations
import os
from typing import List, Dict
import google.generativeai as genai


def _flatten(messages: List[Dict[str, str]]) -> str:
    sys = []
    convo = []
    for m in messages:
        role = m.get('role')
        content = m.get('content', '')
        if role == 'system':
            sys.append(content)
        elif role == 'user':
            convo.append(f"User: {content}")
        else:
            convo.append(f"Assistant: {content}")
    out = ''
    if sys:
        out += '\n'.join(sys).strip() + '\n\n'
    out += '\n'.join(convo).strip()
    return out


class GeminiChat:
    def __init__(self, model: str = 'gemini-2.5-flash', temperature: float = 0.1, api_key: str | None = None):
        key = api_key or os.getenv('GEMINI_API_KEY')
        if not key:
            raise RuntimeError('GEMINI_API_KEY is not set')
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model)
        self.temperature = float(temperature)

    def complete(self, messages: List[Dict[str, str]], max_output_tokens: int = 800) -> str:
        prompt = _flatten(messages)
        resp = self.model.generate_content(
            prompt,
            generation_config={'temperature': self.temperature, 'max_output_tokens': int(max_output_tokens)},
        )
        return (getattr(resp, 'text', '') or '').strip()
