"""
Free LLM via Groq API - very generous free tier.
"""
import os
from groq import Groq

_client = Groq(api_key=os.environ["GROQ_API_KEY"])

class GroqMessage:
    def __init__(self, text):
        self.text = text

class GroqContent:
    def __init__(self, text):
        self.content = [GroqMessage(text)]

class _Messages:
    def create(self, model=None, max_tokens=None, messages=None, system=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        for msg in (messages or []):
            msgs.append(msg)

        response = _client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=msgs,
            max_tokens=max_tokens or 1000,
        )
        return GroqContent(response.choices[0].message.content)

class GroqClient:
    def __init__(self):
        self.messages = _Messages()
