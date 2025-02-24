
import os
import google.generativeai as palm
import asyncio
from typing import Optional, List
from langchain.llms.base import LLM
from pydantic import Field

class GoogleGeminiLLM(LLM):
    model: str = Field(default="default-gemini-model-id")
    api_key: str = Field(...)

    class Config:
        extra = "allow"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key is required for Google Gemini. Please set the GOOGLE_GEMINI_API_KEY environment variable.")
        model = model or os.environ.get("GOOGLE_GEMINI_MODEL_ID") or "default-gemini-model-id"
        super().__init__(model=model, api_key=api_key, **kwargs)
        palm.configure(api_key=api_key)

    @property
    def _llm_type(self) -> str:
        return "google_gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = palm.generate_text(model=self.model, prompt=prompt)
        return response.result if response and hasattr(response, 'result') else ""

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: palm.generate_text(model=self.model, prompt=prompt))
        return response.result if response and hasattr(response, 'result') else ""
