"""
LLM Client abstraction layer to support multiple providers (OpenAI, Ollama, etc.)
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMClient(ABC):
    """Base class for LLM clients"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate a response from the LLM

        Args:
            prompt: User prompt
            system_prompt: System prompt/instructions
            model: Model name (optional, uses default if not provided)

        Returns:
            Generated text response
        """
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client wrapper"""

    def __init__(self, client, model: str = "gpt-4o"):
        """
        Initialize OpenAI client

        Args:
            client: OpenAI client instance
            model: Default model name
        """
        self.client = client
        self.default_model = model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate response using OpenAI API"""
        model_to_use = model or self.default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=messages,
        )

        return response.choices[0].message.content.strip()


class OllamaClient(BaseLLMClient):
    """Ollama LLM client wrapper"""

    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client

        Args:
            model: Model name (e.g., "llama2", "mistral", "gpt-oss:20b")
            base_url: Ollama server URL
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package required for Ollama client. Install with: pip install openai"
            )

        self.model = model
        self.base_url = base_url
        # Ollama is compatible with OpenAI API
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",  # Ollama doesn't require a real API key
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate response using Ollama API"""
        model_to_use = model or self.model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=messages,
        )

        return response.choices[0].message.content.strip()
