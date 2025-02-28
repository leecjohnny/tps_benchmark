"""
Data models for LLM provider configurations.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union


@dataclass
class ProviderConfig:
    """Base configuration for an LLM provider."""

    name: str
    """The name of the provider (e.g., 'openai', 'anthropic')."""

    endpoint: Union[str, Callable[[], str]]
    """The API endpoint path or a callable that returns it."""

    auth_env: str
    """The environment variable name for the API token."""

    model: str
    """The model identifier to use."""

    max_tokens: int = 512
    """Maximum number of tokens to generate."""

    def create_request_payload(self, prompt: str) -> Dict[str, Any]:
        """
        Create the request payload for this provider.

        Args:
            prompt: The prompt to send to the API.

        Returns:
            The request payload as a dictionary.
        """
        raise NotImplementedError("Subclasses must implement create_request_payload")


@dataclass
class OpenAIConfig(ProviderConfig):
    """Configuration for OpenAI-compatible providers."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        super().__init__(
            name="openai",
            endpoint="chat/completions",
            auth_env="OPENAI_TOKEN",
            model=model,
        )

    def create_request_payload(self, prompt: str) -> Dict[str, Any]:
        """Create the request payload for this provider."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        if self.model.startswith("o"):
            del payload["max_tokens"]
            payload["max_completion_tokens"] = self.max_tokens
        return payload

    @classmethod
    def get_all_models(cls) -> List[ProviderConfig]:
        """Get configurations for all OpenAI models to benchmark."""
        models = [
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "o3-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]
        return [cls(model=model) for model in models]


@dataclass
class AnthropicConfig(ProviderConfig):
    """Configuration for Anthropic Claude."""

    def __init__(self, model: str = "claude-3-opus-20240229"):
        super().__init__(
            name="anthropic",
            endpoint="v1/messages",
            auth_env="ANTHROPIC_TOKEN",
            model=model,
        )

    def create_request_payload(self, prompt: str) -> Dict[str, Any]:
        """Create the request payload for this provider."""
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def get_all_models(cls) -> List[ProviderConfig]:
        """Get configurations for all Anthropic models to benchmark."""
        models = [
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-opus-latest",
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
            "claude-2.1",
        ]
        return [cls(model=model) for model in models]


@dataclass
class GoogleVertexConfig(ProviderConfig):
    """Configuration for Google Vertex AI."""

    def __init__(self, model: str = "gemini-1.0-pro"):
        super().__init__(
            name="google-vertex-ai",
            endpoint=self._get_endpoint,
            auth_env="GOOGLE_VERTEX_TOKEN",
            model=model,
        )
        self.project_name = "alien-vim-441515-u1"
        self.region = "us-central1"

    def _get_endpoint(self) -> str:
        """Generate the endpoint URL with project and region details."""

        # For Cloudflare AI Gateway, we use a simpler endpoint format
        return f"v1/projects/{self.project_name}/locations/{self.region}/publishers/google/models/{self.model}:generateContent"

    def create_request_payload(self, prompt: str) -> Dict[str, Any]:
        """Create the request payload for this provider."""
        return {
            "contents": {"role": "user", "parts": [{"text": prompt}]},
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
            },
        }

    @classmethod
    def get_all_models(cls) -> List[ProviderConfig]:
        """Get configurations for all Google Vertex AI models to benchmark."""
        models = [
            "gemini-1.0-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-pro-exp-02-05",
            "gemini-2.0-flash-thinking-exp-01-21",
        ]
        return [cls(model=model) for model in models]


@dataclass
class DeepseekConfig(ProviderConfig):
    """Configuration for Deepseek."""

    def __init__(self, model: str = "deepseek-chat"):
        super().__init__(
            name="deepseek",
            endpoint="chat/completions",
            auth_env="DEEPSEEK_TOKEN",
            model=model,
        )

    def create_request_payload(self, prompt: str) -> Dict[str, Any]:
        """Create the request payload for this provider."""
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def get_all_models(cls) -> List[ProviderConfig]:
        """Get configurations for all Deepseek models to benchmark."""
        models = [
            "deepseek-chat",
            "deepseek-reasoner",
        ]
        return [cls(model=model) for model in models]


@dataclass
class WorkersAIConfig(ProviderConfig):
    """Configuration for Workers AI."""

    def __init__(self, model: str = "@cf/meta/llama-3.1-8b-instruct"):
        super().__init__(
            name="workers-ai",
            endpoint=model,
            auth_env="WORKERS_AI_TOKEN",
            model=model,
        )

    def create_request_payload(self, prompt: str) -> Dict[str, Any]:
        """Create the request payload for this provider."""
        return {
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def get_all_models(cls) -> List[ProviderConfig]:
        """Get configurations for all Workers AI models to benchmark."""
        models = [
            "@cf/meta/llama-3.1-70b-instruct",
            "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        ]
        return [cls(model=model) for model in models]


@dataclass
class MistralConfig(ProviderConfig):
    """Configuration for Mistral AI."""

    def __init__(self, model: str = "mistral-large-latest"):
        super().__init__(
            name="mistral",
            endpoint="chat/completions",
            auth_env="MISTRAL_TOKEN",
            model=model,
        )

    def create_request_payload(self, prompt: str) -> Dict[str, Any]:
        """Create the request payload for this provider."""
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def get_all_models(cls) -> List[ProviderConfig]:
        """Get configurations for all Mistral models to benchmark."""
        models = [
            "mistral-large-latest",
            "ministral-8b-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
        ]
        return [cls(model=model) for model in models]


# Get all models for each provider
ALL_OPENAI_MODELS = OpenAIConfig.get_all_models()
ALL_ANTHROPIC_MODELS = AnthropicConfig.get_all_models()
ALL_GOOGLE_VERTEX_MODELS = GoogleVertexConfig.get_all_models()
ALL_DEEPSEEK_MODELS = DeepseekConfig.get_all_models()
ALL_WORKERS_AI_MODELS = WorkersAIConfig.get_all_models()
ALL_MISTRAL_MODELS = MistralConfig.get_all_models()

# Default provider configurations (one model per provider)
DEFAULT_PROVIDERS = [
    OpenAIConfig(),
    AnthropicConfig(),
    GoogleVertexConfig(),
    DeepseekConfig(),
    # WorkersAIConfig(),
    MistralConfig(),
]

# All models from all providers
ALL_MODELS = (
    ALL_OPENAI_MODELS
    + ALL_ANTHROPIC_MODELS
    + ALL_GOOGLE_VERTEX_MODELS
    + ALL_DEEPSEEK_MODELS
    # + ALL_WORKERS_AI_MODELS
    + ALL_MISTRAL_MODELS
)
