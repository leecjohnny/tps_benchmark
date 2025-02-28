"""
API client implementations for different LLM providers.
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import aiohttp
import tenacity

from llm_benchmark.models import ProviderConfig

# Set up logging
logger = logging.getLogger("llm_benchmark")


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, config: ProviderConfig, run_id: str):
        """
        Initialize the client with a provider configuration.

        Args:
            config: The provider configuration.
        """
        self.config = config
        self.name = config.name
        self.run_id = run_id

    @abstractmethod
    async def call_async(
        self, prompt: str, max_tokens: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Call the LLM API with the given prompt asynchronously.

        Args:
            prompt: The prompt to send to the API.
            max_tokens: Maximum number of tokens to generate (if supported by the provider).

        Returns:
            A tuple of (response_data, elapsed_time).
        """
        pass

    def call(
        self, prompt: str, max_tokens: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Call the LLM API with the given prompt (synchronous wrapper).

        Args:
            prompt: The prompt to send to the API.
            max_tokens: Maximum number of tokens to generate (if supported by the provider).

        Returns:
            A tuple of (response_data, elapsed_time).
        """
        return asyncio.run(self.call_async(prompt, max_tokens))

    @abstractmethod
    def extract_text(self, response: Dict[str, Any]) -> str:
        """
        Extract the generated text from the API response.

        Args:
            response: The API response data.

        Returns:
            The generated text.
        """
        pass

    @abstractmethod
    def extract_detailed_token_counts(self, response: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract detailed token counts from the API response.

        Args:
            response: The API response data.

        Returns:
            A dictionary with 'prompt_tokens', 'completion_tokens', and 'total_tokens'.
            If specific counts are not available, they will be estimated or set to 0.
        """
        pass


class CloudflareGatewayClient(LLMClient):
    """Client for accessing LLM providers through the Cloudflare AI Gateway."""

    def __init__(self, config: ProviderConfig, run_id: str):
        """
        Initialize the client with a provider configuration.

        Args:
            config: The provider configuration.
        """
        super().__init__(config, run_id)
        self.account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        self.gateway_id = os.environ["CLOUDFLARE_GATEWAY_ID"]
        if not self.account_id or not self.gateway_id:
            raise ValueError(
                "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_GATEWAY_ID environment variables must be set"
            )

        self.base_url = (
            f"https://gateway.ai.cloudflare.com/v1/{self.account_id}/{self.gateway_id}"
        )
        self.auth_token = os.environ.get(config.auth_env, "")

        if not self.auth_token:
            logger.warning(
                f"No API token found for {config.name} (env: {config.auth_env})"
            )

    async def call_async(
        self, prompt: str, max_tokens: Optional[int] = None
    ) -> Tuple[Dict[str, Any], float]:
        """
        Call the LLM API with the given prompt asynchronously.

        Args:
            prompt: The prompt to send to the API.
            max_tokens: Maximum number of tokens to generate (if supported by the provider).

        Returns:
            A tuple of (response_data, elapsed_time).
        """
        # Get the endpoint (if callable, e.g., for Google Vertex, else use string)
        endpoint = (
            self.config.endpoint()
            if callable(self.config.endpoint)
            else self.config.endpoint
        )

        # Build the query using the prompt
        query = self.config.create_request_payload(prompt)

        # Add max_tokens to the query if provided
        if max_tokens is not None:
            # Most providers use max_tokens in the top level of the query
            if isinstance(query, dict):
                query["max_tokens"] = max_tokens

            # For providers that use a nested structure with 'parameters'
            if (
                isinstance(query, dict)
                and "parameters" in query
                and isinstance(query["parameters"], dict)
            ):
                query["parameters"]["max_tokens"] = max_tokens

        # Construct the message payload for the universal endpoint
        auth_header_key = (
            "Authorization" if self.config.name != "anthropic" else "x-api-key"
        )
        auth_header_value = (
            f"Bearer {self.auth_token}"
            if self.config.name != "anthropic"
            else f"{self.auth_token}"
        )

        # Create headers with all needed keys upfront
        headers = {
            auth_header_key: auth_header_value,
            "Content-Type": "application/json",
            "cf-aig-metadata": json.dumps(
                {
                    "run_id": self.run_id,
                }
            ),
        }

        # Add Anthropic-specific header if needed
        if self.config.name == "anthropic":
            headers["anthropic-version"] = "2023-06-01"

        message = {
            "provider": self.name,
            "endpoint": endpoint,
            "headers": headers,
            "query": query,
        }
        payload = [message]  # Universal endpoint expects an array of messages

        logger.info(f"Calling {self.name} API...")
        logger.debug(f"Request payload: {json.dumps(payload)}")

        start_time = time.time()
        response_data = {}

        @tenacity.retry(
            retry=tenacity.retry_if_exception_type(aiohttp.ClientResponseError),
            wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
            stop=tenacity.stop_after_attempt(3),
            retry_error_callback=lambda retry_state: retry_state.outcome.result(),  # type: ignore
            before_sleep=lambda retry_state: logger.info(
                f"Rate limited or server error ({retry_state.outcome.exception().status}). Retrying in {retry_state.next_action.sleep} seconds..."  # type: ignore
            ),
        )
        async def make_request():
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=payload) as response:
                    status_code = response.status
                    response_text = await response.text()

                    if status_code == 429 or status_code == 500:
                        # Raise an exception that will trigger the retry
                        raise aiohttp.ClientResponseError(
                            request_info=None,
                            history=None,
                            status=status_code,
                            message=f"Rate limited: {response_text}",
                        )

                    if status_code != 200:
                        error_msg = f"API request failed with status code {status_code}: {response_text}, model: {self.config.model if hasattr(self.config, 'model') else 'unknown'}"
                        logger.error(error_msg)
                        return {
                            "status_code": status_code,
                            "response_text": response_text,
                        }
                    else:
                        try:
                            response_data = await response.json()
                            logger.debug(f"Response: {json.dumps(response_data)}")
                            return response_data
                        except (json.JSONDecodeError, IndexError) as e:
                            error_msg = f"Failed to parse API response: {str(e)}"
                            logger.error(error_msg)
                            logger.error(f"Response content: {response_text}")
                            return {
                                "status_code": 500,
                                "response_text": response_text,
                            }

        try:
            response_data = await make_request()
        except aiohttp.ClientError as e:
            error_msg = f"API request failed: {str(e)}"
            logger.error(error_msg)
            response_data = {"status_code": 500, "response_text": str(e)}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            response_data = {"status_code": 500, "response_text": str(e)}

        elapsed = time.time() - start_time
        logger.info(f"API call completed in {elapsed:.2f} seconds")

        return response_data, elapsed

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract generated text from a Cloudflare Gateway response."""
        raise NotImplementedError("Cloudflare Gateway does not support text extraction")

    def extract_detailed_token_counts(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract detailed token counts from a Cloudflare Gateway response."""
        raise NotImplementedError(
            "Cloudflare Gateway does not support detailed token counts extraction"
        )


class OpenAIClient(CloudflareGatewayClient):
    """Client for OpenAI API."""

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract generated text from an OpenAI response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.warning("Failed to extract text from OpenAI response")
            logger.debug(f"Response structure: {json.dumps(response)}")
            return ""

    def extract_detailed_token_counts(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract detailed token counts from an OpenAI response."""

        try:
            return response["usage"]
        except (KeyError, TypeError):
            logger.warning(
                "Failed to extract detailed token counts from OpenAI response"
            )
            logger.debug(f"Response structure: {json.dumps(response)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class AnthropicClient(CloudflareGatewayClient):
    """Client for Anthropic API."""

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract generated text from an Anthropic response."""
        try:
            return response["content"][0]["text"]
        except (KeyError, TypeError):
            logger.warning("Failed to extract text from Anthropic response")
            logger.debug(f"Response structure: {json.dumps(response)}")
            return ""

    def extract_detailed_token_counts(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract detailed token counts from an Anthropic response."""

        try:
            anthropic_response = response["usage"]
            anthropic_response["total_tokens"] = anthropic_response.get(
                "input_tokens", 0
            ) + anthropic_response.get("output_tokens", 0)
            return anthropic_response
        except (KeyError, TypeError):
            logger.warning(
                "Failed to extract detailed token counts from Anthropic response"
            )
            logger.debug(f"Response structure: {json.dumps(response)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class GoogleVertexClient(CloudflareGatewayClient):
    """Client for Google Vertex AI."""

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract generated text from a Google Vertex AI response."""
        # Check if we have an error response
        if "status_code" in response and response["status_code"] != 200:
            return ""

        try:
            # Join all parts together in case there are multiple parts
            parts = response["candidates"][0]["content"]["parts"]
            return "".join(part["text"] for part in parts)
        except (KeyError, IndexError):
            logger.warning("Failed to extract text from Google Vertex AI response")
            logger.debug(f"Response structure: {json.dumps(response)}")
            return ""

    def extract_detailed_token_counts(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract detailed token counts from a Google Vertex AI response."""

        try:
            usage_metadata = response["usageMetadata"]
            return {
                "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                "completion_tokens": usage_metadata.get("completionTokenCount", 0),
                "total_tokens": usage_metadata.get("totalTokenCount", 0),
            }
        except (KeyError, TypeError):
            logger.warning(
                "Failed to extract detailed token counts from Google Vertex AI response"
            )
            logger.debug(f"Response structure: {json.dumps(response)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class DeepseekClient(CloudflareGatewayClient):
    """Client for Deepseek API."""

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract generated text from a Deepseek response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.warning("Failed to extract text from Deepseek response")
            logger.debug(f"Response structure: {json.dumps(response)}")
            return ""

    def extract_detailed_token_counts(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract detailed token counts from a Deepseek response."""
        try:
            return response["usage"]
        except (KeyError, TypeError):
            logger.warning(
                "Failed to extract detailed token counts from Deepseek response"
            )
            logger.debug(f"Response structure: {json.dumps(response)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class WorkersAIClient(CloudflareGatewayClient):
    """Client for Workers AI."""

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract generated text from a Workers AI response."""
        try:
            return response["result"]["response"]
        except (KeyError, IndexError):
            logger.warning("Failed to extract text from Workers AI response")
            logger.debug(f"Response structure: {json.dumps(response)}")
            return ""

    def extract_detailed_token_counts(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract detailed token counts from a Workers AI response."""

        try:
            return response["result"]["usage"]
        except (KeyError, TypeError):
            logger.warning(
                "Failed to extract detailed token counts from Workers AI response"
            )
            logger.debug(f"Response structure: {json.dumps(response)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class MistralClient(CloudflareGatewayClient):
    """Client for Mistral API."""

    def extract_text(self, response: Dict[str, Any]) -> str:
        """Extract generated text from a Mistral response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            logger.warning("Failed to extract text from Mistral response")
            logger.debug(f"Response structure: {json.dumps(response)}")
            return ""

    def extract_detailed_token_counts(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract detailed token counts from a Mistral response."""
        try:
            return response["usage"]
        except (KeyError, TypeError):
            logger.warning(
                "Failed to extract detailed token counts from Mistral response"
            )
            logger.debug(f"Response structure: {json.dumps(response)}")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# Client factory
def get_client(config: ProviderConfig, run_id: str) -> LLMClient:
    """
    Get the appropriate client for the given provider configuration.

    Args:
        config: The provider configuration.

    Returns:
        An LLM client instance.
    """
    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google-vertex-ai": GoogleVertexClient,
        "deepseek": DeepseekClient,
        "workers-ai": WorkersAIClient,
        "mistral": MistralClient,
    }

    client_class = clients.get(config.name)
    if not client_class:
        raise ValueError(f"No client implementation for provider: {config.name}")

    return client_class(config, run_id)
