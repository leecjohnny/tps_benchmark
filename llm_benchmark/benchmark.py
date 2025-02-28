"""
Benchmarking functionality for LLM providers.
"""

import asyncio
import logging
import os
import secrets
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from llm_benchmark.client import LLMClient, get_client
from llm_benchmark.models import DEFAULT_PROVIDERS, ProviderConfig

# Set up logging
logger = logging.getLogger("llm_benchmark")


@dataclass
class BenchmarkResult:
    """Results of a benchmark run for a single provider."""

    provider_name: str
    """Name of the provider."""

    average_tps: float
    """Average tokens per second."""

    median_tps: float
    """Median tokens per second."""

    average_tokens: float
    """Average number of tokens generated."""

    token_details: List[Dict[str, Any]]
    """Detailed token information for each call."""

    elapsed_times: List[float]
    """Elapsed time for each call."""


class BenchmarkRunner:
    """Runner for LLM benchmarks."""

    def __init__(self, providers: Optional[List[ProviderConfig]] = None):
        """
        Initialize the benchmark runner.

        Args:
            providers: List of provider configurations to benchmark.
                If None, uses the default providers.
        """
        self.providers = providers or DEFAULT_PROVIDERS
        self.results: Dict[str, BenchmarkResult] = {}

        # Set up logging to file and console
        self._setup_logging()
        self.run_id = str(uuid.uuid4())
        # Create a DataFrame for request/response logging
        self.requests_log = pd.DataFrame(
            columns=[
                "timestamp",
                "provider",
                "endpoint",
                "request_payload",
                "status_code",
                "response_time",
                "response_json",
                "error",
            ]
        )

    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        numeric_level = getattr(logging, log_level, logging.INFO)

        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.setLevel(numeric_level)
        logger.addHandler(console_handler)

        # Also log to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"data/logs/benchmark_{timestamp}.log")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info("Logging initialized")

    def generate_decoding_prompt(self) -> str:
        """
        Generate a prompt that includes a list of integers representing a UTF-8 encoded string.

        Creates a 4000-character random string (using hex digits), encodes it to bytes,
        converts to a list of ints, and instructs the model to write Python code that
        decodes this list back into the original string.

        Returns:
            A string prompt.
        """
        logger.info("Generating decoding prompt...")

        # Generate 2000 random bytes as hex -> 4000 hex characters.
        random_string = secrets.token_hex(2000)
        byte_list = list(random_string.encode("utf-8"))
        prompt = (
            "You are given a list of integers representing the UTF-8 encoded bytes of a string. "
            "Write Python code to decode this list back into the original UTF-8 string.\n"
            "The list of integers is as follows:\n"
            f"{byte_list}\n"
            "Please provide the complete Python code as your response."
        )

        logger.debug(f"Generated prompt of length {len(prompt)}")
        return prompt

    def generate_simple_prompt(self) -> str:
        """
        Generate a simple prompt for quick benchmarking.

        Returns:
            A simple prompt string.
        """
        logger.info("Using simple 'hi!' prompt...")
        return "hi!"

    def get_prompt(self, prompt_mode: str = "decode") -> str:
        """
        Get a prompt based on the specified mode.

        Args:
            prompt_mode: The prompt mode to use. Options are:
                - "decode": The standard decoding prompt (default)
                - "simple": A simple "hi!" prompt

        Returns:
            A prompt string.
        """
        if prompt_mode == "simple":
            return self.generate_simple_prompt()
        else:  # Default to decode mode
            return self.generate_decoding_prompt()

    async def benchmark_provider_async(
        self,
        client: LLMClient,
        prompt: str,
        attempts: int = 100,
        max_tokens: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Benchmark a given provider by repeatedly calling the API and measuring tokens per second.

        Args:
            client: The LLM client to use.
            prompt: The prompt string to use for each attempt.
            attempts: Number of attempts.
            max_tokens: Maximum number of tokens to generate (if supported by the provider).

        Returns:
            A BenchmarkResult with performance statistics.
        """
        provider_name = client.name
        logger.info(f"Benchmarking {provider_name} with {attempts} attempts...")

        tps_results = []
        all_token_details = []
        elapsed_times = []
        token_counts = []

        for i in range(attempts):
            logger.info(f"Attempt {i + 1}/{attempts} for {provider_name}")

            # Call the API
            response_data, elapsed = await client.call_async(
                prompt, max_tokens=max_tokens
            )
            elapsed_times.append(elapsed)

            # Get detailed token counts
            token_counts_dict = client.extract_detailed_token_counts(response_data)
            output_token_counts = token_counts_dict["total_tokens"]

            all_token_details.append(token_counts_dict)

            # Calculate tokens per second
            if elapsed > 0:
                tps_results.append(output_token_counts / elapsed)

            # Track token counts
            token_counts.append(output_token_counts)
            logger.info(
                f"Attempt {i + 1} completed: {output_token_counts} tokens in {elapsed:.2f}s"
            )

        # Calculate statistics
        result = BenchmarkResult(
            provider_name=provider_name,
            average_tps=statistics.mean(tps_results),
            median_tps=statistics.median(tps_results),
            average_tokens=statistics.mean(token_counts),
            token_details=all_token_details,
            elapsed_times=elapsed_times,
        )

        logger.info(f"Benchmark completed for {provider_name}")
        logger.info(f"Average TPS: {result.average_tps:.2f}")
        logger.info(f"Median TPS: {result.median_tps:.2f}")
        logger.info(f"Average tokens: {result.average_tokens:.2f}")

        return result

    def benchmark_provider(
        self, client: LLMClient, prompt: str, attempts: int = 100
    ) -> BenchmarkResult:
        """
        Synchronous wrapper for benchmark_provider_async.

        Args:
            client: The LLM client to use.
            prompt: The prompt string to use for each attempt.
            attempts: Number of attempts.

        Returns:
            A BenchmarkResult with performance statistics.
        """
        return asyncio.run(self.benchmark_provider_async(client, prompt, attempts))

    def save_log_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Save the requests log DataFrame to a CSV file.

        Args:
            filename: Optional custom filename. If None, a timestamped filename is generated.

        Returns:
            The filename where the log was saved.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/logs/api_requests_log_{timestamp}.csv"

        # Save the DataFrame to CSV
        self.requests_log.to_csv(filename, index=False)
        logger.info(f"Saved request log to {filename}")
        return filename

    async def run_async(
        self,
        attempts: int = 100,
        prompt_mode: str = "decode",
        max_tokens: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run benchmarks for all configured providers asynchronously.

        Args:
            attempts: Number of API calls to make per provider.
            prompt_mode: The prompt mode to use. Options are:
                - "decode": The standard decoding prompt (default)
                - "simple": A simple "hi!" prompt
            max_tokens: Maximum number of tokens to generate (if supported by the provider).

        Returns:
            Dictionary mapping provider names to benchmark results.
        """
        logger.info(f"Starting benchmark run with {attempts} attempts per provider")

        if max_tokens is not None:
            logger.info(f"Using max_tokens={max_tokens}")

        if prompt_mode == "simple":
            logger.info("Using simple 'hi!' prompt")
        else:
            logger.info("Using standard decoding prompt")

        # Generate the prompt once to be consistent across providers
        prompt = self.get_prompt(prompt_mode)

        # Run benchmarks for each provider
        tasks = []
        provider_names = []
        for provider_config in self.providers:
            try:
                client = get_client(provider_config)
                tasks.append(
                    self.benchmark_provider_async(client, prompt, attempts, max_tokens)
                )
                provider_names.append(provider_config.name)
            except Exception as e:
                logger.error(
                    f"Error setting up benchmark for {provider_config.name}: {str(e)}"
                )

        # Wait for all benchmarks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            provider_name = provider_names[i]
            if isinstance(result, Exception):
                logger.error(f"Error benchmarking {provider_name}: {str(result)}")
            else:
                # Cast the result to BenchmarkResult since we've checked it's not an exception
                benchmark_result = cast(BenchmarkResult, result)
                self.results[provider_name] = benchmark_result

                # Save logs after each provider to prevent data loss
                log_file = self.save_log_to_csv(
                    f"data/results/{provider_name}_requests_log.csv"
                )
                logger.info(f"Request logs saved to: {log_file}")

        # Save final combined log
        final_log_file = self.save_log_to_csv("data/results/all_requests_log.csv")
        logger.info(f"Full request logs saved to: {final_log_file}")

        return self.results

    def print_results(self):
        """Print a summary of benchmark results to the console."""
        if not self.results:
            logger.warning("No benchmark results to print")
            return

        logger.info("\nBenchmark Results:")
        for provider_name, result in self.results.items():
            logger.info(f"{provider_name}:")
            logger.info(f"  Average TPS: {result.average_tps:.2f}")
            logger.info(f"  Median TPS: {result.median_tps:.2f}")
            logger.info(f"  Average Tokens: {result.average_tokens:.2f}")
            logger.info("")
