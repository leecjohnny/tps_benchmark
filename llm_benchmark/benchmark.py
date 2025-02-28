"""
Benchmarking functionality for LLM providers.
"""

import asyncio
import logging
import os
import secrets
import statistics
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from llm_benchmark.client import LLMClient, get_client
from llm_benchmark.models import ProviderConfig

# Set up logging
logger = logging.getLogger("llm_benchmark")


@dataclass
class BenchmarkResult:
    """Results of a benchmark run for a single provider."""

    provider_name: str
    model_name: str
    run_id: str
    average_tps: float
    median_tps: float
    total_tokens: int
    total_elapsed: float
    token_details: List[Dict[str, Any]]
    elapsed_times: List[float]


class BenchmarkRunner:
    """Runner for LLM benchmarks."""

    def __init__(self, providers: List[ProviderConfig]):
        """
        Initialize the benchmark runner.

        Args:
            providers: List of provider configurations to benchmark.
                If None, uses the default providers.
        """
        self.providers = providers
        self.results: Dict[str, BenchmarkResult] = {}

        # Set up logging to file and console
        self._setup_logging()
        self.run_id = str(uuid.uuid4())
        logger.info(f"Run ID: {self.run_id}")

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

        # Create tasks for all attempts
        tasks = []
        for i in range(attempts):
            tasks.append(client.call_async(prompt, max_tokens=max_tokens))

        # Run all attempts concurrently
        logger.info(f"Running {attempts} attempts concurrently for {provider_name}")
        responses = await asyncio.gather(*tasks)

        # Process results
        for i, (response_data, elapsed) in enumerate(responses):
            elapsed_times.append(elapsed)

            # Get detailed token counts
            token_counts_dict = client.extract_detailed_token_counts(response_data)
            output_token_counts = token_counts_dict["completion_tokens"]

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
            model_name=client.config.model,
            run_id=self.run_id,
            average_tps=statistics.mean(tps_results),
            median_tps=statistics.median(tps_results),
            total_tokens=sum(token_counts),
            total_elapsed=sum(elapsed_times),
            token_details=all_token_details,
            elapsed_times=elapsed_times,
        )

        logger.info(f"Benchmark completed for {provider_name}")
        logger.info(f"Average TPS: {result.average_tps:.2f}")
        logger.info(f"Median TPS: {result.median_tps:.2f}")
        logger.info(f"Total tokens: {result.total_tokens}")
        logger.info(f"Total elapsed: {result.total_elapsed:.2f}s")
        return result

    async def run_async(
        self,
        attempts: int,
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
        for provider_config in self.providers:
            try:
                client = get_client(provider_config, self.run_id)
                tasks.append(
                    self.benchmark_provider_async(client, prompt, attempts, max_tokens)
                )
            except Exception as e:
                logger.error(
                    f"Error setting up benchmark for {provider_config.name}: {str(e)}"
                )

        # Wait for all benchmarks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error benchmarking {self.providers[i].name}: {str(result)}"
                )
            else:
                # Cast the result to BenchmarkResult since we've checked it's not an exception
                benchmark_result = cast(BenchmarkResult, result)
                output_results.append(asdict(benchmark_result))

        # Save final combined log
        final_csv_file = f"data/results/all_requests_log_{self.run_id}.csv"
        pd.DataFrame(output_results).to_csv(final_csv_file, index=False)
        logger.info(f"Full request logs saved to: {final_csv_file}")

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
