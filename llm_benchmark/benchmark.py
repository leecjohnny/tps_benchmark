"""
Benchmarking functionality for LLM providers.
"""

import asyncio
import logging
import os
import random
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm.asyncio import tqdm

from llm_benchmark.client import LLMClient, get_client
from llm_benchmark.fetch_logs import fetch_logs
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
        self.results: List[BenchmarkResult] = []

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

        Creates a N-character random string (using hex digits), encodes it to bytes,
        converts to a list of ints, and instructs the model to write Python code that
        decodes this list back into the original string.

        Returns:
            A string prompt.
        """
        logger.info("Generating decoding prompt...")

        random_string = secrets.token_hex(500)
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
        attempts: int,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
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
        model_name = client.config.model
        # Create tasks for all attempts
        tasks = []
        for i in range(attempts):
            tasks.append(client.call_async(prompt, max_tokens=max_tokens))

        responses = []
        for f in tqdm.as_completed(
            tasks, total=len(tasks), desc=f"Benchmarking {provider_name} {model_name}"
        ):
            try:
                result = await f
                responses.append(result)
            except Exception as e:
                logger.error(
                    f"Error benchmarking {provider_name} and {model_name}: {str(e)}"
                )
        return responses

    async def run_async(
        self,
        attempts: int,
        prompt_mode: str = "decode",
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Run benchmarks for all configured providers asynchronously.

        Args:
            attempts: Number of API calls to make per provider.
            prompt_mode: The prompt mode to use. Options are:
                - "decode": The standard decoding prompt (default)
                - "simple": A simple "hi!" prompt
            max_tokens: Maximum number of tokens to generate (if supported by the provider).

        Returns:
            DataFrame with benchmark results.
        """
        logger.info(f"Starting benchmark run with {attempts} attempts per provider")

        if max_tokens is not None:
            logger.info(f"Using max_tokens={max_tokens}")

        if prompt_mode == "simple":
            logger.info("Using simple 'hi!' prompt")
        else:
            logger.info("Using standard decoding prompt")

        # Run benchmarks for each provider
        tasks = []
        prompts = [self.get_prompt(prompt_mode) for _ in range(attempts)]
        for provider_config in self.providers:
            try:
                client = get_client(provider_config, self.run_id)
                tasks.append(
                    self.benchmark_provider_async(
                        client, random.choice(prompts), attempts, max_tokens
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error setting up benchmark for {provider_config.name}: {str(e)}"
                )

        # Wait for all benchmarks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"All benchmarks completed for this run {self.run_id}")

        logger.info(f"Fetching logs from cloudflare for this run {self.run_id}")
        _, details = await fetch_logs(self.run_id)
        output_parquet_path = f"data/cf/{self.run_id}/details.zstd.parquet"
        ## make sure the directory exists
        Path(output_parquet_path).parent.mkdir(parents=True, exist_ok=True)
        # Write each detail as a separate JSON line (JSONL format)
        details_rows = []
        for detail in details:
            if detail.metadata is not None:
                detail.metadata = None
            details_rows.append(detail.model_dump())
        df = pd.DataFrame(details_rows)
        df.to_parquet(output_parquet_path, compression="zstd", index=False)
