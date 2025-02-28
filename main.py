"""
Main entry point for the LLM benchmarking tool.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

from llm_benchmark.benchmark import BenchmarkRunner
from llm_benchmark.models import (
    ALL_ANTHROPIC_MODELS,
    ALL_DEEPSEEK_MODELS,
    ALL_GOOGLE_VERTEX_MODELS,
    ALL_MISTRAL_MODELS,
    ALL_MODELS,
    ALL_OPENAI_MODELS,
    ALL_WORKERS_AI_MODELS,
    DEFAULT_PROVIDERS,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark LLM providers")

    parser.add_argument(
        "--attempts",
        type=int,
        default=int(os.environ["BENCHMARK_ATTEMPTS"]),
        help="Number of API calls to make per provider (default set to BENCHMARK_ATTEMPTS environment variable)",
    )

    provider_group = parser.add_mutually_exclusive_group()

    provider_group.add_argument(
        "--all-openai",
        action="store_true",
        help="Benchmark all OpenAI models",
    )

    provider_group.add_argument(
        "--all-anthropic",
        action="store_true",
        help="Benchmark all Anthropic models",
    )

    provider_group.add_argument(
        "--all-google",
        action="store_true",
        help="Benchmark all Google Vertex AI models",
    )

    provider_group.add_argument(
        "--all-deepseek",
        action="store_true",
        help="Benchmark all Deepseek models",
    )

    provider_group.add_argument(
        "--all-workers",
        action="store_true",
        help="Benchmark all Workers AI models",
    )

    provider_group.add_argument(
        "--all-mistral",
        action="store_true",
        help="Benchmark all Mistral models",
    )

    provider_group.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark all models from all providers",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Set the logging level (default: INFO)",
    )

    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["decode", "simple"],
        default="decode",
        help="Prompt mode to use: 'decode' for the standard decoding prompt, 'simple' for a simple 'hi!' prompt (default: decode)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (default: None, which uses the provider's default)",
    )

    return parser.parse_args()


async def async_main():
    """Asynchronous main entry point for the benchmarking tool."""
    # Load environment variables from .env file
    load_dotenv()

    # Parse command line arguments
    args = parse_args()

    # Set log level in environment for the benchmark runner
    os.environ["LOG_LEVEL"] = args.log_level

    # Determine which providers to benchmark
    providers = DEFAULT_PROVIDERS

    if args.all_models:
        providers = ALL_MODELS
        print(f"Benchmarking all {len(providers)} models from all providers")
    elif args.all_openai:
        providers = ALL_OPENAI_MODELS
        print(f"Benchmarking all {len(providers)} OpenAI models")
    elif args.all_anthropic:
        providers = ALL_ANTHROPIC_MODELS
        print(f"Benchmarking all {len(providers)} Anthropic models")
    elif args.all_google:
        providers = ALL_GOOGLE_VERTEX_MODELS
        print(f"Benchmarking all {len(providers)} Google Vertex AI models")
    elif args.all_deepseek:
        providers = ALL_DEEPSEEK_MODELS
        print(f"Benchmarking all {len(providers)} Deepseek models")
    elif args.all_workers:
        providers = ALL_WORKERS_AI_MODELS
        print(f"Benchmarking all {len(providers)} Workers AI models")
    elif args.all_mistral:
        providers = ALL_MISTRAL_MODELS
        print(f"Benchmarking all {len(providers)} Mistral models")
    # Run benchmarks
    runner = BenchmarkRunner(providers=providers)

    # Log the prompt mode and max tokens
    if args.prompt_mode == "simple":
        print(
            f"Using simple 'hi!' prompt with max_tokens={args.max_tokens or 'default'}"
        )
    else:
        print(
            f"Using standard decoding prompt with max_tokens={args.max_tokens or 'default'}"
        )

    await runner.run_async(
        attempts=args.attempts,
        prompt_mode=args.prompt_mode,
        max_tokens=args.max_tokens,
    )


def main():
    """Main entry point for the benchmarking tool."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
