# LLM API Benchmarking Tool

A tool for benchmarking different Language Model API providers through the Cloudflare AI Gateway.

## Features

- Benchmark multiple LLM API providers including OpenAI, Anthropic, Google Vertex AI, DeepSeek, Workers AI, and Mistral
- Support for multiple models from each provider
- Track tokens per second (TPS) for each provider
- Detailed token usage metrics (prompt tokens, completion tokens, and total tokens)
- Use actual token counts from API responses when available
- Detailed logging of all requests and responses to CSV files
- Environment variable configuration with dotenv support
- Structured logging to both console and file
- Asynchronous API calls for improved performance
- Comprehensive error handling with detailed error logs

## How It Works

This tool uses the Cloudflare AI Gateway as a unified interface to access multiple LLM providers. The Cloudflare AI Gateway provides a single API endpoint that can route requests to different LLM providers based on the request payload.

For each benchmark run:
1. The tool generates a consistent prompt for all providers
2. It sends the prompt to each provider through the Cloudflare AI Gateway asynchronously
3. It measures the response time and token generation speed
4. It extracts detailed token usage metrics (prompt, completion, and total tokens)
5. It extracts and logs the response data
6. If a request fails, it logs detailed error information including HTTP status codes and response content

### Prompt Modes

The tool supports different prompt modes for benchmarking:

- **Decode Mode** (default): Generates a complex prompt that includes a list of integers representing a UTF-8 encoded string. This is useful for benchmarking the model's ability to process and generate longer responses.

- **Simple Mode**: Uses a simple "hi!" prompt with an optional token limit. This is useful for quick benchmarking or when you want to test with minimal token usage.

You can select the prompt mode using the `--prompt-mode` command-line option, and optionally limit the response length with `--max-tokens`.

## Supported Models

### OpenAI
- gpt-4o
- gpt-4o-mini
- o1
- o1-mini
- o3-mini
- gpt-4-turbo
- gpt-4
- gpt-3.5-turbo

### Anthropic
- claude-3-7-sonnet-latest
- claude-3-5-haiku-latest
- claude-3-5-sonnet-latest
- claude-3-opus-latest
- claude-3-haiku-20240307
- claude-3-sonnet-20240229
- claude-2.1

### Google Vertex AI
- gemini-1.0-pro
- gemini-1.5-pro
- gemini-1.5-flash
- gemini-2.0-flash-lite
- gemini-2.0-flash
- gemini-2.0-pro-exp-02-05
- gemini-2.0-flash-thinking-exp-01-21

### Deepseek
- deepseek-chat
- deepseek-reasoner

### Workers AI
- @cf/meta/llama-3.1-70b-instruct
- @cf/meta/llama-3.3-70b-instruct-fp8-fast
- @cf/deepseek-ai/deepseek-r1-distill-qwen-32b

### Mistral
- mistral-large-latest
- ministral-8b-latest
- mistral-medium-latest
- mistral-small-latest

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm_benchmark
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

3. Set up your environment variables by copying the example file and filling in your credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and configuration
   ```

## Cloudflare AI Gateway Configuration

To use this tool, you need to set up a Cloudflare AI Gateway. The gateway provides a unified interface to access multiple LLM providers.

1. Sign up for a Cloudflare account if you don't have one
2. Set up an AI Gateway in the Cloudflare dashboard
3. Configure the providers you want to use in the gateway
4. Get your Cloudflare account ID and gateway ID
5. Add these to your `.env` file along with the API keys for each provider

## Usage

Run the benchmarking tool:

```bash
# Run as a module
python -m llm_benchmark

# Or use the installed command
llm-benchmark
```

### Command Line Options

```
usage: llm-benchmark [-h] [--attempts ATTEMPTS]
                    [--providers PROVIDERS [PROVIDERS ...] | --all-openai | --all-anthropic | --all-google | --all-deepseek | --all-workers | --all-mistral | --all-models]
                    [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                    [--parallel] [--prompt-mode {decode,simple}] [--max-tokens MAX_TOKENS]

Benchmark LLM providers

options:
  -h, --help            show this help message and exit
  --attempts ATTEMPTS   Number of API calls to make per provider (default: 100)
  --providers PROVIDERS [PROVIDERS ...]
                        Specific providers to benchmark (default: one model from each provider)
  --all-openai          Benchmark all OpenAI models
  --all-anthropic       Benchmark all Anthropic models
  --all-google          Benchmark all Google Vertex AI models
  --all-deepseek        Benchmark all Deepseek models
  --all-workers         Benchmark all Workers AI models
  --all-mistral         Benchmark all Mistral models
  --all-models          Benchmark all models from all providers
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level (default: INFO)
  --parallel            Run benchmarks in parallel (default: false)
  --prompt-mode {decode,simple}
                        Prompt mode to use: 'decode' for the standard decoding prompt, 'simple' for a simple 'hi!' prompt (default: decode)
  --max-tokens MAX_TOKENS
                        Maximum number of tokens to generate (default: None, which uses the provider's default)
```

### Examples

```bash
# Benchmark all OpenAI models
python -m llm_benchmark --all-openai --attempts 50

# Benchmark all models from all providers
python -m llm_benchmark --all-models --attempts 10

# Benchmark specific providers
python -m llm_benchmark --providers openai anthropic --attempts 50

# Benchmark with debug logging
python -m llm_benchmark --all-mistral --log-level DEBUG

# Use simple prompt mode with max tokens limit
python -m llm_benchmark --prompt-mode simple --max-tokens 50

# Run benchmarks in parallel with simple prompt
python -m llm_benchmark --parallel --prompt-mode simple --max-tokens 100
```

### Environment Variables

Configure the benchmarking tool with the following environment variables in your `.env` file:

- `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare account ID
- `CLOUDFLARE_GATEWAY_ID`: Your Cloudflare gateway ID
- `OPENAI_TOKEN`, `ANTHROPIC_TOKEN`, etc.: API keys for each provider
- `BENCHMARK_ATTEMPTS`: Number of API calls to make per provider (default: 100)
- `LOG_LEVEL`: Logging level (default: INFO)

## Project Structure

The project is organized into several modules:

- `models.py`: Data classes for provider configurations
- `client.py`: Abstract base class for API clients and implementations
- `benchmark.py`: Benchmarking functionality
- `__main__.py`: Command-line interface

## Log Files

The tool generates several files:
- Individual provider logs (e.g., `openai_requests_log.csv`)
- A combined log of all requests (`all_requests_log.csv`)
- A benchmark log file with detailed information (`benchmark_TIMESTAMP.log`)

The benchmark logs include:
- Average tokens per second (TPS)
- Median tokens per second
- Average total tokens
- Breakdown of prompt tokens and completion tokens
- Comparison between API-reported token counts and estimated token counts

## Extending

To add a new provider:

1. Create a new provider configuration class in `models.py`
2. Implement a client class in `client.py`
3. Add the client to the factory function in `client.py`
4. Add the provider to the default providers list in `models.py`

## API Format

The tool uses the Cloudflare AI Gateway API format. Here's an example of the request format:

```json
[
  {
    "provider": "workers-ai",
    "endpoint": "@cf/meta/llama-3.1-8b-instruct",
    "headers": {
      "Authorization": "Bearer YOUR_API_KEY",
      "Content-Type": "application/json"
    },
    "query": {
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "Your prompt here"
        }
      ],
      "max_tokens": 512
    }
  }
]
```

Different providers may have slightly different query formats, which are handled by the provider-specific configuration classes.
