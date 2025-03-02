import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from cloudflare import AsyncCloudflare
from cloudflare.types.ai_gateway.log_list_params import Filter
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
load_dotenv()
client = AsyncCloudflare(
    api_token=os.environ["WORKERS_AI_TOKEN"],
    max_retries=20,
)
ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
GATEWAY_ID = os.environ["CLOUDFLARE_GATEWAY_ID"]


async def fetch_log_details(log_id: str):
    log = await client.ai_gateway.logs.get(
        log_id,
        account_id=ACCOUNT_ID,
        gateway_id=GATEWAY_ID,
    )
    return log


async def fetch_logs(run_id: Optional[str] = None):
    all_logs = []

    # Prepare filters based on whether run_id is provided
    filters = [Filter(key="success", operator="eq", value=[True])]
    if run_id is not None:
        filters.append(Filter(key="metadata.value", operator="eq", value=[run_id]))
    filter_json = json.dumps(filters)
    initial_page = await client.ai_gateway.logs.list(
        gateway_id=GATEWAY_ID,
        account_id=ACCOUNT_ID,
        order_by="created_at",
        order_by_direction="desc",
        filters=filter_json,  # the typers from the sdk is incorrect # type: ignore
        per_page=50,  # Maximize results per page
    )
    all_logs.extend(initial_page.result)

    # Check if result_info exists and has total_count
    total_count = 0
    if initial_page.result_info and hasattr(initial_page.result_info, "total_count"):
        total_count = initial_page.result_info.total_count
        print(f"Total results: {total_count}")

    total_pages = (total_count // 50) + 1

    # Generate all tasks first
    tasks = []
    for page in range(2, total_pages + 1):
        task = client.ai_gateway.logs.list(
            gateway_id=GATEWAY_ID,
            account_id=ACCOUNT_ID,
            filters=filter_json,  # the typers from the sdk is incorrect  # type: ignore
            per_page=50,  # Maximize results per page
            page=page,
        )
        tasks.append(task)

    for task in tqdm.as_completed(tasks, desc="Fetching log pages"):
        result = await task
        all_logs.extend(result.result)

    # Fetch details for each log
    details_tasks = [fetch_log_details(log.id) for log in all_logs]
    details_results = []
    for task in tqdm.as_completed(details_tasks, desc="Fetching log details"):
        result = await task
        details_results.append(result)

    return all_logs, details_results


def main(run_id: Optional[str] = None):
    logs, details = asyncio.run(fetch_logs(run_id))
    if run_id is not None:
        ## make run_id folder under data/cf
        Path(f"data/cf/{run_id}").mkdir(parents=True, exist_ok=True)
        logs_path = Path(f"data/cf/{run_id}/logs.jsonl")
        details_path = Path(f"data/cf/{run_id}/details.jsonl")
    else:
        logs_path = Path("data/cf/all_logs.jsonl")
        details_path = Path("data/cf/all_details.jsonl")
    # Write each log as a separate JSON line (JSONL format)
    with open(logs_path, "w") as f:
        for log in logs:
            if log.metadata is not None:
                log.metadata = None
            f.write(log.model_dump_json() + "\n")

    # Write each detail as a separate JSON line (JSONL format)
    with open(details_path, "w") as f:
        for detail in details:
            if detail.metadata is not None:
                detail.metadata = None
            f.write(detail.model_dump_json() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=False)
    args = parser.parse_args()
    main(args.run_id)
