CREATE TEMP TABLE input_data as (select * from 'data/cf/*/details.jsonl' UNION select * from 'data/cf/*/details.zstd.parquet');

CREATE TEMP TABLE all_details AS SELECT * FROM input_data where coalesce(response_head, '') != '';

CREATE TEMP TABLE parsed_data AS (SELECT
    ad.* EXCLUDE (tokens_out, tokens_in),
    CASE
        WHEN provider IN ('deepseek', 'openai', 'mistral') THEN (response_head::json->'usage'->> 'completion_tokens')::int
        when provider = 'anthropic' then (response_head::json->'usage'->>'output_tokens')::int
        when provider = 'google-vertex-ai' then (response_head::json->'usageMetadata'->>'candidatesTokenCount')::int
        else tokens_out
    end as tokens_out,
    CASE
        WHEN provider IN ('deepseek', 'openai', 'mistral') THEN (response_head::json->'usage'->>'prompt_tokens')::int
        when provider = 'anthropic' then (response_head::json->'usage'->>'input_tokens')::int
        when provider = 'google-vertex-ai' then (response_head::json->'usageMetadata'->>'promptTokenCount')::int
        else tokens_in
    end as tokens_in,
    CASE
        WHEN provider IN ('deepseek', 'openai', 'mistral') THEN response_head::json-> 'choices' -> 0 ->> 'finish_reason'
        when provider = 'anthropic' then response_head::json->>'stop_reason'
        when provider = 'google-vertex-ai' then response_head::json->'candidates'->0->>'finishReason'
    end as finish_reason_parsed,
    CASE WHEN lower(finish_reason_parsed) IN ('stop', 'end_turn') THEN 'stop'
    WHEN lower(finish_reason_parsed) IN ('length', 'max_tokens') THEN 'length'
    END as finish_reason,
    (timings::json -> 'total')::float as duration_ms,
    (timings::json -> 'latency')::float  as cloudflare_gateway_latency_ms,
    duration_ms - cloudflare_gateway_latency_ms as origin_duration_ms

from all_details ad);

COPY (select * from parsed_data) TO 'data/parquets/processed_details.zstd.parquet' WITH (FORMAT PARQUET, COMPRESSION ZSTD);

COPY (select model, provider, tokens_in, tokens_out, duration_ms, cloudflare_gateway_latency_ms,  origin_duration_ms, finish_reason, sha256(request_head) as request_hash from parsed_data) TO '~/Downloads/tps_analysis.csv' (HEADER, DELIMITER ',');

-- statss

-- exclude deepseek; calculate new column for output_tokens_per_second, exclude rows with tokens_in < 50

CREATE TEMP TABLE model_stats as (
with stats as (SELECT
    model,
    provider,
    tokens_in,
    tokens_out,
    origin_duration_ms,
    tokens_out / origin_duration_ms * 1000 as output_tokens_per_second,
    origin_duration_ms / tokens_out as time_per_output_token
FROM parsed_data
WHERE provider != 'deepseek'
AND tokens_in >= 50
ORDER BY output_tokens_per_second DESC)

SELECT
    model,
    provider,
    count(*) as samples_num,
    sum(tokens_in) as total_tokens_in,
    sum(tokens_out) as total_tokens_out,
    sum(origin_duration_ms) as total_origin_duration_ms,
    sum(tokens_out) / sum(origin_duration_ms) * 1000 as avg_output_tokens_per_second,
    percentile_cont(0.5) within group (order by output_tokens_per_second) as median_output_tokens_per_second,
    -- time per output token
    sum(origin_duration_ms) / sum(tokens_out) as avg_time_per_output_token,
    percentile_cont(0.5) within group (order by time_per_output_token) as median_time_per_output_token,
    (avg_output_tokens_per_second - 1.96 * stddev_samp(output_tokens_per_second) / sqrt(count(*)) ) as lower_ci,
    --upper
    (avg_output_tokens_per_second + 1.96 * stddev_samp(output_tokens_per_second) / sqrt(count(*)) ) as upper_ci,
    -- cost per million tokens assuming GPU costs of $2.5/hr per GPU, 8 GPUs per node, with batch size across the node
    -- ((2.5 * 8 / 3600) / median_output_tokens_per_second * 1000000)/1 as cpm_batch_size_1,
    -- ((2.5 * 8 / 3600) / median_output_tokens_per_second * 1000000)/2 as cpm_batch_size_2,
    -- ((2.5 * 8 / 3600) / median_output_tokens_per_second * 1000000)/4 as cpm_batch_size_4,
    -- ((2.5 * 8 / 3600) / median_output_tokens_per_second * 1000000)/8 as cpm_batch_size_8,
    -- ((2.5 * 8 / 3600) / median_output_tokens_per_second * 1000000)/16 as cpm_batch_size_16,
    -- ((2.5 * 8 / 3600) / median_output_tokens_per_second * 1000000)/32 as cpm_batch_size_32
FROM stats
GROUP BY model, provider
ORDER BY median_output_tokens_per_second);

COPY (select * from model_stats) TO '~/Downloads/model_stats.csv' (HEADER, DELIMITER ',');
