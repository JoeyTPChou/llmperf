export AWS_ACCESS_KEY_ID=""
export AWS_SECRET_ACCESS_KEY=""
export AWS_REGION_NAME=""

python3 token_benchmark_ray.py \
--model "anthropic.claude-3-sonnet-20240229-v1:0" \
--llm-api "bedrock" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "benchmark_result_outputs" \
--metadata "stream=True" \
--additional-sampling-params '{"temperature": 0.9, "stream": true}' \
--additional-request-body '{"stream": true}' \
