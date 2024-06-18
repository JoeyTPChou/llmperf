"""
Reference:
    * Conversation APIs Python example: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
    * converse Request and Response: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html 
    * converse_stream Request and Response: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html 

Note:
    * The ttft (time-to-first-token) and time_to_next_token are actually time-to-first-word and time_to_next_word
      because Bedrock doesn't provide APIs to return tokens.


TODO(Joey Chou):
    * Add error and exception handling to the llm_request
        ** converse: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html 
        ** converse_stream: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html#API_runtime_ConverseStream_ResponseElements
"""
import io
import json
import os
import time
from typing import Any, Dict

import boto3
import ray
from transformers import LlamaTokenizerFast

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class BedrockClient(LLMClient):
    """Client for Bedrock Chat Completion API."""

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            raise ValueError("AWS_ACCESS_KEY_ID must be set.")
        if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            raise ValueError("AWS_SECRET_ACCESS_KEY must be set.")
        if not os.environ.get("AWS_REGION_NAME"):
            raise ValueError("AWS_REGION_NAME must be set.")

        prompt = request_config.prompt
        prompt, prompt_len = prompt

        model_id = request_config.model

        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=os.environ.get("AWS_REGION_NAME"))

        # User should make sure the sampling_params follows the InferenceConfiguration reference:
        #   https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html#API_runtime_InferenceConfiguration_Contents
        inference_config = request_config.sampling_params

        if inference_config and "max_tokens" in inference_config:
            inference_config["maxTokens"] = inference_config["max_tokens"]
            del inference_config["max_tokens"]

        if inference_config and "stream" in inference_config:
            stream_model = inference_config["stream"]
            del inference_config["stream"]
        else:
            stream_model = False

        # Get prompt
        messages = [
            {"role": "user", "content": [{"text": prompt}]},
        ]

        # Parse additional_request_body
        # If user wants to have additional request body, please make sure the additional_request_body dictionary
        # follow the reference:
        #   converse: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestBody
        #   converse_stream: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html#API_runtime_ConverseStream_RequestBody
        additional_request_body = request_config.additional_request_body

        # Prepare API inputs
        api_inputs = {
            "modelId": model_id,
            "messages": messages
        }

        if inference_config:
            api_inputs["inferenceConfig"] = inference_config

        if additional_request_body is not None:
            # Get system prompt if given 
            system = "" if "system" not in additional_request_body else additional_request_body["system"]
            system_prompts = [{"text": system}] if system else None
                
            additionalModelRequestFields = "" if "additionalModelRequestFields" not in additional_request_body else additional_request_body["additionalModelRequestFields"] 
            additionalModelResponseFieldPaths = "" if "additionalModelResponseFieldPaths" not in additional_request_body else additional_request_body["additionalModelResponseFieldPaths"]
            toolConfig = "" if "toolConfig" not in additional_request_body else additional_request_body["toolConfig"] 

            if system_prompts:
                api_inputs["system"] = system_prompts

            if additionalModelRequestFields:
                api_inputs["additionalModelRequestFields"] = additionalModelRequestFields 

            if additionalModelResponseFieldPaths:
                api_inputs["additionalModelResponseFieldPaths"] =  additionalModelResponseFieldPaths

            if toolConfig:
                api_inputs["toolConfig"] = tool_config

        #----------
        role = ""
        generated_text = ""
        message_stop = ""
        usage = {}
        metrics = {}

        benchmark_metrics = {}
        benchmark_metrics[common_metrics.ERROR_CODE] = None
        benchmark_metrics[common_metrics.ERROR_MSG] = ""

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = None
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        try:
            if stream_model:
                response = bedrock_client.converse_stream(**api_inputs)
                stream = response.get('stream')
                for event in stream:
                    if 'messageStart' in event:
                        role = event['messageStart']['role']

                    if 'contentBlockDelta' in event:
                        # This is similar as openai_chat_completions_client.py
                        if not ttft:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()

                        generated_text += event['contentBlockDelta']['delta']['text']

                    if 'messageStop' in event:
                        message_stop = event['messageStop']['stopReason']

                    if 'metadata' in event:
                        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
                        metadata = event['metadata']
                        if 'usage' in metadata:
                            usage = metadata["usage"]
                        if 'metrics' in event['metadata']:
                            metrics = metadata["metrics"]
            else:
                response = bedrock_client.converse(**api_inputs)

                # Non-stream mode doesn't have time-to-next-token
                time_to_next_token = []
                ttft = 0

                role = response['output']['message']["role"]
                generated_text = response['output']['message']["content"][0]["text"]

                message_stop = response["stopReason"]
                usage = response["usage"]
                metrics = response["metrics"]
                token_usage = response['usage']

            tokens_received = usage["outputTokens"] 
            total_request_time = time.monotonic() - start_time
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            benchmark_metrics[common_metrics.ERROR_MSG] = error_msg
            benchmark_metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        benchmark_metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token) # This should be same as benchmark_metrics[common_metrics.E2E_LAT]. Leave it here for now
        benchmark_metrics[common_metrics.TTFT] = ttft
        benchmark_metrics[common_metrics.E2E_LAT] = total_request_time
        benchmark_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        benchmark_metrics[common_metrics.NUM_TOTAL_TOKENS] = usage["totalTokens"]
        benchmark_metrics[common_metrics.NUM_OUTPUT_TOKENS] = usage["outputTokens"]
        benchmark_metrics[common_metrics.NUM_INPUT_TOKENS] = usage["inputTokens"]

        return benchmark_metrics, generated_text, request_config
