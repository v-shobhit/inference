#!/usr/bin/env python3
"""TensorRT-LLM backend implementation for gpt-oss.

Supports round-robin load balancing across multiple server endpoints.
"""

import asyncio
import json
import logging
import requests
import threading
import time
from typing import List, Dict, Any, Optional, AsyncIterator, Union
import aiohttp
from transformers import AutoTokenizer
from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class TRTLLMBackend(BaseBackend):
    """TensorRT-LLM inference backend using OpenAI-compatible HTTP API.

    Connects to TensorRT-LLM server(s) running the gpt-oss model via
    the /v1/completions endpoint. Supports round-robin load balancing
    across multiple servers.
    """

    def __init__(
        self,
        server_url: Union[str, List[str]] = "http://localhost:30000",
        model_name: str = "gpt-oss-120b",
        tokenizer_name: str = "openai/gpt-oss-120b",
        timeout: int = 1200,
        max_pool_size: int = 2000,
        **kwargs
    ):
        """Initialize TRT-LLM backend.

        Args:
            server_url: URL(s) of TRT-LLM server(s). Can be:
                - Single URL string: "http://localhost:30000"
                - Comma-separated string: "http://host1:30000,http://host2:30000"
                - List of URLs: ["http://host1:30000", "http://host2:30000"]
            model_name: Model name to use in API requests
            tokenizer_name: HuggingFace tokenizer name for re-tokenizing responses
            timeout: Request timeout in seconds
            max_pool_size: Maximum connection pool size per server
            **kwargs: Additional configuration
        """
        # Parse server URLs
        if isinstance(server_url, str):
            # Split by comma and strip whitespace
            self.server_urls = [url.strip() for url in server_url.split(',')]
        else:
            self.server_urls = list(server_url)

        # Ensure URLs have http:// prefix
        self.server_urls = [
            url if url.startswith('http://') or url.startswith('https://')
            else f'http://{url}'
            for url in self.server_urls
        ]

        self.num_servers = len(self.server_urls)
        self.current_server_index = 0
        self._index_lock = threading.Lock()  # Thread-safe round-robin

        config = {
            "server_urls": self.server_urls,
            "model_name": model_name,
            "tokenizer_name": tokenizer_name,
            "timeout": timeout,
            "max_pool_size": max_pool_size,
            **kwargs
        }
        super().__init__(config)
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.sessions = []  # One session per server
        self.tokenizer = None

    def _get_next_server_url(self) -> str:
        """Get the next server URL using thread-safe round-robin selection."""
        with self._index_lock:
            url = self.server_urls[self.current_server_index]
            self.current_server_index = (self.current_server_index + 1) % self.num_servers
            return url

    def initialize(self) -> None:
        """Initialize connections to TRT-LLM server(s) and load tokenizer."""
        if self.initialized:
            logger.warning("Backend already initialized")
            return

        logger.info(f"Initializing TRT-LLM backend with {self.num_servers} server(s):")
        for i, url in enumerate(self.server_urls):
            logger.info(f"  Server {i+1}: {url}")
        logger.info(
            f"Configuring connection pool with max_pool_size={self.max_pool_size} per server")

        # Load tokenizer for re-tokenizing responses
        logger.info(f"Loading tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Create session for each server with connection pooling
        for server_url in self.server_urls:
            session = requests.Session()

            # Increase connection pool size to support high concurrency
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=min(100, self.max_pool_size // 10),
                pool_maxsize=self.max_pool_size,
                max_retries=3,
                pool_block=False
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            self.sessions.append(session)

        # Test connection to first server
        try:
            test_response = self._send_request(
                input_ids=[1, 2, 3],
                max_tokens=5,
                temperature=0.001,
                top_k=1,
                top_p=1.0
            )
            if "error" in test_response:
                raise ConnectionError(
                    f"Failed to connect to TRT-LLM server: {test_response['error']}"
                )
            logger.info(f"Successfully connected to TRT-LLM server(s)")
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize TRT-LLM backend: {e}")
            raise

    def _send_request(
        self,
        input_ids: List[int],
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> Dict[str, Any]:
        """Send a single request to a TRT-LLM server using round-robin.

        Args:
            input_ids: Token IDs for the prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter

        Returns:
            Response dictionary with output_ids, text, and meta_info
        """
        # Build payload for OpenAI-compatible completions API
        payload = {
            "model": self.model_name,
            "prompt": input_ids,  # TRT-LLM accepts token IDs directly
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,  # Always include top_k
            "top_p": top_p,
            "stream": False,
            "min_tokens": 1,
            "skip_special_tokens": False,
        }

        # Select server using round-robin
        server_url = self._get_next_server_url()
        server_index = self.server_urls.index(server_url)
        session = self.sessions[server_index]

        try:
            response = session.post(
                f"{server_url}/v1/completions",
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                # Extract text from OpenAI-compatible response
                text = data.get("choices", [{}])[0].get("text", "")
                usage = data.get("usage", {})

                # Re-tokenize the response text to get output_ids
                # (TRT-LLM doesn't return token_ids in response)
                output_ids = self.tokenizer.encode(
                    text, add_special_tokens=False
                ) if text else []

                return {
                    "output_ids": output_ids,
                    "text": text,
                    "meta_info": {
                        "completion_tokens": usage.get("completion_tokens", len(output_ids)),
                        "prompt_tokens": usage.get("prompt_tokens", len(input_ids)),
                        "total_tokens": usage.get("total_tokens", 0),
                        "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
                        "server_url": server_url,
                    }
                }
            else:
                logger.error(
                    f"Request to {server_url} failed with status {response.status_code}: {response.text}"
                )
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {server_url} failed: {e}")
            return {"error": str(e)}

    def generate(
        self,
        prompts: List[List[int]],
        max_tokens: int = 100,
        temperature: float = 0.001,
        top_k: int = 1,
        top_p: float = 1.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of token ID sequences
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            **kwargs: Additional parameters (ignored)

        Returns:
            List of response dictionaries with keys:
                - output_ids: List of generated token IDs
                - output_text: Generated text (if available)
                - metadata: Additional metadata (latencies, etc.)
        """
        if not self.initialized:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first.")

        results = []
        for prompt_ids in prompts:
            start_time = time.time()
            response = self._send_request(
                input_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            end_time = time.time()
            latency = end_time - start_time

            # Extract output_ids from response
            output_ids = []
            output_text = ""
            if "error" not in response:
                output_ids = response.get("output_ids", [])
                output_text = response.get("text", "")

            result = {
                "output_ids": output_ids,
                "output_text": output_text,
                "metadata": {
                    "latency": latency,
                    "completion_tokens": response.get("meta_info", {}).get(
                        "completion_tokens", len(output_ids)
                    ),
                    "error": response.get("error"),
                }
            }
            results.append(result)

        return results

    async def generate_stream(
        self,
        input_ids: List[int],
        max_tokens: int = 100,
        temperature: float = 0.001,
        top_k: int = 1,
        top_p: float = 1.0,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate response with streaming support.

        Yields incremental responses as tokens are generated.

        Args:
            input_ids: Token IDs for the prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter

        Yields:
            Dict with:
                - delta_token_ids: List of new token IDs in this chunk
                - delta_text: New text in this chunk
                - is_first_token: True if this is the first token
                - is_finished: True if generation is complete
                - accumulated_token_ids: All tokens generated so far
                - metadata: Additional info (TTFT, completion_tokens, etc.)
        """
        if not self.initialized:
            raise RuntimeError(
                "Backend not initialized. Call initialize() first.")

        # Select server using round-robin
        server_url = self._get_next_server_url()

        # Build payload for OpenAI-compatible completions API with streaming
        payload = {
            "model": self.model_name,
            "prompt": input_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,  # Always include top_k to match SGLang behavior
            "top_p": top_p,
            "stream": True,
            "min_tokens": 1,
            "skip_special_tokens": False,
        }

        start_time = time.time()
        first_token_time = None
        accumulated_text = ""
        accumulated_token_ids = []
        is_first = True

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server_url}/v1/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"Streaming request to {server_url} failed: {response.status} - {error_text}")
                        yield {
                            "delta_token_ids": [],
                            "delta_text": "",
                            "is_first_token": False,
                            "is_finished": True,
                            "accumulated_token_ids": [],
                            "error": f"HTTP {response.status}: {error_text}",
                            "metadata": {}
                        }
                        return

                    # Read streaming response (SSE format)
                    async for line in response.content:
                        if not line:
                            continue

                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith('data:'):
                            continue

                        try:
                            json_str = line_str[5:].strip()
                            if json_str == '[DONE]':
                                break

                            chunk = json.loads(json_str)

                            # Extract delta text from OpenAI-compatible response
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue

                            choice = choices[0]
                            delta_text = choice.get("text", "")
                            finish_reason = choice.get("finish_reason")
                            is_finished = finish_reason is not None

                            # Accumulate text
                            if delta_text:
                                accumulated_text += delta_text

                            # Re-tokenize to get token IDs
                            # Note: This is approximate for streaming since we're
                            # tokenizing incrementally
                            if delta_text:
                                # Tokenize the new text
                                new_tokens = self.tokenizer.encode(
                                    delta_text, add_special_tokens=False
                                )
                                delta_token_ids = new_tokens
                                accumulated_token_ids.extend(delta_token_ids)
                            else:
                                delta_token_ids = []

                            # Mark first token timing
                            if is_first and (delta_token_ids or delta_text):
                                first_token_time = time.time()
                                is_first = False

                            yield {
                                "delta_token_ids": delta_token_ids,
                                "delta_text": delta_text,
                                "is_first_token": (first_token_time is not None and len(accumulated_token_ids) <= len(delta_token_ids)),
                                "is_finished": is_finished,
                                "accumulated_token_ids": accumulated_token_ids.copy(),
                                "accumulated_text": accumulated_text,
                                "metadata": {
                                    "ttft_ms": (first_token_time - start_time) * 1000 if first_token_time else None,
                                    "latency_ms": (time.time() - start_time) * 1000,
                                    "finish_reason": finish_reason,
                                    "server_url": server_url,
                                }
                            }

                            if is_finished:
                                break

                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse streaming chunk: {e}")
                            continue

        except asyncio.TimeoutError:
            logger.error(f"Streaming request to {server_url} timed out after {self.timeout}s")
            yield {
                "delta_token_ids": [],
                "delta_text": "",
                "is_first_token": False,
                "is_finished": True,
                "accumulated_token_ids": accumulated_token_ids,
                "error": "Timeout",
                "metadata": {}
            }
        except Exception as e:
            logger.error(f"Streaming request to {server_url} failed: {e}", exc_info=True)
            yield {
                "delta_token_ids": [],
                "delta_text": "",
                "is_first_token": False,
                "is_finished": True,
                "accumulated_token_ids": accumulated_token_ids,
                "error": str(e),
                "metadata": {}
            }

    def cleanup(self) -> None:
        """Clean up backend resources."""
        for session in self.sessions:
            if session:
                session.close()
        self.sessions = []
        self.tokenizer = None
        self.initialized = False
        logger.info("TRT-LLM backend cleaned up")
