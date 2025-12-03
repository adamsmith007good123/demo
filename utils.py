import asyncio
import os
import time

import requests
from openai import AsyncOpenAI

API_KEY = os.getenv("API_KEY")


def clean_ping_base(base_url: str):
    if not base_url:
        return None
    base = base_url.rstrip("/")
    if base.endswith("v1"):
        base = base[:-2]
        base = base.rstrip("/")
    return base

def health_check_with_retry(base_url, api_key, max_wait_seconds=600, delay=5):
    if not base_url:
        return False
    ping_base = clean_ping_base(base_url)
    if not ping_base:
        return False
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    start = time.time()
    while time.time() - start < max_wait_seconds:
        try:
            resp = requests.get(f"{ping_base}/ping", headers=headers, timeout=60)
            print(f"Health check response from {ping_base}/ping: {resp.status_code}")
            if resp.status_code == 200:
                return True
        except Exception as e:
            print(e)
        time.sleep(delay)
    return False

async def warmup_single(name, host, status_placeholder):
    if not host:
        status_placeholder.error(f"{name}: missing host URL.")
        return False
    status_placeholder.info(f"{name}: starting on serverless RunPod.")
    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, health_check_with_retry, host, API_KEY)
    if ok:
        status_placeholder.success(f"{name}: ready.")
    else:
        status_placeholder.error(f"{name}: failed to start within 10 minutes.")
    return ok

async def warmup_in_parallel(models):
    tasks = [warmup_single(m["name"], m["host"], m["placeholder"]) for m in models]
    if not tasks:
        return False
    results = await asyncio.gather(*tasks)
    return all(results)


async def run_request(base_url, messages, temperature, max_tokens, use_reasoning):
    thinking = ""
    answer = ""
    token_count = 0
    in_think = False
    buffer = ""
    start = time.time()

    async with AsyncOpenAI(base_url=base_url, api_key=API_KEY) as client:
        async for chunk in await client.chat.completions.create(
            model="anything",
            stream=True,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": use_reasoning}},
        ):
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            token_count += len(delta)
            buffer += delta

            if "<think>" in buffer:
                in_think = True
                buffer = buffer.replace("<think>", "")
            if "</think>" in buffer:
                in_think = False
                before, after = buffer.split("</think>", 1)
                thinking += before
                buffer = after

            if in_think:
                thinking += buffer
                buffer = ""
            else:
                answer += buffer
                buffer = ""

            tps = token_count / (time.time() - start + 1e-5)
            elapsed_time = time.time() - start

            yield thinking, answer, token_count, tps, elapsed_time
