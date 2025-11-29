import os
import time
import requests
import openai
import asyncio
import streamlit as st

st.set_page_config(layout="wide")

TPRO_WITH_EAGLE_HOST = os.getenv("TPRO_WITH_EAGLE_HOST")
TPRO_HOST = os.getenv("TPRO_HOST")
QWEN_HOST = os.getenv("QWEN_HOST")
API_KEY = os.getenv("API_KEY")

PROMPTS_DIR = "prompts"
prompt_presets = {}
for fname in os.listdir(PROMPTS_DIR):
    if fname.endswith(".txt"):
        label = fname.replace(".txt", "").replace("_", " ")
        with open(os.path.join(PROMPTS_DIR, fname), "r", encoding="utf-8") as f:
            prompt_presets[label] = f.read().strip()

default_preset_index = 0
for i, key in enumerate(prompt_presets.keys()):
    if key.lower().strip() == "code":
        default_preset_index = i
        break

col_left_sel, col_right_sel = st.columns(2)
with col_left_sel:
    left_option = st.selectbox("Left model", ["t pro 32b with eagle"])
    left_label = "t pro 32b with eagle"
    left_host = TPRO_WITH_EAGLE_HOST
with col_right_sel:
    right_option = st.selectbox("Right model", ["qwen 32b", "t pro 32b"], index=0)
    if right_option == "qwen 32b":
        right_label = "qwen 32b"
        right_host = QWEN_HOST
    else:
        right_label = "t pro 32b"
        right_host = TPRO_HOST

row1 = st.columns([1, 1, 1, 1])
with row1[0]:
    preset = st.selectbox("Preset", list(prompt_presets.keys()), index=default_preset_index)
with row1[1]:
    temperature = st.number_input("Temperature", 0.0, 1.0, 0.0, 0.1)
with row1[2]:
    max_tokens = st.number_input("Tokens", 1, 4096, 512, 1)
with row1[3]:
    reasoning_choice = st.selectbox("Reasoning", ["No reasoning", "Use reasoning"])
    use_reasoning = reasoning_choice == "Use reasoning"

prompt = st.text_area("Prompt", value=prompt_presets[preset])

row2 = st.columns([1])
with row2[0]:
    generate = st.button("Generate", use_container_width=True)

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
            resp = requests.get(f"{ping_base}/ping", headers=headers, timeout=10)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
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

async def run_request(base_url, label, container, prompt, temperature, max_tokens, use_reasoning):
    client = openai.Client(base_url=base_url, api_key=API_KEY)
    speed_area = container.empty()
    thinking_area = container.empty()
    response_area = container.empty()

    thinking = ""
    answer = ""
    token_count = 0
    in_think = False
    buffer = ""
    start = time.time()

    try:
        resp = client.chat.completions.create(
            model="anything",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": use_reasoning}}
        )

        for chunk in resp:
            await asyncio.sleep(0)
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
            speed_area.markdown(f"**{label} — chars/s:** `{tps:.2f}`")

            if thinking:
                thinking_area.markdown(
                    f"""
                    <div style="border:1px solid #999;padding:0.5em;border-radius:8px;margin-bottom:0.5em;">
                        <strong>💭 {label} thinking:</strong><br>{thinking.strip()}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
            if answer:
                response_area.markdown(f"**{label} — Answer:**\n\n{answer.strip()}")

    except Exception as e:
        container.error(f"{label} — Error: {e}")

async def run_both():
    left_col, right_col = st.columns(2)
    await asyncio.gather(
        run_request(left_host, left_label, left_col, prompt, temperature, max_tokens, use_reasoning),
        run_request(right_host, right_label, right_col, prompt, temperature, max_tokens, use_reasoning),
    )

if generate and prompt.strip():
    status_cols = st.columns(2)
    left_status = status_cols[0].empty()
    right_status = status_cols[1].empty()
    models_to_warm = []
    if left_host:
        models_to_warm.append({"name": f"Left ({left_label})", "host": left_host, "placeholder": left_status})
    if right_host:
        models_to_warm.append({"name": f"Right ({right_label})", "host": right_host, "placeholder": right_status})
    st.info(
        "Selected models are hosted on serverless RunPod. Cold starts can take up to ~5 minutes."
    )
    with st.spinner("Starting selected models..."):
        ok = asyncio.run(warmup_in_parallel(models_to_warm))
        ok = asyncio.run(warmup_in_parallel(models_to_warm))
        ok = asyncio.run(warmup_in_parallel(models_to_warm))
    if ok:
        asyncio.run(run_both())
