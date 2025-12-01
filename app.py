import asyncio
import os
from pathlib import Path
import random

import streamlit as st
from streamlit_extras.bottom_container import bottom

from utils import run_request, warmup_in_parallel

MODEL_HOSTNAME = {
    "T-pro 2.0 32B + EAGLE": os.getenv("TPRO_WITH_EAGLE_HOST"),
    "T-pro 2.0 32B": os.getenv("TPRO_HOST"),
    "Qwen3 32B": os.getenv("QWEN_HOST"),
}


st.set_page_config(layout="wide")

st.markdown(
    """
<style>
    .st-key-preset_buttons {
        align-items: center;
    }
    .st-key-preset_buttons button {
        min-height: 1rem;
    }
    [data-testid="stBottomBlockContainer"] .stVerticalBlock {
        gap: 0.5rem;
    }
    [data-testid="stHeaderActionElements"] {
        display: none;
    }
    [data-testid="stHeadingWithActionElements"] h1 {
        font-size: 1.8rem;
    }
    [data-testid="stHeadingWithActionElements"] h2 {
        font-size: 1.4rem;
    }
    [data-testid="stHeadingWithActionElements"] h3 {
        font-size: 1.25rem;
    }
    [data-testid="stHeadingWithActionElements"] h4 {
        font-size: 1.1rem;
    }
    [data-testid="stHeadingWithActionElements"] h5 {
        font-size: 1.05rem;
    }
    .stChatInput div {
        min-height: 150px
    }
    .stChatInput div textarea {
        height: 100%
    }
    .stAppToolbar > div > :first-child::after {
        content: "T-pro 2.0 Interactive Demo";
        font-weight: 500;
        margin-left: 20px;
    }
    .st-key-right_container .stHorizontalBlock {
        align-items: center;
    }
    .st-key-left_container .stHorizontalBlock {
        align-items: center;
    }

    [data-testid="stSidebarUserContent"] .stAlert p {
        font-size: 0.8rem;
    }
    [data-testid="stSidebarUserContent"] [data-testid="stAlertContentWarning"] > div {
        align-items: center;
        gap: 0.8rem;
    }

</style>
""",
    unsafe_allow_html=True,
)

if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "input_preset" not in st.session_state:
    st.session_state.input_preset = ""
if "last_state" not in st.session_state:
    st.session_state.last_state = {
        "model1": None,
        "model2": None,
    }
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False


async def run_model_response(container, model_key):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for conv in st.session_state.conversations:
        messages.append({"role": "user", "content": conv["user"]})
        messages.append({"role": "assistant", "content": conv[model_key]})
    messages.pop()

    if model_key == "model1":
        base_url = MODEL_HOSTNAME[left_option]
        speed_container = left_speed_container
        reasoning = left_reasoning
    else:
        base_url = MODEL_HOSTNAME[right_option]
        speed_container = right_speed_container
        reasoning = right_reasoning

    with container.chat_message("assistant"):
        expander_container = st.empty()
        placeholder = st.empty()

        thinking_stopped = False
        async for thinking, answer, token_count, tps, elapsed_time in run_request(
            base_url, messages, temperature, max_tokens, reasoning
        ):
            speed_container.html(f"""<div style="display: flex; justify-content: space-between; align-items: center;">
                                 <span>{token_count} symbols</span>
                                 <span><b>Speed:</b> {tps:.2f} symbols/s</span>
                                 <span><b>Time:</b> {elapsed_time:.1f}s</span>
                                 </div>
                                 """)
            if not thinking_stopped and thinking:
                expander_container.expander("Thinking...", expanded=True).markdown(thinking)

            if answer and not thinking_stopped:
                thinking_stopped = True
                if thinking:
                    expander_container.expander("Reasoning content", expanded=False).markdown(thinking)
            placeholder.markdown(answer)

    # Add response to current conversation
    st.session_state.conversations[-1][model_key] = answer
    st.session_state.conversations[-1][f"{model_key}_reasoning"] = thinking
    st.session_state.last_state[model_key] = (token_count, tps, elapsed_time)


conversation_container = st.empty()


def display_intro_screen():
    with conversation_container.container(key="intro"):
        st.markdown("""
        <div style="text-align: center;">
            <h1 style="color: #D64E7B; font-size: 3.5rem; font-weight: bold; margin-bottom: 0.5rem; padding: 0">T-pro 2.0 Interactive Demo</h1>
            <p style="color: #666; font-size: 1.2rem; margin-bottom: 1rem;">A hybrid-reasoning assistant with observable inference optimizations</p>
            <p style="color: #888; font-size: 1rem; margin-bottom: 3rem;">Compare T-pro 2.0 with EAGLE speculative decoding against baseline models under identical infrastructure</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='text-align: center'>✨ Key Features</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background-color: #f8f9fa; border-radius: 10px; height: 220px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔬</div>
                <h4>Research & Comparison</h4>
                <p style="color: #666; font-size: 0.9rem;">Probe reasoning capabilities and compare models side-by-side with explicit reasoning traces.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background-color: #f8f9fa; border-radius: 10px; height: 220px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚡</div>
                <h4>Performance Telemetry</h4>
                <p style="color: #666; font-size: 0.9rem;">Observe latency, throughput, and streaming speed with token-by-token outputs.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background-color: #f8f9fa; border-radius: 10px; height: 220px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎓</div>
                <h4>Educational Content</h4>
                <p style="color: #666; font-size: 0.9rem;">Solve olympiad-level problems with step-by-step reasoning for students and educators.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background-color: #f8f9fa; border-radius: 10px; height: 220px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📚</div>
                <h4>Curated Benchmarks</h4>
                <p style="color: #666; font-size: 0.9rem;">Predefined prompts from evaluation suites across Math, Code, QA, and Sciences.</p>
            </div>
            """, unsafe_allow_html=True)


def display_conversations():
    # Display all previous conversations
    with conversation_container.container():
        for conv in st.session_state.conversations:
            with st.chat_message("user"):
                st.markdown(conv["user"])

            col1, col2 = st.columns(2)
            with col1:
                with st.chat_message("assistant"):
                    if conv.get("model1_reasoning"):
                        st.expander("Reasoning content", expanded=False).markdown(conv["model1_reasoning"])
                    st.markdown(conv["model1"])
            with col2:
                with st.chat_message("assistant"):
                    if conv.get("model2_reasoning"):
                        st.expander("Reasoning content", expanded=False).markdown(conv["model2_reasoning"])
                    st.markdown(conv["model2"])
    for c, m in ((left_speed_container, "model1"), (right_speed_container, "model2")):
        state = st.session_state.last_state[m]
        if state is None:
            c.empty()
            continue
        token_count, tps, elapsed_time = state
        c.html(f"""<div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>{token_count} tokens</span>
                    <span><b>Speed:</b> {tps:.2f} symbols/s</span>
                    <span><b>Time:</b> {elapsed_time:.1f}s</span>
                    </div>
                    """)


async def run_both_models(prompt):
    display_conversations()

    # Create new conversation entry
    st.session_state.conversations.append({"user": prompt, "model1": "", "model2": ""})

    # # Display new user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display new assistant responses in columns
    col1, col2 = st.columns(2)

    await asyncio.gather(
        run_model_response(col1, "model1"),
        run_model_response(col2, "model2"),
    )


with st.sidebar:
    st.header("Settings")
    system_prompt = st.text_area("System Prompt", disabled=st.session_state.is_generating)
    temperature = st.number_input("Temperature", 0.0, 1.0, 0.0, 0.1, disabled=st.session_state.is_generating)
    
    # Show warning if reasoning is enabled
    if st.session_state.get("left_reasoning", False) or st.session_state.get("right_reasoning", False):
        st.warning("Reasoning mode requires more tokens. Consider increasing max tokens.", icon="⚠️")
    
    max_tokens = st.number_input("Tokens", 1, 16384, 1024, 1, disabled=st.session_state.is_generating)
    
    if st.button("Clear Chat", disabled=st.session_state.is_generating, use_container_width=True):
        st.session_state.conversations = []
        st.session_state.last_state = {"model1": None, "model2": None}
        st.rerun()

# Accept user input
with bottom():
    col_left_sel, col_right_sel = st.columns(2)
    with col_left_sel:
        left_speed_container = st.empty()
        left_status = st.empty()
        with st.container(key="left_container"):
            col_model, col_reasoning = st.columns([4, 1])
            with col_model:
                left_option = st.selectbox(
                    "Left model",
                    ["T-pro 2.0 32B + EAGLE", "T-pro 2.0 32B", "Qwen3 32B"],
                    disabled=st.session_state.is_generating or bool(st.session_state.conversations),
                )
            with col_reasoning:
                st.markdown("<div style='height: 34px'></div>", unsafe_allow_html=True)
                left_reasoning = st.checkbox("Reasoning", disabled=st.session_state.is_generating, key="left_reasoning")

    with col_right_sel:
        right_speed_container = st.empty()
        right_status = st.empty()
        with st.container(key="right_container"):
            col_model, col_reasoning = st.columns([4, 1])
            with col_model:
                right_option = st.selectbox(
                    "Right model",
                    ["T-pro 2.0 32B + EAGLE", "T-pro 2.0 32B", "Qwen3 32B"],
                    index=2,
                    disabled=st.session_state.is_generating or bool(st.session_state.conversations),
                )
            with col_reasoning:
                st.markdown("<div style='height: 34px'></div>", unsafe_allow_html=True)
                right_reasoning = st.checkbox("Reasoning", disabled=st.session_state.is_generating, key="right_reasoning")


def disable():
    st.session_state.is_generating = True


if prompt := st.chat_input(disabled=st.session_state.is_generating, on_submit=disable, key="input_preset"):
    models_to_warm = []
    models_to_warm.append(
        {"name": f"Left ({left_option})", "host": MODEL_HOSTNAME[left_option], "placeholder": left_status}
    )
    models_to_warm.append(
        {"name": f"Right ({right_option})", "host": MODEL_HOSTNAME[right_option], "placeholder": right_status}
    )
    if not st.session_state.conversations:
        conversation_container.empty()
    else:
        display_conversations()
    with st.spinner(
        "Selected models are hosted on serverless RunPod. Cold starts can take up to ~5 minutes.\nStarting selected models..."
    ):
        ok = asyncio.run(warmup_in_parallel(models_to_warm))
    left_status.empty()
    right_status.empty()
    asyncio.run(run_both_models(prompt))
    st.session_state.is_generating = False
    st.rerun()
else:
    if st.session_state.conversations:
        display_conversations()
    else:
        display_intro_screen()



PROMPTS_DIR = Path("prompts")
prompt_presets: dict[str, list[str]] = {}

# prompt loading
for category_dir in PROMPTS_DIR.iterdir():
    if category_dir.is_dir():
        prompts = []
        for prompt_file in category_dir.glob("*.txt"):
            prompts.append(prompt_file.read_text().strip())
        if prompts:
            prompt_presets[category_dir.name] = prompts


with bottom():
    with st.container(horizontal=True, key="preset_buttons"):
        st.markdown("Preset Prompts:", width="content")
        for e, v in prompt_presets.items():
            st.button(
                e,
                on_click=lambda p=random.choice(v): st.session_state.update({"input_preset": p}),
                disabled=st.session_state.is_generating,
            )
