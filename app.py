import asyncio
import os

import streamlit as st
from streamlit_extras.bottom_container import bottom

from utils import run_request, warmup_in_parallel

MODEL_HOSTNAME = {
    "t pro 32b with eagle": os.getenv("TPRO_WITH_EAGLE_HOST"),
    "t pro 32b": os.getenv("TPRO_HOST"),
    "qwen 32b": os.getenv("QWEN_HOST"),
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
</style>
""",
    unsafe_allow_html=True,
)

if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "input_preset" not in st.session_state:
    st.session_state.input_preset = ""
if "last_state" not in st.session_state:
    st.session_state.last_state = {}
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
        model_name = left_option
    else:
        base_url = MODEL_HOSTNAME[right_option]
        model_name = right_option

    with container.chat_message("assistant"):
        speed_container = st.empty()
        expander_container = st.empty()
        placeholder = st.empty()

        thinking_stopped = False
        async for thinking, answer, token_count, tps in run_request(
            base_url, messages, temperature, max_tokens, reasoning_choice
        ):
            speed_container.html(f"""<div style="display: flex; justify-content: space-between; align-items: center; font-size: 10px; opacity: 70%">
                                 <span>{model_name}</span>
                                 <span><b>Speed:</b> {tps:.2f} symbols/s</span>
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


conversation_container = st.empty()


def display_conversations():
    # Display all previous conversations
    with conversation_container.container():
        for conv in st.session_state.conversations:
            with st.chat_message("user"):
                st.markdown(conv["user"])

            col1, col2 = st.columns(2)
            with col1:
                with st.chat_message("assistant"):
                    st.markdown(conv["model1"])
            with col2:
                with st.chat_message("assistant"):
                    st.markdown(conv["model2"])


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
    max_tokens = st.number_input("Tokens", 1, 16384, 1024, 1, disabled=st.session_state.is_generating)
    reasoning_choice = st.checkbox("Reasoning", disabled=st.session_state.is_generating)

# Accept user input
with bottom():
    col_left_sel, col_right_sel = st.columns(2)
    with col_left_sel:
        left_status = st.empty()
        left_option = st.selectbox(
            "Left model",
            ["t pro 32b with eagle"],
            disabled=st.session_state.is_generating or bool(st.session_state.conversations),
        )

    with col_right_sel:
        right_status = st.empty()
        right_option = st.selectbox(
            "Right model",
            ["qwen 32b", "t pro 32b"],
            index=0,
            disabled=st.session_state.is_generating or bool(st.session_state.conversations),
        )


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
    display_conversations()


PROMPTS_DIR = "prompts"
prompt_presets = {}
for fname in os.listdir(PROMPTS_DIR):
    if fname.endswith(".txt"):
        label = fname.replace(".txt", "").replace("_", " ")
        with open(os.path.join(PROMPTS_DIR, fname), encoding="utf-8") as f:
            prompt_presets[label] = f.read().strip()


with bottom():
    with st.container(horizontal=True, key="preset_buttons"):
        st.markdown("Preset Prompts:", width="content")
        for e, v in prompt_presets.items():
            st.button(
                e,
                on_click=lambda p=v: st.session_state.update({"input_preset": p}),
                disabled=st.session_state.is_generating,
            )
