import json
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Writing Assistant",
    page_icon="🤖",
    layout="centered"
)


# =========================================================
# HEADER
# =========================================================

st.title("🤖 AI Writing Assistant")

st.write(
    "Transform, rewrite, summarize and improve your text "
    "using a locally hosted AI model."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ AI Controls")

    temperature = st.slider(
        "🌡️ Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help=(
            "Lower values produce more predictable responses. "
            "Higher values produce more creative responses."
        )
    )

    st.divider()

    st.caption("Model")

    st.code(MODEL)

    st.caption("Runtime")

    st.code("Ollama (Local)")


# =========================================================
# TASK
# =========================================================

task = st.selectbox(
    "🎯 Choose an AI task",
    [
        "Summarize",
        "Explain",
        "Rewrite",
        "Make Professional",
        "Simplify",
        "Extract Key Points"
    ]
)


# =========================================================
# WRITING STYLE
# =========================================================

writing_style = st.selectbox(
    "✍️ Writing style",
    [
        "General",
        "Email",
        "LinkedIn Post",
        "Business Report",
        "Blog",
        "Academic",
        "Marketing"
    ]
)


# =========================================================
# TONE
# =========================================================

tone = st.selectbox(
    "🎨 Choose a tone",
    [
        "Professional",
        "Friendly",
        "Casual",
        "Academic",
        "Persuasive"
    ]
)


# =========================================================
# RESPONSE LENGTH
# =========================================================

length = st.selectbox(
    "📏 Response length",
    [
        "Short",
        "Medium",
        "Detailed"
    ]
)


# =========================================================
# USER INPUT
# =========================================================

text = st.text_area(
    "✍️ Enter your text",
    height=220,
    placeholder="Paste or type your text here..."
)


# =========================================================
# INPUT STATISTICS
# =========================================================

word_count = len(text.split())
character_count = len(text)

st.caption(
    f"📝 Input: {word_count} words · "
    f"{character_count} characters"
)


# =========================================================
# INPUT VALIDATION
# =========================================================

MAX_CHARACTERS = 10000

if character_count > MAX_CHARACTERS:

    st.warning(
        f"⚠️ Your input is {character_count:,} characters. "
        f"Please keep it below {MAX_CHARACTERS:,} characters."
    )


# =========================================================
# PROMPT ENGINEERING
# =========================================================

def create_prompt(
    task,
    text,
    tone,
    length,
    writing_style
):

    length_instruction = {

        "Short":
            "Keep the response concise.",

        "Medium":
            "Provide a balanced response with useful detail.",

        "Detailed":
            "Provide a detailed and comprehensive response."
    }

    style_instruction = {

        "General":
            "Use a natural and clear writing style.",

        "Email":
            "Format the response appropriately for a professional email.",

        "LinkedIn Post":
            "Write in an engaging LinkedIn-style format.",

        "Business Report":
            "Use a structured and professional business-report style.",

        "Blog":
            "Use an engaging and readable blog-writing style.",

        "Academic":
            "Use a clear, formal and academically appropriate style.",

        "Marketing":
            "Use persuasive and engaging marketing language."
    }

    prompt = f"""
You are an expert AI writing assistant.

Task:
{task}

Writing style:
{writing_style}

Style instruction:
{style_instruction[writing_style]}

Tone:
{tone}

Response length:
{length_instruction[length]}

Important instructions:
- Preserve the original meaning.
- Do not invent facts.
- Make the response clear and useful.
- Follow the requested style and tone.

User text:
{text}
"""

    return prompt.strip()


# =========================================================
# OLLAMA STREAMING
# =========================================================

def stream_response(prompt, temperature):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "temperature": temperature,
            "stream": True
        },
        stream=True,
        timeout=120
    )

    response.raise_for_status()

    for line in response.iter_lines():

        if not line:
            continue

        try:

            data = json.loads(
                line.decode("utf-8")
            )

            chunk = data.get(
                "response",
                ""
            )

            if chunk:
                yield chunk

            if data.get("done", False):
                break

        except json.JSONDecodeError:
            continue


# =========================================================
# BUTTONS
# =========================================================

generate_col, clear_col = st.columns(2)


with generate_col:

    generate_clicked = st.button(
        "✨ Generate",
        type="primary",
        use_container_width=True
    )


with clear_col:

    clear_clicked = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


# =========================================================
# CLEAR SESSION
# =========================================================

if clear_clicked:

    st.session_state.pop(
        "result",
        None
    )

    st.session_state.pop(
        "prompt",
        None
    )

    st.rerun()


# =========================================================
# GENERATE RESPONSE
# =========================================================

if generate_clicked:

    if not text.strip():

        st.warning(
            "⚠️ Please enter some text first."
        )

    elif character_count > MAX_CHARACTERS:

        st.error(
            "❌ Input is too long. "
            "Please reduce the amount of text."
        )

    else:

        prompt = create_prompt(
            task,
            text,
            tone,
            length,
            writing_style
        )

        st.divider()

        st.subheader(
            "🤖 AI Response"
        )

        try:

            result = st.write_stream(
                stream_response(
                    prompt,
                    temperature
                )
            )

            st.session_state["result"] = result

            st.session_state["prompt"] = prompt

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Could not connect to Ollama. "
                "Please make sure Ollama is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ The AI request took too long. "
                "Please try again."
            )

        except Exception as error:

            st.error(
                f"❌ Something went wrong: {error}"
            )


# =========================================================
# RESPONSE DETAILS
# =========================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.subheader(
        "📊 Response Statistics"
    )

    output_words = len(
        result.split()
    )

    output_characters = len(
        result
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Input Words",
            word_count
        )

    with col2:

        st.metric(
            "Output Words",
            output_words
        )

    st.caption(
        f"Output characters: "
        f"{output_characters}"
    )


    # =====================================================
    # DOWNLOAD
    # =====================================================

    st.download_button(
        label="💾 Download Response",
        data=result,
        file_name="ai_response.txt",
        mime="text/plain",
        use_container_width=True
    )


    # =====================================================
    # GENERATED PROMPT
    # =====================================================

    with st.expander(
        "🔍 Show Generated Prompt"
    ):

        st.code(
            st.session_state["prompt"],
            language="text"
        )