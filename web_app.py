import requests
import streamlit as st


# ==========================================
# CONFIGURATION
# ==========================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


# ==========================================
# STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="AI Writing Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Writing Assistant")

st.write(
    "Transform, rewrite and analyze text using a locally hosted GenAI model."
)

st.divider()


# ==========================================
# AI TASK
# ==========================================

task = st.selectbox(
    "Choose an AI task",
    [
        "Summarize",
        "Explain",
        "Rewrite",
        "Make Professional",
        "Simplify",
        "Extract Key Points"
    ]
)


# ==========================================
# TONE
# ==========================================

tone = st.selectbox(
    "Choose a tone",
    [
        "Professional",
        "Friendly",
        "Casual",
        "Academic",
        "Persuasive"
    ]
)


# ==========================================
# RESPONSE LENGTH
# ==========================================

length = st.selectbox(
    "Response length",
    [
        "Short",
        "Medium",
        "Detailed"
    ]
)


# ==========================================
# USER INPUT
# ==========================================

text = st.text_area(
    "Enter your text",
    height=200,
    placeholder="Paste your text here..."
)


# ==========================================
# WORD COUNTER
# ==========================================

word_count = len(text.split())
character_count = len(text)

st.caption(
    f"📝 {word_count} words · {character_count} characters"
)


# ==========================================
# PROMPT ENGINEERING
# ==========================================

def create_prompt(task, text, tone, length):

    length_instruction = {
        "Short": "Keep the response concise.",
        "Medium": "Provide a balanced response with useful detail.",
        "Detailed": "Provide a detailed and comprehensive response."
    }

    prompts = {

        "Summarize":
            f"""
Summarize the following text clearly and concisely.

Tone: {tone}
Response requirement: {length_instruction[length]}

Text:
{text}
""",

        "Explain":
            f"""
Explain the following text clearly so that it is easy to understand.

Tone: {tone}
Response requirement: {length_instruction[length]}

Text:
{text}
""",

        "Rewrite":
            f"""
Rewrite the following text to make it clearer and more effective.

Tone: {tone}
Response requirement: {length_instruction[length]}

Text:
{text}
""",

        "Make Professional":
            f"""
Rewrite the following text in a professional business style.

Tone: {tone}
Response requirement: {length_instruction[length]}

Text:
{text}
""",

        "Simplify":
            f"""
Simplify the following text so that it is easy to understand.

Tone: {tone}
Response requirement: {length_instruction[length]}

Text:
{text}
""",

        "Extract Key Points":
            f"""
Extract the most important key points from the following text.

Tone: {tone}
Response requirement: {length_instruction[length]}

Text:
{text}
"""
    }

    return prompts[task]


# ==========================================
# OLLAMA API
# ==========================================

def generate_response(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


# ==========================================
# GENERATE BUTTON
# ==========================================

if st.button("✨ Generate", type="primary"):

    if not text.strip():

        st.warning("Please enter some text first.")

    else:

        prompt = create_prompt(
            task,
            text,
            tone,
            length
        )

        with st.spinner("🤖 Qwen is thinking..."):

            try:

                result = generate_response(prompt)

                st.divider()

                st.subheader("AI Response")

                st.write(result)

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
                    f"Something went wrong: {error}"
                )