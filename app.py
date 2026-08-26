import requests
import streamlit as st


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


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


text = st.text_area(
    "Enter your text",
    height=200,
    placeholder="Paste your text here..."
)


def create_prompt(task, text):

    prompts = {

        "Summarize":
            f"Summarize the following text clearly and concisely:\n\n{text}",

        "Explain":
            f"Explain the following text in simple language:\n\n{text}",

        "Rewrite":
            f"Rewrite the following text to make it clearer and more effective:\n\n{text}",

        "Make Professional":
            f"Rewrite the following text in a professional business tone:\n\n{text}",

        "Simplify":
            f"Simplify the following text so it is easy to understand:\n\n{text}",

        "Extract Key Points":
            f"Extract the most important points from the following text:\n\n{text}"
    }

    return prompts[task]


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

    return response.json()["response"]


if st.button("✨ Generate", type="primary"):

    if not text.strip():

        st.warning("Please enter some text first.")

    else:

        prompt = create_prompt(task, text)

        with st.spinner("🤖 Generating response..."):

            try:

                result = generate_response(prompt)

                st.divider()

                st.subheader("AI Response")

                st.write(result)

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to Ollama. "
                    "Make sure Ollama is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "The AI request took too long. "
                    "Please try again."
                )

            except Exception as error:

                st.error(f"Something went wrong: {error}")
