import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def ask_ai(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    return response.json()["response"]


def build_prompt(role, task, requirements, text):
    requirements_text = "\n".join(
        f"- {requirement}" for requirement in requirements
    )

    return f"""
Role:
{role}

Task:
{task}

Requirements:
{requirements_text}

User Text:
{text}

Response:
"""


def compare_prompts(text):
    basic_prompt = f"""
Summarize this:

{text}
"""

    structured_prompt = build_prompt(
        role="You are an experienced project manager.",
        task="Analyze the following project update.",
        requirements=[
            "Identify the main issue.",
            "Identify the impact.",
            "Identify the next action.",
            "Use concise bullet points.",
            "Do not invent information."
        ],
        text=text
    )

    print("\nGenerating basic prompt response...")
    basic_result = ask_ai(basic_prompt)

    print("Generating structured prompt response...")
    structured_result = ask_ai(structured_prompt)

    print("\n" + "=" * 60)
    print("BASIC PROMPT")
    print("=" * 60)
    print(basic_result)

    print("\n" + "=" * 60)
    print("STRUCTURED PROMPT")
    print("=" * 60)
    print(structured_result)


def main():

    print("=" * 60)
    print("              AI WRITING ASSISTANT")
    print("=" * 60)

    print("\nChoose what you want the AI to do:\n")

    print("1. Summarize")
    print("2. Explain")
    print("3. Rewrite")
    print("4. Make Professional")
    print("5. Simplify")
    print("6. Extract Key Points")
    print("7. Compare Prompt Quality")

    choice = input("\nEnter your choice (1-7): ")
    text = input("\nEnter your text:\n")

    prompt_configs = {

        "1": {
            "role": "You are a professional summarization assistant.",
            "task": "Summarize the provided text.",
            "requirements": [
                "Provide exactly 3 bullet points.",
                "Focus on the most important information.",
                "Keep each bullet concise.",
                "Do not add information not present in the text."
            ]
        },

        "2": {
            "role": "You are an expert teacher.",
            "task": "Explain the provided text in simple language.",
            "requirements": [
                "Assume the reader is a beginner.",
                "Explain difficult concepts clearly.",
                "Avoid unnecessary technical terminology.",
                "Do not change the original meaning."
            ]
        },

        "3": {
            "role": "You are a professional writing assistant.",
            "task": "Rewrite the provided text while preserving its meaning.",
            "requirements": [
                "Improve grammar.",
                "Improve clarity.",
                "Improve readability.",
                "Do not add new information."
            ]
        },

        "4": {
            "role": "You are a professional business communication assistant.",
            "task": "Rewrite the provided text in a professional and polished tone.",
            "requirements": [
                "Keep the original meaning.",
                "Use clear business language.",
                "Be concise.",
                "Do not invent information."
            ]
        },

        "5": {
            "role": "You are a communication assistant.",
            "task": "Rewrite the provided text using simple language.",
            "requirements": [
                "Use short sentences.",
                "Avoid unnecessary technical terms.",
                "Preserve the original meaning.",
                "Make the text easy to understand."
            ]
        },

        "6": {
            "role": "You are an information extraction assistant.",
            "task": "Extract the most important information from the provided text.",
            "requirements": [
                "Return concise bullet points.",
                "Focus on important facts and decisions.",
                "Do not add information.",
                "Remove unnecessary details."
            ]
        }
    }

    if choice == "7":
        print("\nRunning prompt engineering experiment...")
        compare_prompts(text)
        return

    if choice not in prompt_configs:
        print("\nInvalid choice. Please select 1-7.")
        return

    config = prompt_configs[choice]

    prompt = build_prompt(
        role=config["role"],
        task=config["task"],
        requirements=config["requirements"],
        text=text
    )

    print("\nGenerating AI response...")
    print("-" * 60)

    result = ask_ai(prompt)

    print("\nAI Response:")
    print("-" * 60)
    print(result)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
