import requests


def ask_ai(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]


def compare_prompts(text):
    basic_prompt = f"""
Summarize this:

{text}
"""

    structured_prompt = f"""
You are an experienced project manager.

Task:
Analyze the following project update.

Requirements:
- Identify the main issue.
- Identify the impact.
- Identify the next action.
- Use concise bullet points.
- Do not invent information.

Project update:
{text}
"""

    print("\nGenerating response using basic prompt...")
    basic_result = ask_ai(basic_prompt)

    print("Generating response using structured prompt...")
    structured_result = ask_ai(structured_prompt)

    print("\n" + "=" * 60)
    print("BASIC PROMPT RESULT")
    print("=" * 60)
    print(basic_result)

    print("\n" + "=" * 60)
    print("STRUCTURED PROMPT RESULT")
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

    prompts = {
        "1": f"""
You are a professional summarization assistant.

Task:
Summarize the following text.

Requirements:
- Provide exactly 3 bullet points.
- Focus only on the most important information.
- Keep each bullet concise.
- Do not introduce information that is not present.

Text:
{text}
""",

        "2": f"""
You are an expert teacher.

Task:
Explain the following text in simple language.

Requirements:
- Assume the reader is a beginner.
- Explain difficult concepts clearly.
- Avoid unnecessary technical terminology.
- Do not change the meaning.

Text:
{text}
""",

        "3": f"""
You are a professional writing assistant.

Task:
Rewrite the following text while preserving its original meaning.

Requirements:
- Improve grammar.
- Improve clarity.
- Improve readability.
- Do not add new information.

Text:
{text}
""",

        "4": f"""
You are a professional business communication assistant.

Task:
Rewrite the following text in a professional and polished tone.

Requirements:
- Keep the original meaning.
- Use clear business language.
- Be concise.
- Do not invent information.

Text:
{text}
""",

        "5": f"""
You are a communication assistant.

Task:
Rewrite the following text using simple language.

Requirements:
- Use short sentences.
- Avoid unnecessary technical terms.
- Preserve the original meaning.
- Make the text easy to understand.

Text:
{text}
""",

        "6": f"""
You are an information extraction assistant.

Task:
Extract the most important information from the following text.

Requirements:
- Return concise bullet points.
- Focus on important facts and decisions.
- Do not add information.
- Remove unnecessary details.

Text:
{text}
"""
    }

    if choice not in prompts and choice != "7":
        print("\nInvalid choice. Please select 1-7.")
        return

    if choice == "7":
        print("\nRunning prompt engineering experiment...")
        compare_prompts(text)
        return

    print("\nGenerating AI response...")
    print("-" * 60)

    result = ask_ai(prompts[choice])

    print("\nAI Response:")
    print("-" * 60)
    print(result)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
