import streamlit as st
import requests
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Writing Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(prompt, temperature=0.7):

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            },
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:

        return (
            "Could not connect to Ollama. "
            "Please make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        return (
            "The request took too long. Please try again."
        )

    except Exception as e:

        return f"Error: {str(e)}"


# ============================================================
# PAGE-AWARE CHUNKING
# ============================================================

def create_page_chunks(
    page_text,
    page_number,
    chunk_size=800,
    overlap=100
):

    words = page_text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_text = " ".join(
            words[start:end]
        )

        if chunk_text.strip():

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "page": page_number
                }
            )

        start += chunk_size - overlap

    return chunks


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    all_chunks = []

    pages_with_text = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if not page_text:
            continue

        page_text = page_text.strip()

        if not page_text:
            continue

        pages_with_text += 1

        page_chunks = create_page_chunks(
            page_text,
            page_number,
            chunk_size=800,
            overlap=100
        )

        all_chunks.extend(page_chunks)

    return reader, all_chunks, pages_with_text


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query_embedding,
    document_embeddings
):

    query_embedding = query_embedding / (
        np.linalg.norm(query_embedding) + 1e-10
    )

    document_embeddings = (
        document_embeddings /
        (
            np.linalg.norm(
                document_embeddings,
                axis=1,
                keepdims=True
            ) + 1e-10
        )
    )

    return np.dot(
        document_embeddings,
        query_embedding
    )


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    question,
    chunks,
    embeddings,
    top_k=3
):

    question_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        question_embedding,
        embeddings
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "text": chunks[index]["text"],
                "page": chunks[index]["page"],
                "score": float(scores[index]),
                "index": int(index)
            }
        )

    return results


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def build_chat_history():

    if not st.session_state.chat_history:
        return "No previous conversation."

    history = []

    for message in st.session_state.chat_history:

        role = message["role"]
        content = message["content"]

        if role == "user":

            history.append(
                f"User: {content}"
            )

        elif role == "assistant":

            history.append(
                f"Assistant: {content}"
            )

    return "\n\n".join(history)


# ============================================================
# PLATFORM INSTRUCTIONS
# ============================================================

PLATFORM_INSTRUCTIONS = {

    "LinkedIn Post": """
Create content suitable for LinkedIn.

Use:
- A strong opening
- Short paragraphs
- Professional but natural language
- A useful takeaway
- Good readability

Avoid:
- Excessive hashtags
- Generic corporate language
- Unnecessary emojis
""",

    "Instagram Caption": """
Create content suitable for an Instagram caption.

Use:
- An engaging opening
- Short readable paragraphs
- Conversational language
- Emojis where appropriate
- A call to action when useful
""",

    "X (Twitter) Post": """
Create content suitable for X/Twitter.

Use:
- Concise wording
- Strong opening
- Short sentences
- Clear message
""",

    "Blog Article": """
Create content suitable for a blog article.

Use:
- Clear title
- Introduction
- Logical headings
- Well-developed sections
- Examples where useful
- Conclusion
""",

    "Email": """
Create content suitable for an email.

Use:
- Subject line
- Greeting
- Clear body
- Professional structure
- Appropriate closing
""",

    "YouTube Description": """
Create content suitable for a YouTube description.

Use:
- Strong opening
- Brief explanation
- Useful details
- Clear structure
- Relevant keywords naturally
""",

    "Product Description": """
Create a clear and persuasive product description.

Focus on:
- What the product is
- Main benefits
- Important features
- Customer value

Avoid exaggerated or unsupported claims.
""",

    "Professional / Business Writing": """
Create polished professional or business writing.

Use:
- Clear language
- Professional structure
- Direct communication
- Appropriate business tone
"""
}


# ============================================================
# WRITING PROMPT
# ============================================================

def build_writing_prompt(
    task,
    text,
    tone,
    length,
    publish_platform,
    custom_instruction
):

    length_instructions = {

        "Short":
            "Keep the output concise and focused.",

        "Medium":
            "Provide a balanced amount of detail.",

        "Long":
            "Provide detailed and well-developed content."
    }

    task_instructions = {

        "Summarize": """
Summarize the provided text.

Keep the most important information.
Remove unnecessary repetition.
Do not introduce information that is not in the original text.
""",

        "Rewrite": """
Rewrite the provided text to make it clearer,
more natural and effective.

Preserve the original meaning.
""",

        "Improve Grammar": """
Improve grammar, spelling, punctuation,
sentence structure and clarity.

Do not change the intended meaning.
""",

        "Change Tone": f"""
Rewrite the text using a {tone} tone.

Preserve the original meaning.
""",

        "Expand": """
Expand the provided text with useful information.

Keep the original meaning and topic.
Do not add unsupported facts.
""",

        "Extract Key Points": """
Extract the most important points from the text.

Use clear bullet points.
""",

        "Compare Prompt Quality": """
Analyze the text as an AI prompt.

Evaluate:

1. Clarity
2. Context
3. Specificity
4. Expected output
5. Missing information
6. Potential improvements

Provide practical recommendations.
"""
    }

    prompt = f"""
You are an AI writing assistant.

TASK:

{task_instructions[task]}

TONE:

{tone}

LENGTH:

{length_instructions[length]}

SOURCE TEXT:

{text}
"""

    if publish_platform != "No specific platform":

        platform_instruction = PLATFORM_INSTRUCTIONS.get(
            publish_platform,
            ""
        )

        prompt += f"""

PUBLISHING / FORMAT:

Optimize the output for:

{publish_platform}

Requirements:

{platform_instruction}
"""

    if custom_instruction.strip():

        prompt += f"""

ADDITIONAL USER INSTRUCTION:

{custom_instruction}
"""

    prompt += """

Return only the useful final answer.

Do not explain your reasoning.
Do not mention these instructions.
"""

    return prompt


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ AI Controls")

    creativity = st.slider(
        "🎨 Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    st.divider()

    st.subheader("Model")

    st.code(MODEL_NAME)

    st.subheader("Runtime")

    st.info("Ollama (Local)")


# ============================================================
# HEADER
# ============================================================

st.title("🤖 AI Writing Assistant")

st.write(
    "Transform, rewrite and analyze text with AI."
)


# ============================================================
# TABS
# ============================================================

writing_tab, document_tab = st.tabs(
    [
        "✍️ Writing Assistant",
        "📄 Document AI"
    ]
)


# ============================================================
# WRITING ASSISTANT
# ============================================================

with writing_tab:

    st.header("✍️ Writing Assistant")

    st.write(
        "Transform your text for different purposes and platforms."
    )

    task = st.selectbox(
        "Choose an AI task",
        [
            "Summarize",
            "Rewrite",
            "Improve Grammar",
            "Change Tone",
            "Expand",
            "Extract Key Points",
            "Compare Prompt Quality"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:

        tone = st.selectbox(
            "Tone",
            [
                "Professional",
                "Friendly",
                "Formal",
                "Casual",
                "Confident",
                "Simple"
            ]
        )

    with col2:

        length = st.selectbox(
            "Length",
            [
                "Short",
                "Medium",
                "Long"
            ]
        )

    text = st.text_area(
        "Enter your text",
        height=250,
        placeholder="Paste your text here..."
    )

    with st.expander("📣 Publishing & format — Optional"):

        st.caption(
            "Choose a platform only when you want the output "
            "optimized for a specific format."
        )

        publish_platform = st.selectbox(
            "Where will you publish this?",
            [
                "No specific platform",
                "LinkedIn Post",
                "Instagram Caption",
                "X (Twitter) Post",
                "Blog Article",
                "Email",
                "YouTube Description",
                "Product Description",
                "Professional / Business Writing"
            ]
        )

        custom_instruction = st.text_input(
            "Additional instruction — Optional",
            placeholder="Example: Make it more engaging and concise"
        )

    generate = st.button(
        "✨ Generate",
        type="primary"
    )

    if generate:

        if not text.strip():

            st.warning(
                "Please enter some text first."
            )

        else:

            prompt = build_writing_prompt(
                task,
                text,
                tone,
                length,
                publish_platform,
                custom_instruction
            )

            with st.spinner("Generating..."):

                result = ask_ollama(
                    prompt,
                    temperature=creativity
                )

            st.subheader("Result")

            st.write(result)


# ============================================================
# DOCUMENT AI / RAG
# ============================================================

with document_tab:

    st.header("📄 Document AI")

    st.write(
        "Upload a PDF and have a conversation about its content."
    )

    uploaded_file = st.file_uploader(
        "📤 Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        # ----------------------------------------------------
        # RESET CHAT WHEN A DIFFERENT DOCUMENT IS UPLOADED
        # ----------------------------------------------------

        if (
            st.session_state.document_name
            != uploaded_file.name
        ):

            st.session_state.chat_history = []

            st.session_state.document_name = (
                uploaded_file.name
            )

        try:

            reader, chunks, pages_with_text = process_pdf(
                uploaded_file
            )

            st.success(
                f"PDF loaded successfully — "
                f"{len(reader.pages)} pages"
            )

        except Exception as e:

            st.error(
                f"Could not read PDF: {str(e)}"
            )

            chunks = []

        if chunks:

            # ------------------------------------------------
            # DOCUMENT STATS
            # ------------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "PDF Pages",
                    len(reader.pages)
                )

            with col2:

                st.metric(
                    "Pages With Text",
                    pages_with_text
                )

            with col3:

                st.metric(
                    "Text Chunks",
                    len(chunks)
                )

            # ------------------------------------------------
            # EMBEDDINGS
            # ------------------------------------------------

            with st.spinner(
                "Creating document embeddings..."
            ):

                chunk_texts = [
                    chunk["text"]
                    for chunk in chunks
                ]

                embeddings = embedding_model.encode(
                    chunk_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

            st.success(
                "Document embeddings created successfully."
            )

            st.caption(
                f"Embedding shape: {embeddings.shape}"
            )

            # ------------------------------------------------
            # CLEAR CHAT
            # ------------------------------------------------

            col1, col2 = st.columns(
                [5, 1]
            )

            with col2:

                if st.button("🗑️ Clear Chat"):

                    st.session_state.chat_history = []

                    st.rerun()

            # ------------------------------------------------
            # CHAT HISTORY
            # ------------------------------------------------

            if st.session_state.chat_history:

                st.subheader("💬 Conversation")

                for message in (
                    st.session_state.chat_history
                ):

                    with st.chat_message(
                        message["role"]
                    ):

                        st.write(
                            message["content"]
                        )

                        if (
                            message["role"]
                            == "assistant"
                            and "sources"
                            in message
                        ):

                            sources = message[
                                "sources"
                            ]

                            if sources:

                                st.caption(
                                    "Sources: "
                                    + ", ".join(
                                        f"Page {page}"
                                        for page in sources
                                    )
                                )

            # ------------------------------------------------
            # CHAT INPUT
            # ------------------------------------------------

            question = st.chat_input(
                "Ask a question about your document..."
            )

            if question:

                # --------------------------------------------
                # SHOW USER QUESTION
                # --------------------------------------------

                with st.chat_message("user"):

                    st.write(question)

                # --------------------------------------------
                # SAVE USER QUESTION
                # --------------------------------------------

                st.session_state.chat_history.append(
                    {
                        "role": "user",
                        "content": question
                    }
                )

                # --------------------------------------------
                # RETRIEVE RELEVANT CHUNKS
                # --------------------------------------------

                with st.spinner(
                    "Searching the document..."
                ):

                    results = retrieve_chunks(
                        question,
                        chunks,
                        embeddings,
                        top_k=3
                    )

                # --------------------------------------------
                # BUILD DOCUMENT CONTEXT
                # --------------------------------------------

                context_parts = []

                for result in results:

                    context_parts.append(
                        f"""
SOURCE: Page {result['page']}

{result['text']}
"""
                    )

                context = "\n\n".join(
                    context_parts
                )

                # --------------------------------------------
                # PREVIOUS CONVERSATION
                # --------------------------------------------

                conversation = build_chat_history()

                # --------------------------------------------
                # RAG + MEMORY PROMPT
                # --------------------------------------------

                rag_prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the
provided document context.

You also have access to the previous conversation
to understand references and follow-up questions.

RULES:

1. Use the document context as the source of truth.
2. Do not invent information.
3. Do not use outside knowledge.
4. Use previous conversation only to understand
   what the user is referring to.
5. If the requested information is not available
   in the document context, say:
   "The information is not available in the uploaded document."
6. Give a clear and concise answer.
7. At the end, provide the relevant source pages.

PREVIOUS CONVERSATION:

{conversation}

CURRENT DOCUMENT CONTEXT:

{context}

CURRENT USER QUESTION:

{question}

Answer the user's current question.

End with:

Sources: Page X, Page Y
"""

                # --------------------------------------------
                # GENERATE ANSWER
                # --------------------------------------------

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = ask_ollama(
                        rag_prompt,
                        temperature=0.2
                    )

                # --------------------------------------------
                # SOURCE PAGES
                # --------------------------------------------

                source_pages = sorted(
                    set(
                        result["page"]
                        for result in results
                    )
                )

                # --------------------------------------------
                # SHOW ANSWER
                # --------------------------------------------

                with st.chat_message("assistant"):

                    st.write(answer)

                    st.caption(
                        "Sources: "
                        + ", ".join(
                            f"Page {page}"
                            for page in source_pages
                        )
                    )

                # --------------------------------------------
                # SAVE ANSWER
                # --------------------------------------------

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": source_pages
                    }
                )

            # ------------------------------------------------
            # VIEW CHUNKS
            # ------------------------------------------------

            with st.expander(
                "📦 View document chunks"
            ):

                for i, chunk in enumerate(chunks):

                    st.markdown(
                        f"**Chunk {i + 1} — "
                        f"Page {chunk['page']}**"
                    )

                    st.write(
                        chunk["text"]
                    )

                    st.divider()