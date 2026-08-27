import streamlit as st
import requests
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Writing Assistant",
    page_icon="🤖",
    layout="wide"
)


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
            "The request took too long. "
            "Please try again."
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
# BUILD DOCUMENT CHUNKS
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
            page_text=page_text,
            page_number=page_number,
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

    scores = np.dot(
        document_embeddings,
        query_embedding
    )

    return scores


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

Remove unnecessary explanations.
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

    # --------------------------------------------------------
    # TASK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TONE + LENGTH
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    text = st.text_area(
        "Enter your text",
        height=250,
        placeholder="Paste your text here..."
    )

    # --------------------------------------------------------
    # OPTIONAL PUBLISHING
    # --------------------------------------------------------

    st.subheader("📣 Publishing / Format")

    st.caption(
        "Optional — choose a platform only if you want "
        "the output optimized for it."
    )

    publish_platform = st.selectbox(
        "Where will you publish this? (Optional)",
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

    # --------------------------------------------------------
    # ADDITIONAL INSTRUCTION
    # --------------------------------------------------------

    custom_instruction = st.text_input(
        "Additional instruction (optional)",
        placeholder=(
            "Example: Make it more engaging"
        )
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

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
                task=task,
                text=text,
                tone=tone,
                length=length,
                publish_platform=publish_platform,
                custom_instruction=custom_instruction
            )

            with st.spinner("Generating..."):

                result = ask_ollama(
                    prompt,
                    temperature=creativity
                )

            st.subheader("Result")

            st.write(result)


# ============================================================
# DOCUMENT AI
# ============================================================

with document_tab:

    st.header("📄 Document AI")

    st.write(
        "Upload a PDF and ask questions about its content."
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "📤 Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

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

        # ----------------------------------------------------
        # DOCUMENT
        # ----------------------------------------------------

        if chunks:

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
            # VIEW CHUNKS
            # ------------------------------------------------

            with st.expander(
                "📦 View document chunks"
            ):

                for i, chunk in enumerate(chunks):

                    st.markdown(
                        f"**Chunk {i + 1} — Page "
                        f"{chunk['page']}**"
                    )

                    st.write(
                        chunk["text"]
                    )

                    st.divider()

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
            # QUESTION
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "🔎 Ask Your Document"
            )

            question = st.text_input(
                "Ask a question about the PDF",
                placeholder=(
                    "Example: What is the main topic "
                    "of this document?"
                )
            )

            top_k = st.slider(
                "Number of relevant chunks",
                min_value=1,
                max_value=min(5, len(chunks)),
                value=min(3, len(chunks))
            )

            ask_question = st.button(
                "🔍 Search & Answer",
                type="primary"
            )

            if ask_question:

                if not question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    # ----------------------------------------
                    # RETRIEVAL
                    # ----------------------------------------

                    with st.spinner(
                        "Searching the document..."
                    ):

                        results = retrieve_chunks(
                            question,
                            chunks,
                            embeddings,
                            top_k
                        )

                    # ----------------------------------------
                    # SOURCES
                    # ----------------------------------------

                    st.subheader(
                        "📚 Retrieved Sources"
                    )

                    source_pages = sorted(
                        set(
                            result["page"]
                            for result in results
                        )
                    )

                    st.write(
                        "Relevant pages: "
                        + ", ".join(
                            f"Page {page}"
                            for page in source_pages
                        )
                    )

                    # ----------------------------------------
                    # RETRIEVED CHUNKS
                    # ----------------------------------------

                    with st.expander(
                        "View retrieved content"
                    ):

                        for result in results:

                            st.markdown(
                                f"**Page "
                                f"{result['page']}**"
                            )

                            st.caption(
                                f"Similarity score: "
                                f"{result['score']:.3f}"
                            )

                            st.write(
                                result["text"]
                            )

                            st.divider()

                    # ----------------------------------------
                    # BUILD CONTEXT
                    # ----------------------------------------

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

                    # ----------------------------------------
                    # RAG PROMPT
                    # ----------------------------------------

                    rag_prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the
provided document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not in the context,
   say that the information is not available
   in the uploaded document.
4. Give a clear and concise answer.
5. At the end, provide the source page numbers.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer using the document context.

Include a final line:

Sources: Page X, Page Y
"""

                    # ----------------------------------------
                    # GENERATE
                    # ----------------------------------------

                    with st.spinner(
                        "Generating document answer..."
                    ):

                        answer = ask_ollama(
                            rag_prompt,
                            temperature=0.2
                        )

                    # ----------------------------------------
                    # ANSWER
                    # ----------------------------------------

                    st.subheader(
                        "🤖 Answer"
                    )

                    st.write(answer)