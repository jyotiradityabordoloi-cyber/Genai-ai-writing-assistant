import json
import requests
import streamlit as st
from pypdf import PdfReader


# =========================================================
# CONFIGURATION
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"
MAX_CHARACTERS = 10000


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Writing Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #9ca3af;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3a3a45;
        background-color: #171821;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 AI Writing Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Transform, rewrite and analyze text with AI.'
    '</div>',
    unsafe_allow_html=True
)


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
        step=0.1
    )

    st.divider()

    st.subheader("Model")

    st.code(
        MODEL,
        language="text"
    )

    st.caption("Runtime")

    st.code(
        "Ollama (Local)",
        language="text"
    )


# =========================================================
# TABS
# =========================================================

writing_tab, document_tab = st.tabs(
    [
        "✍️ Writing Assistant",
        "📄 Document AI"
    ]
)


# =========================================================
# WRITING ASSISTANT
# =========================================================

with writing_tab:

    st.markdown(
        '<div class="section-title">✍️ Writing Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Write, rewrite and improve your content.'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # AI TASK
    # =====================================================

    task = st.selectbox(
        "🎯 What do you want AI to do?",
        [
            "Summarize",
            "Explain",
            "Rewrite",
            "Make Professional",
            "Simplify",
            "Extract Key Points"
        ],
        key="writing_task"
    )


    # =====================================================
    # WRITING PURPOSE
    # =====================================================

    purpose = st.selectbox(
        "🎯 What is the purpose?",
        [
            "Inform",
            "Persuade",
            "Request",
            "Follow Up",
            "Announce",
            "Apologize",
            "Thank",
            "Respond",
            "Sell",
            "Explain",
            "General"
        ],
        key="writing_purpose"
    )


    # =====================================================
    # WHERE WILL IT BE USED?
    # =====================================================

    platform = st.selectbox(
        "📍 Where will you use this?",
        [
            "General",
            "Email",
            "LinkedIn",
            "WhatsApp",
            "Slack",
            "Business Report",
            "Resume",
            "Cover Letter",
            "Presentation",
            "Blog",
            "Marketing",
            "Academic"
        ],
        key="writing_platform"
    )


    # =====================================================
    # WRITING STYLE
    # =====================================================

    writing_style = st.selectbox(
        "✍️ Writing style",
        [
            "General",
            "Professional",
            "Conversational",
            "Business",
            "Storytelling",
            "Academic",
            "Marketing",
            "Technical",
            "Executive"
        ],
        key="writing_style"
    )


    # =====================================================
    # TONE
    # =====================================================

    tone = st.selectbox(
        "🎨 Tone",
        [
            "Professional",
            "Friendly",
            "Casual",
            "Formal",
            "Academic",
            "Persuasive",
            "Confident",
            "Empathetic",
            "Direct"
        ],
        key="writing_tone"
    )


    # =====================================================
    # RESPONSE LENGTH
    # =====================================================

    length = st.selectbox(
        "📏 Response length",
        [
            "Short",
            "Medium",
            "Detailed"
        ],
        key="writing_length"
    )


    # =====================================================
    # INPUT TEXT
    # =====================================================

    text = st.text_area(
        "📝 Enter your text",
        height=250,
        placeholder="Paste or type your text here...",
        key="writing_text"
    )


    # =====================================================
    # TEXT STATISTICS
    # =====================================================

    word_count = len(text.split())
    character_count = len(text)

    st.caption(
        f"📝 Input: {word_count} words · "
        f"{character_count:,} characters"
    )


    if character_count > MAX_CHARACTERS:

        st.warning(
            f"⚠️ Please keep your input below "
            f"{MAX_CHARACTERS:,} characters."
        )


    # =====================================================
    # PROMPT CREATION
    # =====================================================

    def create_writing_prompt(
        task,
        purpose,
        platform,
        writing_style,
        tone,
        length,
        text
    ):

        length_instruction = {

            "Short":
                "Keep the response concise and focused.",

            "Medium":
                "Provide a balanced response with enough detail.",

            "Detailed":
                "Provide a detailed and comprehensive response."
        }


        platform_instruction = {

            "General":
                "Write in a clear and natural format.",

            "Email":
                "Format the response as a professional email.",

            "LinkedIn":
                "Format the response as an engaging LinkedIn post.",

            "WhatsApp":
                "Make the response natural and suitable for WhatsApp.",

            "Slack":
                "Make the response concise and suitable for workplace Slack.",

            "Business Report":
                "Use a structured business-report format.",

            "Resume":
                "Use strong, concise, achievement-oriented resume language.",

            "Cover Letter":
                "Write in a professional cover-letter format.",

            "Presentation":
                "Make the response suitable for presentation slides.",

            "Blog":
                "Use an engaging and readable blog format.",

            "Marketing":
                "Use persuasive marketing language.",

            "Academic":
                "Use a formal academic writing format."
        }


        style_instruction = {

            "General":
                "Use clear and natural language.",

            "Professional":
                "Use polished professional language.",

            "Conversational":
                "Use natural conversational language.",

            "Business":
                "Use business-oriented language.",

            "Storytelling":
                "Use an engaging storytelling style.",

            "Academic":
                "Use formal academic language.",

            "Marketing":
                "Use persuasive and engaging marketing language.",

            "Technical":
                "Use precise and technically clear language.",

            "Executive":
                "Use concise executive-level communication."
        }


        return f"""
You are an expert AI writing assistant.

Your job is to help the user create high-quality written content.

TASK:
{task}

PURPOSE:
{purpose}

PLATFORM:
{platform}

PLATFORM INSTRUCTION:
{platform_instruction[platform]}

WRITING STYLE:
{writing_style}

STYLE INSTRUCTION:
{style_instruction[writing_style]}

TONE:
{tone}

RESPONSE LENGTH:
{length}

LENGTH INSTRUCTION:
{length_instruction[length]}

IMPORTANT RULES:
- Preserve the original meaning when rewriting.
- Do not invent facts.
- Do not add information that is not supported by the user's text.
- Make the response clear and useful.
- Follow the requested platform and tone.
- Return only the final useful response unless explanation is necessary.

USER TEXT:
{text}
""".strip()


    # =====================================================
    # STREAMING FUNCTION
    # =====================================================

    def stream_response(prompt):

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


                if data.get(
                    "done",
                    False
                ):
                    break


            except json.JSONDecodeError:

                continue


    # =====================================================
    # GENERATE BUTTON
    # =====================================================

    if st.button(
        "✨ Generate",
        type="primary",
        use_container_width=True,
        key="generate_writing"
    ):

        if not text.strip():

            st.warning(
                "⚠️ Please enter some text first."
            )


        elif character_count > MAX_CHARACTERS:

            st.error(
                "❌ Input is too long."
            )


        else:

            prompt = create_writing_prompt(
                task,
                purpose,
                platform,
                writing_style,
                tone,
                length,
                text
            )


            st.divider()

            st.subheader(
                "🤖 AI Response"
            )


            try:

                result = st.write_stream(
                    stream_response(prompt)
                )


                st.session_state[
                    "writing_result"
                ] = result


                st.session_state[
                    "writing_prompt"
                ] = prompt


            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to Ollama. "
                    "Make sure Ollama is running."
                )


            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The request took too long. "
                    "Try again with shorter text."
                )


            except Exception as error:

                st.error(
                    f"❌ Something went wrong: {error}"
                )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    if "writing_result" in st.session_state:

        result = st.session_state[
            "writing_result"
        ]


        st.divider()

        st.subheader(
            "📄 Result"
        )


        output_words = len(
            result.split()
        )


        st.caption(
            f"📊 Output: {output_words} words"
        )


        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.write(result)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        st.download_button(

            "💾 Download Response",

            data=result,

            file_name="ai_response.txt",

            mime="text/plain",

            use_container_width=True,

            key="download_writing"
        )


        with st.expander(
            "🔍 Show Generated Prompt"
        ):

            st.code(
                st.session_state[
                    "writing_prompt"
                ],
                language="text"
            )


# =========================================================
# DOCUMENT AI
# =========================================================

with document_tab:

    st.markdown(
        '<div class="section-title">📄 Document AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Upload a PDF and work with its extracted text.'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # PDF UPLOAD
    # =====================================================

    uploaded_file = st.file_uploader(
        "📤 Upload a PDF",
        type=["pdf"],
        key="pdf_upload"
    )


    if uploaded_file:

        try:

            reader = PdfReader(
                uploaded_file
            )


            # =================================================
            # EXTRACT TEXT
            # =================================================

            page_count = len(
                reader.pages
            )


            extracted_text = ""


            for page in reader.pages:

                page_text = page.extract_text()


                if page_text:

                    extracted_text += (
                        page_text + "\n"
                    )


            # =================================================
            # DOCUMENT STATISTICS
            # =================================================

            document_word_count = len(
                extracted_text.split()
            )


            document_character_count = len(
                extracted_text
            )


            # =================================================
            # SUCCESS MESSAGE
            # =================================================

            st.success(
                f"✅ PDF loaded successfully — "
                f"{page_count} pages"
            )


            # =================================================
            # METRICS
            # =================================================

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Pages",
                    page_count
                )


            with col2:

                st.metric(
                    "Words",
                    document_word_count
                )


            with col3:

                st.metric(
                    "Characters",
                    f"{document_character_count:,}"
                )


            st.divider()


            # =================================================
            # EXTRACTED TEXT
            # =================================================

            st.subheader(
                "📖 Extracted Text"
            )


            st.text_area(

                "Document content",

                value=extracted_text,

                height=450,

                key="extracted_document_text"
            )


            # =================================================
            # DOWNLOAD TEXT
            # =================================================

            st.download_button(

                "💾 Download Extracted Text",

                data=extracted_text,

                file_name="extracted_text.txt",

                mime="text/plain",

                use_container_width=True,

                key="download_extracted_text"
            )


            # =================================================
            # STORE DOCUMENT
            # =================================================

            st.session_state[
                "document_text"
            ] = extracted_text


        except Exception as error:

            st.error(
                f"❌ Could not read the PDF: {error}"
            )