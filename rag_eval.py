# ============================================================
# RAG EVALUATION DATASET
# ============================================================
# Test questions used to measure the quality of our RAG system.
#
# Each test contains:
# - Question asked by the user
# - Expected answer
# - Expected source page
#
# We will use these later to evaluate retrieval and answers.
# ============================================================


EVALUATION_DATASET = [

    {
        "question": "What is the main topic of the document?",

        "expected_answer": (
            "Replace this with the correct answer "
            "from your PDF."
        ),

        "expected_pages": [
            1
        ]
    },

    {
        "question": "What are the main points discussed?",

        "expected_answer": (
            "Replace this with the correct answer "
            "from your PDF."
        ),

        "expected_pages": [
            1
        ]
    },

    {
        "question": "What is one important detail mentioned in the document?",

        "expected_answer": (
            "Replace this with the correct answer "
            "from your PDF."
        ),

        "expected_pages": [
            1
        ]
    }
]