# ✦ GenAI Writing Assistant

> A local AI-powered writing and document intelligence application combining
> content generation, PDF-based Retrieval-Augmented Generation (RAG),
> conversational memory, and RAG evaluation.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![RAG](https://img.shields.io/badge/AI-RAG-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview

GenAI Writing Assistant is a local-first AI application designed to solve
two common problems:

1. **Creating and transforming written content**
2. **Understanding information inside PDF documents**

The application combines a traditional AI writing workflow with a
Retrieval-Augmented Generation (RAG) pipeline.

Users can:

- Rewrite and improve content
- Summarize text
- Change tone
- Expand content
- Extract key points
- Analyze prompt quality
- Optimize content for different publishing formats
- Upload PDF documents
- Ask questions about documents
- Retrieve relevant document sections
- Receive answers grounded in the uploaded document
- See the source pages used for answers
- Continue conversations with document context
- Evaluate RAG performance

The application runs locally using **Ollama**, reducing dependency on
paid cloud LLM APIs during development.

---

# 🎯 Product Goal

The goal was not simply to build a chatbot.

The goal was to design a practical AI assistant that demonstrates how
Generative AI can improve an existing writing and knowledge workflow.

### Original Workflow

A typical workflow looks like:

```text
Write content
      ↓
Manually rewrite / summarize
      ↓
Search documents manually
      ↓
Find relevant information
      ↓
Read multiple pages
      ↓
Write an answer manually

