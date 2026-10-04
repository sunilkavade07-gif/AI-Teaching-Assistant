# 🤖 AI Teaching Assistant

> An AI-powered learning assistant that helps students study educational content using Retrieval-Augmented Generation (RAG), document processing, semantic retrieval, and Google Gemini.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61.1-red?logo=streamlit)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-orange)](https://ai.google.dev/)
[![RAG](https://img.shields.io/badge/AI-RAG-purple)]()
[![Status](https://img.shields.io/badge/Status-Active-success)]()
[![License](https://img.shields.io/badge/License-Educational-blue)]()

---

## 📌 Project Overview

**AI Teaching Assistant** is an AI-powered educational application designed to help students understand and interact with their study material.

The system uses a **Retrieval-Augmented Generation (RAG)** pipeline to process educational documents, retrieve relevant information, and generate context-aware answers.

Students can upload PDF study material, select subjects, ask questions, choose different study modes, and receive AI-generated answers based on their learning content.

---

## 🎯 Project Goals

The main goal of the project is to provide students with an intelligent and interactive learning assistant that can:

- 📚 Process educational documents
- 📄 Read and process PDF study material
- 🧩 Split documents into meaningful chunks
- 🧠 Retrieve relevant information using semantic retrieval
- 🤖 Generate AI-powered answers
- 💬 Maintain subject-based conversations
- 🎯 Support different study modes
- 🌐 Provide online AI responses
- 📊 Track basic learning activity
- 🔐 Keep API credentials outside the source code

---

## ✨ Key Features

### 📚 PDF & Document Processing

- Upload PDF study material
- Extract text from documents
- Process educational content
- Split documents into smaller chunks
- Store processed document information

### 🧠 Retrieval-Augmented Generation

The application follows a RAG workflow:

```text
PDF / Study Material
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
Embeddings
        ↓
Vector / Semantic Retrieval
        ↓
Relevant Context
        ↓
Google Gemini
        ↓
Context-Aware Answer
