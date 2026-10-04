import streamlit as st

from rag.pdf_processor import extract_text_from_pdf
from rag.chunker import create_chunks
from rag.retriever import Retriever
from rag.generator import AnswerGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever()

if "generator" not in st.session_state:
    st.session_state.generator = AnswerGenerator()

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "questions_asked" not in st.session_state:
    st.session_state.questions_asked = 0

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


# ============================================================
# HEADER
# ============================================================

st.title("🎓 AI Teaching Assistant")

st.subheader(
    "AI-Powered Learning Assistant using RAG"
)

st.write(
    "Welcome to the AI Teaching Assistant. "
    "Upload your study material and ask questions "
    "based on your documents."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Study Material")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    st.info(
        "RAG Pipeline:\n\n"
        "PDF → Text → Chunks → Embeddings → "
        "Vector Search → Relevant Context → "
        "Gemini Answer"
    )


# ============================================================
# PROCESS UPLOADED PDF FILES
# ============================================================

if uploaded_files:

    all_chunks = []

    with st.spinner("Processing your PDF documents..."):

        for uploaded_file in uploaded_files:

            try:

                # --------------------------------------------
                # STEP 1: Extract text
                # --------------------------------------------

                pdf_text = extract_text_from_pdf(
                    uploaded_file
                )

                if not pdf_text.strip():

                    st.warning(
                        f"No readable text found in "
                        f"{uploaded_file.name}"
                    )

                    continue

                # --------------------------------------------
                # STEP 2: Create chunks
                # --------------------------------------------

                chunks = create_chunks(pdf_text)

                all_chunks.extend(chunks)

            except Exception as e:

                st.error(
                    f"Error processing "
                    f"{uploaded_file.name}: {e}"
                )

    # --------------------------------------------
    # STEP 3: Create embeddings
    # --------------------------------------------

    if all_chunks:

        st.session_state.retriever.add_documents(
            all_chunks
        )

        st.session_state.document_chunks = all_chunks

        st.session_state.documents_loaded = True

        st.success(
            f"✅ {len(uploaded_files)} document(s) "
            f"processed successfully."
        )

    else:

        st.session_state.documents_loaded = False

        st.warning(
            "No readable text was found in the "
            "uploaded PDF files."
        )


# ============================================================
# UPLOADED DOCUMENTS
# ============================================================

if uploaded_files:

    st.header("📄 Uploaded Documents")

    for uploaded_file in uploaded_files:

        st.write(
            f"📄 **{uploaded_file.name}**"
        )

    st.divider()


# ============================================================
# MAIN AREA
# ============================================================

col1, col2 = st.columns([2, 1])


# ============================================================
# QUESTION AREA
# ============================================================

with col1:

    st.header("💬 Ask Your Question")

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: Explain the OSI model "
            "in simple language."
        )
    )

    ask_button = st.button(
        "🔍 Ask Assistant",
        type="primary"
    )

    if ask_button:

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF first."
            )

        elif not question.strip():

            st.warning(
                "Please enter a question."
            )

        elif not st.session_state.documents_loaded:

            st.warning(
                "The uploaded document could not "
                "be processed."
            )

        else:

            # ==================================================
            # STEP 4: RETRIEVE RELEVANT CONTEXT
            # ==================================================

            with st.spinner(
                "Searching your study material..."
            ):

                results = (
                    st.session_state
                    .retriever
                    .search(
                        question,
                        top_k=3
                    )
                )

            if not results:

                st.warning(
                    "I could not find relevant "
                    "information in the uploaded document."
                )

            else:

                st.session_state.last_results = results

                # --------------------------------------------
                # Build context
                # --------------------------------------------

                context_parts = []

                for result in results:

                    context_parts.append(
                        result["text"]
                    )

                context = "\n\n".join(
                    context_parts
                )

                # ==================================================
                # STEP 5: GENERATE GEMINI ANSWER
                # ==================================================

                with st.spinner(
                    "Generating answer..."
                ):

                    try:

                        answer = (
                            st.session_state
                            .generator
                            .generate_answer(
                                question,
                                context
                            )
                        )

                        st.session_state.last_answer = answer

                        st.session_state.questions_asked += 1

                    except Exception as e:

                        st.error(
                            f"Gemini error: {e}"
                        )


# ============================================================
# LEARNING DASHBOARD
# ============================================================

with col2:

    st.header("📊 Learning Dashboard")

    st.metric(
        "Documents",
        len(uploaded_files)
        if uploaded_files
        else 0
    )

    st.metric(
        "Questions Asked",
        st.session_state.questions_asked
    )

    progress = min(
        st.session_state.questions_asked * 10,
        100
    )

    st.metric(
        "Learning Progress",
        f"{progress}%"
    )


# ============================================================
# AI ANSWER
# ============================================================

if st.session_state.last_answer:

    st.divider()

    st.header("🤖 AI Assistant Answer")

    st.success(
        st.session_state.last_answer
    )


# ============================================================
# RETRIEVED CONTEXT
# ============================================================

if st.session_state.last_results:

    st.header("📚 Retrieved Context")

    for number, result in enumerate(
        st.session_state.last_results,
        start=1
    ):

        with st.expander(
            f"Context {number} "
            f"(Similarity: {result['score']:.3f})"
        ):

            st.write(
                result["text"]
            )


# ============================================================
# RAG PIPELINE
# ============================================================

st.divider()

st.header("🔄 RAG Pipeline")

pipeline1, pipeline2, pipeline3, pipeline4, pipeline5 = (
    st.columns(5)
)

with pipeline1:

    st.write("📄 **1. PDF**")

    st.caption(
        "Upload study material"
    )

with pipeline2:

    st.write("✂️ **2. Chunks**")

    st.caption(
        "Split document text"
    )

with pipeline3:

    st.write("🧠 **3. Embeddings**")

    st.caption(
        "Convert text into vectors"
    )

with pipeline4:

    st.write("🔎 **4. Retrieval**")

    st.caption(
        "Find relevant content"
    )

with pipeline5:

    st.write("🤖 **5. Gemini**")

    st.caption(
        "Generate grounded answer"
    )


# ============================================================
# FUTURE MODULES
# ============================================================

st.divider()

st.header("🚀 Coming Modules")

feature1, feature2, feature3, feature4 = (
    st.columns(4)
)

with feature1:

    st.write(
        "📖 **RAG Question Answering**"
    )

    st.caption(
        "Ask questions from uploaded "
        "study material."
    )

with feature2:

    st.write(
        "📝 **Quiz Generator**"
    )

    st.caption(
        "Generate MCQs and practice questions."
    )

with feature3:

    st.write(
        "🤖 **ML Prediction**"
    )

    st.caption(
        "Predict student performance."
    )

with feature4:

    st.write(
        "🎯 **Personalized Learning**"
    )

    st.caption(
        "Detect weak topics and recommend "
        "learning material."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Teaching Assistant Using RAG | "
    "Capstone Project | "
    "Government Polytechnic, Pune | "
    "2026–2027"
)