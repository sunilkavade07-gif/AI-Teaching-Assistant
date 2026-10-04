# ============================================================
# AI TEACHING ASSISTANT
# COMPLETE STREAMLIT APPLICATION
# ============================================================

import time
import hashlib
import json
import uuid
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st


# ============================================================
# IMPORTS
# ============================================================

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks
from retriever import Retriever
from generator import generate_answer

from storage import (
    initialize_database,
    save_pdf,
    document_exists,
    save_processed_document,
    load_processed_document,
    update_chunk_count,
    get_user_documents,
    save_question,
    get_user_questions,
    get_progress,
    update_progress,
)


# ============================================================
# ONLINE AI
# ============================================================

try:
    from online import ask_online
except ImportError:
    try:
        from rag.online import ask_online
    except ImportError:
        ask_online = None



# ============================================================
# BEAUTIFUL STUDY ANSWER RENDERER
# ============================================================

import re
import html
from html import escape


def _inline_html(text):
    value = escape(str(text))
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    return value


def _is_heading(line):
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^#{1,4}\s+", stripped):
        return True
    if re.match(r"^(?:[0-9]+[.)]|[0-9]+️⃣)\s+", stripped):
        return len(re.sub(r"[*#]", "", stripped)) <= 110
    if re.match(r"^(?:💡|📌|🎯|🧠|🏆|⚠️|✅|❌)\s+", stripped):
        return len(stripped) <= 120
    return False


def _clean_heading(line):
    line = line.strip()
    return re.sub(r"^#{1,4}\s+", "", line)


def _render_block(lines):
    html_parts = []
    list_items = []
    paragraphs = []

    def flush_list():
        nonlocal list_items
        if list_items:
            html_parts.append(
                '<ul class="ta-answer-list">'
                + ''.join(f'<li>{_inline_html(item)}</li>' for item in list_items)
                + '</ul>'
            )
            list_items = []

    def flush_paragraphs():
        nonlocal paragraphs
        if paragraphs:
            text = " ".join(x.strip() for x in paragraphs)
            if text:
                html_parts.append(
                    f'<p class="ta-answer-text">{_inline_html(text)}</p>'
                )
            paragraphs = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_list()
            flush_paragraphs()
            continue

        bullet = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line)
        if bullet:
            flush_paragraphs()
            list_items.append(bullet.group(1))
            continue

        flush_list()
        paragraphs.append(line)

    flush_list()
    flush_paragraphs()
    return "".join(html_parts)



def _enhance_retrieval(question, semantic_results, all_chunks, max_results=6):
    """
    Improve RAG recall by combining semantic retrieval with a small
    lexical/phrase fallback.

    This is especially useful for OCR/PDF text where the exact wording
    of a question may be present in a chunk but the embedding score is
    lower than unrelated chunks.
    """

    question_text = re.sub(
        r"\s+",
        " ",
        str(question).lower()
    ).strip()

    if not question_text:
        return semantic_results[:max_results]

    stop_words = {
        "what", "is", "are", "the", "a", "an", "of", "in", "on",
        "to", "for", "and", "or", "with", "from", "explain",
        "define", "describe", "about", "give", "write", "how",
        "why", "can", "does", "do", "which", "this", "that"
    }

    question_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", question_text)
        if len(token) >= 3 and token not in stop_words
    }

    merged = {}

    # Keep semantic results first.
    for result in semantic_results or []:
        text_value = str(result.get("text", "")).strip()
        if not text_value:
            continue

        merged[text_value] = {
            "text": text_value,
            "score": float(result.get("score", 0.0)),
            "_semantic": float(result.get("score", 0.0)),
            "_lexical": 0.0,
        }

    # Scan all stored chunks for exact phrases and important keywords.
    for chunk in all_chunks or []:
        chunk_text = str(chunk).strip()

        if not chunk_text:
            continue

        chunk_lower = re.sub(
            r"\s+",
            " ",
            chunk_text.lower()
        )

        token_set = set(
            re.findall(r"[a-z0-9]+", chunk_lower)
        )

        overlap = (
            len(question_tokens & token_set)
            / max(len(question_tokens), 1)
        )

        phrase_bonus = 0.0

        # Full question phrase.
        if len(question_text) >= 6 and question_text in chunk_lower:
            phrase_bonus = 0.45

        # Strong bonus for the important content words appearing together.
        important_phrase = " ".join(
            sorted(question_tokens)
        )

        if (
            len(question_tokens) >= 2
            and all(
                token in chunk_lower
                for token in question_tokens
            )
        ):
            phrase_bonus = max(
                phrase_bonus,
                0.25
            )

        lexical_score = min(
            1.0,
            overlap + phrase_bonus
        )

        if lexical_score <= 0:
            continue

        existing = merged.get(chunk_text)

        if existing:
            existing["_lexical"] = max(
                existing["_lexical"],
                lexical_score
            )
        else:
            merged[chunk_text] = {
                "text": chunk_text,
                "score": lexical_score,
                "_semantic": 0.0,
                "_lexical": lexical_score,
            }

    ranked = []

    for item in merged.values():

        combined_score = min(
            1.0,
            (
                item["_semantic"] * 0.70
                + item["_lexical"] * 0.30
            )
        )

        # Exact phrase should be strongly preferred.
        if item["_lexical"] >= 0.45:
            combined_score = max(
                combined_score,
                item["_lexical"]
            )

        ranked.append(
            {
                "text": item["text"],
                "score": float(combined_score),
            }
        )

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked[:max_results]


def render_beautiful_answer(answer, source="AI Answer"):
    """
    Render PDF/RAG and Online AI answers in unmistakably
    different colored cards.

    This renderer uses INLINE styles on the actual container,
    rather than relying on global CSS classes.
    """

    if not answer:
        st.warning("No answer was generated.")
        return

    source_text = str(source).lower()

    if "pdf" in source_text or "rag" in source_text:
        background = "#EAF4FF"
        border = "#1976D2"
        title_color = "#0D4F8B"
        icon = "📚"
        title = "PDF / RAG ANSWER"
        label = "Based on your uploaded study material"

    elif "online" in source_text or "gemini" in source_text:
        background = "#EAF8EE"
        border = "#16A05D"
        title_color = "#08783F"
        icon = "🌐"
        title = "ONLINE AI ANSWER"
        label = "Generated independently by Online AI"

    else:
        background = "#F3EDFF"
        border = "#7C3AED"
        title_color = "#5B21B6"
        icon = "🤖"
        title = "AI ANSWER"
        label = "AI-generated explanation"

    # Escape the answer so arbitrary answer text cannot break
    # the card's HTML.
    safe_answer = html.escape(
        str(answer),
        quote=False
    )

    # Keep line breaks visible.
    safe_answer = safe_answer.replace(
        "\n",
        "<br>"
    )

    st.markdown(
        f"""
        <div style="
            background: {background};
            border: 2px solid {border};
            border-left: 8px solid {border};
            border-radius: 18px;
            padding: 22px 24px;
            margin: 12px 0 26px 0;
            box-shadow: 0 6px 18px rgba(0,0,0,0.10);
            width: 100%;
            box-sizing: border-box;
        ">

            <div style="
                color: {title_color};
                font-size: 21px;
                font-weight: 800;
                padding-bottom: 10px;
                margin-bottom: 5px;
                border-bottom: 1px solid rgba(0,0,0,0.12);
            ">
                {icon}&nbsp;&nbsp;{title}
            </div>

            <div style="
                color: #4B5563;
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 16px;
            ">
                {label}
            </div>

            <div style="
                background: rgba(255,255,255,0.72);
                border-radius: 12px;
                padding: 18px 20px;
                color: #202124;
                font-size: 16px;
                line-height: 1.75;
                overflow-wrap: anywhere;
            ">
                {safe_answer}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# STARTUP TIMER
# ============================================================

APP_START_TIME = time.time()


# ============================================================
# SUBJECT CHAT STORAGE
# ============================================================
#
# Each subject has its own conversation history.
# Example:
#
# Subject: Operating System
#   ├── Question
#   ├── PDF answer
#   ├── Online answer
#   └── Question ...
#
# Subjects are stored locally so they remain available after
# restarting Streamlit.
# ============================================================

SUBJECTS_FILE = (
    Path(__file__).resolve().parent
    / "subjects.json"
)

MEMORY_FILE = (
    Path(__file__).resolve().parent
    / "subject_memory.json"
)


# ============================================================
# LONG-TERM SUBJECT MEMORY
# ============================================================

def load_memories():

    if not MEMORY_FILE.exists():
        return {}

    try:

        with open(
            MEMORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except Exception as error:

        print(
            "MEMORY LOAD ERROR:",
            error
        )

        return {}


def save_memories(memories):

    try:

        with open(
            MEMORY_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                memories,
                file,
                indent=2,
                ensure_ascii=False
            )

        return True

    except Exception as error:

        print(
            "MEMORY SAVE ERROR:",
            error
        )

        return False


def get_subject_memory(subject_id):

    memories = load_memories()

    memory = memories.get(
        str(subject_id),
        {}
    )

    if not isinstance(memory, dict):
        memory = {}

    memory.setdefault(
        "summary",
        ""
    )

    memory.setdefault(
        "important_points",
        []
    )

    memory.setdefault(
        "topics_discussed",
        []
    )

    memory.setdefault(
        "student_preferences",
        []
    )

    memory.setdefault(
        "questions_count",
        0
    )

    return memory


def save_subject_memory(
    subject_id,
    memory
):

    memories = load_memories()

    memories[str(subject_id)] = memory

    return save_memories(
        memories
    )


def update_subject_memory(
    subject_id,
    question,
    answer
):
    """
    Lightweight persistent memory.

    This intentionally does NOT call Gemini on every question.
    It stores useful subject-level information locally and
    periodically rebuilds a compact memory summary.

    The latest 20 chats are still available separately.
    """

    memory = get_subject_memory(
        subject_id
    )

    memory["questions_count"] = (
        int(
            memory.get(
                "questions_count",
                0
            )
        ) + 1
    )

    # --------------------------------------------------------
    # Store recent question topics.
    # --------------------------------------------------------

    recent_questions = memory.setdefault(
        "topics_discussed",
        []
    )

    clean_question = (
        str(question)
        .strip()
    )

    if clean_question:

        recent_questions.append(
            clean_question
        )

        # Keep a compact list.
        memory["topics_discussed"] = (
            recent_questions[-30:]
        )


    # --------------------------------------------------------
    # Detect simple study preferences from the conversation.
    # --------------------------------------------------------

    preference_text = (
        clean_question
        + " "
        + str(answer)
    ).lower()

    preferences = memory.setdefault(
        "student_preferences",
        []
    )

    preference_rules = [
        (
            "simple explanations",
            [
                "simple language",
                "in simple language",
                "easy explanation",
                "explain simply",
                "easy to understand"
            ]
        ),

        (
            "exam-oriented answers",
            [
                "exam answer",
                "for exam",
                "marks",
                "important points"
            ]
        ),

        (
            "step-by-step explanations",
            [
                "step by step",
                "step-by-step",
                "steps"
            ]
        ),

        (
            "examples are useful",
            [
                "give example",
                "with example",
                "example"
            ]
        )
    ]

    for label, patterns in preference_rules:

        if any(
            pattern in preference_text
            for pattern in patterns
        ):

            if label not in preferences:

                preferences.append(
                    label
                )


    memory["student_preferences"] = (
        preferences[:10]
    )


    # --------------------------------------------------------
    # Keep a compact extract of important answer content.
    # --------------------------------------------------------
    #
    # We keep complete older chats in subjects.json.
    # This memory file is intentionally compact.
    # --------------------------------------------------------

    answer_text = (
        str(answer)
        .strip()
    )

    if answer_text:

        # Keep only useful-sized snippets.
        snippet = answer_text[:700]

        important_points = memory.setdefault(
            "important_points",
            []
        )

        important_points.append(
            snippet
        )

        memory["important_points"] = (
            important_points[-20:]
        )


    # --------------------------------------------------------
    # Build a compact subject summary.
    # --------------------------------------------------------

    topic_count = len(
        memory.get(
            "topics_discussed",
            []
        )
    )

    preference_list = memory.get(
        "student_preferences",
        []
    )

    preference_text = (
        ", ".join(
            preference_list
        )
        if preference_list
        else "No specific preference detected."
    )

    memory["summary"] = (
        f"This subject has {memory['questions_count']} "
        f"recorded question(s). "
        f"About {topic_count} recent topic/question entries "
        f"have been discussed. "
        f"Student preferences detected: "
        f"{preference_text}"
    )

    memory["updated_at"] = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    save_subject_memory(
        subject_id,
        memory
    )

    return memory


def build_memory_context(
    subject_id
):
    """
    Convert persistent memory into a compact prompt section.
    """

    memory = get_subject_memory(
        subject_id
    )

    parts = []

    summary = memory.get(
        "summary",
        ""
    )

    if summary:
        parts.append(
            "LONG-TERM SUBJECT SUMMARY:\n"
            + summary
        )


    preferences = memory.get(
        "student_preferences",
        []
    )

    if preferences:

        parts.append(
            "STUDENT STUDY PREFERENCES:\n- "
            + "\n- ".join(
                preferences
            )
        )


    topics = memory.get(
        "topics_discussed",
        []
    )

    if topics:

        parts.append(
            "RECENTLY DISCUSSED TOPICS:\n- "
            + "\n- ".join(
                topics[-15:]
            )
        )


    important_points = memory.get(
        "important_points",
        []
    )

    if important_points:

        parts.append(
            "SELECTED PREVIOUS ANSWER POINTS:\n- "
            + "\n- ".join(
                important_points[-8:]
            )
        )


    if not parts:

        return ""

    return (
        "\n\n"
        + "\n\n".join(
            parts
        )
        + "\n\n"
    )


def load_subjects():

    if not SUBJECTS_FILE.exists():

        return []


    try:

        with open(
            SUBJECTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        if isinstance(data, list):

            return data


        return []


    except Exception as error:

        print(
            "SUBJECT LOAD ERROR:",
            error
        )

        return []


def save_subjects(subjects):

    try:

        with open(
            SUBJECTS_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                subjects,
                file,
                indent=2,
                ensure_ascii=False
            )

        return True


    except Exception as error:

        print(
            "SUBJECT SAVE ERROR:",
            error
        )

        return False


def create_subject(subject_name):

    subject_name = (
        subject_name
        .strip()
    )

    if not subject_name:

        return None


    subjects = load_subjects()


    # Prevent duplicate subject names.
    for subject in subjects:

        if (
            subject.get("name", "").strip().lower()
            == subject_name.lower()
        ):

            return subject["id"]


    subject = {

        "id": str(
            uuid.uuid4()
        ),

        "name": subject_name,

        "created_at": (
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),

        "messages": []

    }


    subjects.append(
        subject
    )

    save_subjects(
        subjects
    )

    return subject["id"]


def delete_subject(subject_id):

    subjects = load_subjects()

    remaining = [
        subject
        for subject in subjects
        if subject.get("id") != subject_id
    ]

    if len(remaining) == len(subjects):

        return False


    # Always keep at least one subject.
    if not remaining:

        remaining.append(
            {
                "id": str(uuid.uuid4()),
                "name": "General Study",
                "created_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "messages": []
            }
        )


    return save_subjects(
        remaining
    )


def get_subject(subject_id):

    subjects = load_subjects()

    for subject in subjects:

        if subject.get("id") == subject_id:

            return subject


    return None


def add_subject_message(
    subject_id,
    question,
    answer,
    study_mode,
    answer_mode
):

    subjects = load_subjects()

    for subject in subjects:

        if subject.get("id") != subject_id:
            continue


        subject.setdefault(
            "messages",
            []
        )

        subject["messages"].append(
            {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),

                "question": question,

                "answer": answer,

                "study_mode": study_mode,

                "answer_mode": answer_mode
            }
        )

        # Keep the latest 100 exchanges per subject.
        # This gives the student much more conversation memory
        # while preventing the local JSON file from growing
        # without limit.
        subject["messages"] = (
            subject["messages"][-100:]
        )

        save_subjects(
            subjects
        )

        return True


    return False


# ============================================================
# SUBJECT SESSION INITIALIZATION
# ============================================================

if "subjects" not in st.session_state:

    subjects = load_subjects()

    if not subjects:

        default_subject_id = create_subject(
            "General Study"
        )

        subjects = load_subjects()

    st.session_state.subjects = subjects


if "current_subject_id" not in st.session_state:

    if st.session_state.subjects:

        st.session_state.current_subject_id = (
            st.session_state.subjects[0]["id"]
        )

    else:

        st.session_state.current_subject_id = None


# ============================================================
# DATABASE
# ============================================================

initialize_database()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CACHED RETRIEVER
# ============================================================

@st.cache_resource
def get_retriever():

    print("=" * 60)
    print("CREATING RETRIEVER")
    print("=" * 60)

    start_time = time.time()

    retriever = Retriever()

    print(
        f"RETRIEVER READY: "
        f"{time.time() - start_time:.2f} seconds"
    )

    print("=" * 60)

    return retriever


# ============================================================
# RETRIEVER
# ============================================================

if "retriever" not in st.session_state:

    st.session_state.retriever = get_retriever()


# ============================================================
# SESSION STATE
# ============================================================

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


if "last_question" not in st.session_state:
    st.session_state.last_question = ""


if "last_online_answer" not in st.session_state:
    st.session_state.last_online_answer = ""


if "last_question" not in st.session_state:
    st.session_state.last_question = ""


if "saved_data_loaded" not in st.session_state:
    st.session_state.saved_data_loaded = False


if "loaded_document_hashes" not in st.session_state:
    st.session_state.loaded_document_hashes = set()


if "answer_mode" not in st.session_state:
    st.session_state.answer_mode = "📚 PDF / RAG"


# ============================================================
# CURRENT USER
# TEMPORARY USER
# LOGIN SYSTEM WILL BE ADDED LATER
# ============================================================

CURRENT_USER = "guest"


# ============================================================
# LOAD SAVED DOCUMENTS
# ============================================================

def load_saved_documents():

    documents = get_user_documents(
        CURRENT_USER
    )

    if not documents:
        return

    all_chunks = []
    all_embeddings = []

    loaded_hashes = set()

    for document in documents:

        file_hash = document["file_hash"]

        try:

            data = load_processed_document(
                file_hash
            )

        except Exception as error:

            print(
                f"Could not load saved document "
                f"{file_hash}: {error}"
            )

            continue

        if data is None:
            continue

        chunks = data.get(
            "chunks",
            []
        )

        embeddings = data.get(
            "embeddings",
            None
        )

        if not chunks:
            continue

        if embeddings is None:
            continue

        all_chunks.extend(
            chunks
        )

        all_embeddings.append(
            np.asarray(embeddings)
        )

        loaded_hashes.add(
            file_hash
        )

    if not all_chunks:
        return

    combined_embeddings = np.vstack(
        all_embeddings
    )

    retriever = st.session_state.retriever

    retriever.chunks = all_chunks

    retriever.embeddings = combined_embeddings

    # --------------------------------------------------------
    # NORMALIZED EMBEDDINGS
    # --------------------------------------------------------

    norms = np.linalg.norm(
        combined_embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1

    retriever.normalized_embeddings = (
        combined_embeddings / norms
    )

    st.session_state.document_chunks = (
        all_chunks
    )

    st.session_state.documents_loaded = True

    st.session_state.loaded_document_hashes = (
        loaded_hashes
    )

    print(
        f"⚡ Loaded {len(loaded_hashes)} "
        f"saved document(s) from storage."
    )

    print(
        f"⚡ Loaded {len(all_chunks)} chunks "
        f"from saved storage."
    )


# ============================================================
# LOAD SAVED DATA ONLY ONCE
# ============================================================

if not st.session_state.saved_data_loaded:

    try:

        saved_start = time.time()

        load_saved_documents()

        print(
            f"SAVED DATA LOAD TIME: "
            f"{time.time() - saved_start:.2f} seconds"
        )

    except Exception as error:

        print(
            "Saved data loading error:",
            error
        )

    st.session_state.saved_data_loaded = True


# ============================================================
# HEADER
# ============================================================

# Resolve the active subject before the header uses it.
current_subject = get_subject(
    st.session_state.current_subject_id
) if st.session_state.current_subject_id else None

st.title(
    "🎓 AI Teaching Assistant"
)

st.subheader(
    "AI-Powered Learning Assistant using RAG"
)

st.write(
    "Upload your study material, ask questions, "
    "learn from your PDFs, or use Online AI."
)

if current_subject:

    st.caption(
        f"📚 Current Subject: **{current_subject['name']}**"
    )


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # ========================================================
    # SUBJECTS / CHAT
    # ========================================================

    st.header(
        "📚 My Subjects"
    )

    # Refresh subject list from disk.
    st.session_state.subjects = load_subjects()

    if not st.session_state.subjects:

        create_subject(
            "General Study"
        )

        st.session_state.subjects = (
            load_subjects()
        )


    subject_names = [
        subject["name"]
        for subject in st.session_state.subjects
    ]

    subject_ids = [
        subject["id"]
        for subject in st.session_state.subjects
    ]


    current_index = 0

    if (
        st.session_state.current_subject_id
        in subject_ids
    ):

        current_index = (
            subject_ids.index(
                st.session_state.current_subject_id
            )
        )


    selected_subject_name = st.selectbox(
        "Continue Subject Chat",
        subject_names,
        index=current_index,
        key="subject_selector"
    )


    selected_subject_index = (
        subject_names.index(
            selected_subject_name
        )
    )

    selected_subject_id = (
        subject_ids[selected_subject_index]
    )


    # Detect subject change.
    if (
        selected_subject_id
        != st.session_state.current_subject_id
    ):

        st.session_state.current_subject_id = (
            selected_subject_id
        )

        # Clear only the currently displayed answer.
        # The subject's old conversation remains saved.
        st.session_state.last_answer = ""
        st.session_state.last_online_answer = ""
        st.session_state.last_results = []

        st.rerun()


    current_subject = get_subject(
        st.session_state.current_subject_id
    )


    if current_subject:

        st.caption(
            f"💬 {len(current_subject.get('messages', []))} "
            "saved conversation(s)"
        )

        current_memory = get_subject_memory(
            current_subject["id"]
        )

        st.caption(
            "🧠 Long-term memory: "
            f"{current_memory.get('questions_count', 0)} "
            "question(s)"
        )

        with st.expander(
            "🧠 View Subject Memory"
        ):

            st.write(
                current_memory.get(
                    "summary",
                    "No memory yet."
                )
            )

            preferences = current_memory.get(
                "student_preferences",
                []
            )

            if preferences:

                st.write(
                    "**Student preferences:**"
                )

                for preference in preferences:

                    st.write(
                        "• " + preference
                    )


    # --------------------------------------------------------
    # CREATE SUBJECT
    # --------------------------------------------------------

    with st.expander(
        "➕ Create New Subject"
    ):

        new_subject_name = st.text_input(
            "Subject name",
            placeholder="Example: Operating System",
            key="new_subject_name"
        )

        if st.button(
            "Create Subject",
            use_container_width=True
        ):

            if not new_subject_name.strip():

                st.warning(
                    "Enter a subject name."
                )

            else:

                new_id = create_subject(
                    new_subject_name
                )

                if new_id:

                    st.session_state.subjects = (
                        load_subjects()
                    )

                    st.session_state.current_subject_id = (
                        new_id
                    )

                    st.success(
                        f"Created: {new_subject_name.strip()}"
                    )

                    st.rerun()


    # --------------------------------------------------------
    # DELETE SUBJECT
    # --------------------------------------------------------

    if current_subject:

        if st.button(
            "🗑️ Delete Current Subject",
            use_container_width=True
        ):

            delete_subject(
                st.session_state.current_subject_id
            )

            remaining_subjects = (
                load_subjects()
            )

            st.session_state.subjects = (
                remaining_subjects
            )

            st.session_state.current_subject_id = (
                remaining_subjects[0]["id"]
            )

            st.session_state.last_answer = ""
            st.session_state.last_online_answer = ""
            st.session_state.last_results = []

            st.rerun()


    st.divider()


    # ========================================================
    # STUDY MATERIAL
    # ========================================================

    st.header(
        "📚 Study Material"
    )

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

    st.divider()

    # --------------------------------------------------------
    # SAVED DOCUMENTS
    # --------------------------------------------------------

    st.subheader(
        "💾 Saved Documents"
    )

    saved_documents = get_user_documents(
        CURRENT_USER
    )

    if saved_documents:

        for document in saved_documents:

            st.write(
                "📄 "
                + document["filename"]
            )

    else:

        st.caption(
            "No saved documents yet."
        )


# ============================================================
# PROCESS UPLOADED PDF FILES
# ============================================================

if uploaded_files:

    new_documents = 0
    loaded_from_cache = 0

    for uploaded_file in uploaded_files:

        try:

            # ------------------------------------------------
            # READ PDF
            # ------------------------------------------------

            file_bytes = (
                uploaded_file.getvalue()
            )

            if not file_bytes:

                st.warning(
                    f"{uploaded_file.name} "
                    "is empty."
                )

                continue

            # ------------------------------------------------
            # HASH
            # ------------------------------------------------

            file_hash = hashlib.sha256(
                file_bytes
            ).hexdigest()

            # ------------------------------------------------
            # ALREADY LOADED IN THIS SESSION
            # ------------------------------------------------

            if (
                file_hash
                in st.session_state.loaded_document_hashes
            ):

                st.info(
                    f"⚡ {uploaded_file.name} "
                    "already loaded."
                )

                continue

            # ------------------------------------------------
            # CHECK DATABASE
            # ------------------------------------------------

            existing_document = (
                document_exists(
                    CURRENT_USER,
                    file_hash
                )
            )

            # =================================================
            # LOAD SAVED PROCESSING
            # =================================================

            if existing_document:

                cache_start = time.time()

                saved_data = (
                    load_processed_document(
                        file_hash
                    )
                )

                if saved_data:

                    chunks = saved_data.get(
                        "chunks",
                        []
                    )

                    embeddings = saved_data.get(
                        "embeddings",
                        None
                    )

                    if (
                        chunks
                        and embeddings is not None
                    ):

                        retriever = (
                            st.session_state.retriever
                        )

                        embeddings = np.asarray(
                            embeddings
                        )

                        # ------------------------------------
                        # ADD ONLY ONCE
                        # ------------------------------------

                        if (
                            file_hash
                            not in
                            st.session_state
                            .loaded_document_hashes
                        ):

                            if (
                                retriever.embeddings
                                is None
                            ):

                                retriever.chunks = (
                                    list(chunks)
                                )

                                retriever.embeddings = (
                                    embeddings.copy()
                                )

                            else:

                                retriever.chunks.extend(
                                    chunks
                                )

                                retriever.embeddings = (
                                    np.vstack(
                                        [
                                            retriever.embeddings,
                                            embeddings
                                        ]
                                    )
                                )

                            # --------------------------------
                            # NORMALIZE
                            # --------------------------------

                            norms = np.linalg.norm(
                                retriever.embeddings,
                                axis=1,
                                keepdims=True
                            )

                            norms[norms == 0] = 1

                            retriever.normalized_embeddings = (
                                retriever.embeddings
                                / norms
                            )

                            st.session_state.document_chunks = (
                                retriever.chunks
                            )

                            st.session_state.documents_loaded = (
                                True
                            )

                            st.session_state.loaded_document_hashes.add(
                                file_hash
                            )

                            loaded_from_cache += 1

                            print(
                                f"⚡ SAVED DOCUMENT LOADED: "
                                f"{uploaded_file.name}"
                            )

                            print(
                                f"⚡ CACHE LOAD TIME: "
                                f"{time.time() - cache_start:.2f} seconds"
                            )

                            continue

            # =================================================
            # NEW DOCUMENT
            # =================================================

            with st.spinner(
                f"📄 Processing {uploaded_file.name}..."
            ):

                # ---------------------------------------------
                # SAVE ORIGINAL PDF
                # ---------------------------------------------

                saved_document = save_pdf(
                    CURRENT_USER,
                    uploaded_file.name,
                    file_bytes
                )

                # ---------------------------------------------
                # PDF EXTRACTION
                # ---------------------------------------------

                extraction_start = time.time()

                pdf_text = extract_text_from_pdf(
                    uploaded_file
                )

                extraction_time = (
                    time.time()
                    - extraction_start
                )

                print(
                    f"PDF EXTRACTION TIME: "
                    f"{extraction_time:.2f} seconds"
                )

                if not pdf_text.strip():

                    st.warning(
                        f"No readable text found in "
                        f"{uploaded_file.name}"
                    )

                    continue

                # ---------------------------------------------
                # CHUNKING
                # ---------------------------------------------

                chunk_start = time.time()

                chunks = create_chunks(
                    pdf_text
                )

                chunk_time = (
                    time.time()
                    - chunk_start
                )

                print(
                    f"CHUNKING TIME: "
                    f"{chunk_time:.2f} seconds"
                )

                if not chunks:

                    st.warning(
                        f"No chunks created for "
                        f"{uploaded_file.name}"
                    )

                    continue

                # ---------------------------------------------
                # EMBEDDINGS
                # ---------------------------------------------

                embedding_start = time.time()

                retriever = (
                    st.session_state.retriever
                )

                new_embeddings = (
                    retriever.model.encode(
                        chunks,
                        convert_to_numpy=True,
                        batch_size=32,
                        show_progress_bar=True,
                        normalize_embeddings=True
                    )
                )

                embedding_time = (
                    time.time()
                    - embedding_start
                )

                print(
                    f"EMBEDDING TIME: "
                    f"{embedding_time:.2f} seconds"
                )

                # ---------------------------------------------
                # ADD EMBEDDINGS
                # ---------------------------------------------

                if retriever.embeddings is None:

                    retriever.chunks = (
                        list(chunks)
                    )

                    retriever.embeddings = (
                        new_embeddings
                    )

                else:

                    retriever.chunks.extend(
                        chunks
                    )

                    retriever.embeddings = (
                        np.vstack(
                            [
                                retriever.embeddings,
                                new_embeddings
                            ]
                        )
                    )

                # ---------------------------------------------
                # NORMALIZED EMBEDDINGS
                # ---------------------------------------------

                retriever.normalized_embeddings = (
                    retriever.embeddings
                )

                # ---------------------------------------------
                # SAVE PROCESSED DATA
                # ---------------------------------------------

                save_processed_document(
                    saved_document["file_hash"],
                    chunks,
                    new_embeddings
                )

                update_chunk_count(
                    CURRENT_USER,
                    saved_document["file_hash"],
                    len(chunks)
                )

                # ---------------------------------------------
                # UPDATE SESSION
                # ---------------------------------------------

                st.session_state.document_chunks = (
                    retriever.chunks
                )

                st.session_state.documents_loaded = (
                    True
                )

                st.session_state.loaded_document_hashes.add(
                    file_hash
                )

                new_documents += 1

                # ---------------------------------------------
                # PERFORMANCE
                # ---------------------------------------------

                print(
                    "=" * 60
                )

                print(
                    f"DOCUMENT: {uploaded_file.name}"
                )

                print(
                    f"PDF SIZE: "
                    f"{len(file_bytes) / (1024 * 1024):.2f} MB"
                )

                print(
                    f"EXTRACTION: "
                    f"{extraction_time:.2f}s"
                )

                print(
                    f"CHUNKING: "
                    f"{chunk_time:.2f}s"
                )

                print(
                    f"EMBEDDING: "
                    f"{embedding_time:.2f}s"
                )

                print(
                    "=" * 60
                )

        except Exception as error:

            st.error(
                f"Error processing "
                f"{uploaded_file.name}: "
                f"{error}"
            )

    # ========================================================
    # PROCESSING RESULT
    # ========================================================

    if new_documents > 0:

        st.success(
            f"✅ {new_documents} new document(s) "
            "processed and permanently saved."
        )

    if loaded_from_cache > 0:

        st.success(
            f"⚡ {loaded_from_cache} document(s) "
            "loaded from saved storage."
        )


# ============================================================
# SAVED DOCUMENTS
# ============================================================

saved_documents = get_user_documents(
    CURRENT_USER
)


if saved_documents:

    st.header(
        "📄 Your Study Material"
    )

    for document in saved_documents:

        st.write(
            f"📄 **{document['filename']}** "
            f"— {document['chunks_count']} chunks"
        )

    st.divider()


# ============================================================
# MAIN AREA
# ============================================================

col1, col2 = st.columns(
    [2, 1]
)


# ============================================================
# QUESTION AREA
# ============================================================

with col1:

    st.header(
        "💬 Ask Your Question"
    )

    # --------------------------------------------------------
    # ANSWER MODE
    # --------------------------------------------------------

    answer_mode = st.radio(
        "Choose Answer Mode",
        [
            "📚 PDF / RAG",
            "🌐 Online AI",
            "🔀 PDF + Online"
        ],
        horizontal=True
    )

    st.session_state.answer_mode = (
        answer_mode
    )

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: Explain the OSI model "
            "in simple language."
        )
    )

    # --------------------------------------------------------
    # STUDY MODE
    # --------------------------------------------------------

    study_mode = st.selectbox(
        "🎯 Choose Study Mode",
        [
            "Normal Answer",
            "📖 Simple Explanation",
            "📝 Exam Answer",
            "⚡ Quick Revision",
            "❓ MCQ Practice",
            "🎯 Predicted Questions"
        ],
        help=(
            "Choose how the AI should format the answer. "
            "The same study mode is applied to PDF and Online AI "
            "in comparison mode."
        )
    )

    # ========================================================
    # PREDICTED QUESTION OPTIONS
    # ========================================================

    selected_prediction_marks = []

    prediction_questions_per_mark = 3

    if study_mode == "🎯 Predicted Questions":

        st.markdown("### 📌 Select Question Marks")

        st.caption(
            "Select any combination of marks. "
            "For example: 2 + 4, 4 + 6, or 2 + 5 + 10."
        )

        mark_col1, mark_col2, mark_col3 = st.columns(3)
        mark_col4, mark_col5, _ = st.columns(3)

        with mark_col1:
            mark_2 = st.checkbox(
                "2 Marks",
                key="predict_2_marks"
            )

        with mark_col2:
            mark_4 = st.checkbox(
                "4 Marks",
                key="predict_4_marks"
            )

        with mark_col3:
            mark_5 = st.checkbox(
                "5 Marks",
                key="predict_5_marks"
            )

        with mark_col4:
            mark_6 = st.checkbox(
                "6 Marks",
                key="predict_6_marks"
            )

        with mark_col5:
            mark_10 = st.checkbox(
                "10 Marks",
                key="predict_10_marks"
            )

        if mark_2:
            selected_prediction_marks.append("2 Marks")

        if mark_4:
            selected_prediction_marks.append("4 Marks")

        if mark_5:
            selected_prediction_marks.append("5 Marks")

        if mark_6:
            selected_prediction_marks.append("6 Marks")

        if mark_10:
            selected_prediction_marks.append("10 Marks")

        prediction_questions_per_mark = st.number_input(
            "🔢 Questions per selected mark type",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help=(
                "Example: select 2 + 4 + 6 and choose 3. "
                "The AI will generate 3 questions for each "
                "selected mark type."
            )
        )

        if selected_prediction_marks:

            st.info(
                "Selected: "
                + " + ".join(selected_prediction_marks)
                + f"  •  {prediction_questions_per_mark} "
                  "questions for each selected type"
            )

        else:

            st.warning(
                "Please select at least one question-mark type."
            )

    study_instructions = {
        "Normal Answer": (
            "Answer normally and clearly. Give a useful student-friendly explanation."
        ),
        "📖 Simple Explanation": (
            "Explain the topic in very simple student-friendly language. "
            "Use a small example or analogy when useful. Avoid unnecessary complexity."
        ),
        "📝 Exam Answer": (
            "Write an exam-ready answer. Start with a definition, then key points, "
            "important details, and a suitable example if relevant. Keep the structure "
            "easy to write in an exam."
        ),
        "⚡ Quick Revision": (
            "Give a quick revision note using short headings, bullet points, "
            "keywords, important facts, and a short memory trick when useful."
        ),
        "❓ MCQ Practice": (
            "Create 5 multiple-choice questions related to the student's question. "
            "Give four options (A-D), then provide the correct answer and a one-line "
            "explanation for each question. If PDF/RAG context is available, base the "
            "questions primarily on that material."
        ),
        "🎯 Predicted Questions": (
            "Generate likely exam questions from the available study material. "
            "This is an AI-based prediction, not a guarantee of the actual exam. "
            "Identify important topics, repeated concepts, definitions, processes, "
            "comparisons, diagrams, and concepts that are suitable for examination. "
            f"Selected question-mark types: {selected_prediction_marks}. "
            f"Generate exactly {prediction_questions_per_mark} questions for EACH "
            "selected mark type. Keep the requested mark types clearly separated. "
            "Rank every question as Very Important, Important, or Possible. "
            "Briefly state the topic/reason for the ranking. "
            "Do not invent facts that are not supported by the provided material."
        )
    }

    mode_instruction = study_instructions[study_mode]


    # ========================================================
    # CURRENT SUBJECT CONVERSATION CONTEXT
    # ========================================================

    current_subject = get_subject(
        st.session_state.current_subject_id
    )

    subject_history_text = ""


    if current_subject:

        # Use the latest 20 exchanges as context for follow-up
        # questions. The full subject history remains stored.
        recent_messages = (
            current_subject
            .get("messages", [])[-20:]
        )

        if recent_messages:

            history_parts = []

            for message in recent_messages:

                history_parts.append(
                    "Student: "
                    + message.get(
                        "question",
                        ""
                    )
                )

                history_parts.append(
                    "Assistant: "
                    + message.get(
                        "answer",
                        ""
                    )[:2500]
                )

            subject_history_text = (
                "\n\n".join(
                    history_parts
                )
            )


    long_term_memory = build_memory_context(
        st.session_state.current_subject_id
    )


    if subject_history_text:

        ai_question = (
            f"{question}\n\n"
            f"STUDY MODE INSTRUCTION:\n"
            f"{mode_instruction}\n\n"
            "LONG-TERM SUBJECT MEMORY:\n"
            "Use this memory to maintain continuity, but "
            "do not treat it as more reliable than the "
            "uploaded study material when answering PDF-based "
            "questions.\n"
            f"{long_term_memory}\n"
            "CURRENT SUBJECT CONVERSATION:\n"
            "Use the recent conversation to understand "
            "follow-up questions. Answer the CURRENT question.\n\n"
            f"{subject_history_text}"
        )

    else:

        ai_question = (
            f"{question}\n\n"
            f"STUDY MODE INSTRUCTION:\n"
            f"{mode_instruction}\n\n"
            "LONG-TERM SUBJECT MEMORY:\n"
            f"{long_term_memory}"
        )

    ask_button = st.button(
        "🔍 Ask Assistant",
        type="primary"
    )

    # ========================================================
    # ASK ASSISTANT
    # ========================================================

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ------------------------------------------------
            # VALIDATE PREDICTED QUESTION MARK SELECTION
            # ------------------------------------------------

            if (
                study_mode == "🎯 Predicted Questions"
                and not selected_prediction_marks
            ):

                st.warning(
                    "Please select at least one question-mark type "
                    "before generating predicted questions."
                )

                st.stop()


            # ------------------------------------------------
            # RESET OLD ANSWERS
            # ------------------------------------------------

            st.session_state.last_answer = ""

            st.session_state.last_online_answer = ""

            st.session_state.last_results = []

            st.session_state.last_question = question.strip()

            pdf_answer = ""

            online_answer = ""

            results = []

            # =================================================
            # PDF / RAG
            # =================================================

            if answer_mode in [
                "📚 PDF / RAG",
                "🔀 PDF + Online"
            ]:

                if not st.session_state.documents_loaded:

                    if answer_mode == "📚 PDF / RAG":

                        st.warning(
                            "📚 Please upload a PDF first."
                        )

                else:

                    retrieval_start = time.time()

                    with st.spinner(
                        "🔎 Searching your study material..."
                    ):

                        try:

                            semantic_results = (
                                st.session_state
                                .retriever
                                .search(
                                    question,
                                    top_k=8
                                )
                            )

                            results = _enhance_retrieval(
                                question,
                                semantic_results,
                                st.session_state.document_chunks,
                                max_results=6
                            )

                            print(
                                f"ENHANCED RETRIEVAL: "
                                f"{len(results)} results"
                            )

                            if results:
                                print(
                                    "TOP RETRIEVAL SCORES: "
                                    + str([
                                        round(
                                            item["score"],
                                            4
                                        )
                                        for item in results
                                    ])
                                )

                        except Exception as error:

                            st.error(
                                f"Retrieval error: {error}"
                            )

                            results = []

                    print(
                        f"RETRIEVAL TIME: "
                        f"{time.time() - retrieval_start:.2f} seconds"
                    )

                    if results:

                        st.session_state.last_results = (
                            results
                        )

                        # -------------------------------------
                        # BUILD CONTEXT
                        # -------------------------------------

                        context_parts = []

                        for result in results:

                            context_parts.append(
                                result["text"]
                            )

                        context = "\n\n".join(
                            context_parts
                        )

                        # -------------------------------------
                        # GEMINI
                        # -------------------------------------

                        generation_start = time.time()

                        with st.spinner(
                            "🤖 Generating PDF-based answer..."
                        ):

                            try:

                                pdf_prompt = (
                                    ai_question
                                    + "\n\n"
                                    "RAG GROUNDING RULE: If the retrieved study material "
                                    "contains the answer directly or contains the relevant "
                                    "definition/topic, answer from that material. Do not "
                                    "say the answer is unavailable merely because the "
                                    "question wording differs from the PDF wording."
                                )

                                if study_mode == "🎯 Predicted Questions":
                                    pdf_prompt = (
                                        "Using ONLY the provided study-material context, "
                                        "analyze the important examinable topics and generate "
                                        "likely exam questions. Do not claim certainty about "
                                        "the real exam.\n\n"
                                        "IMPORTANT QUESTION GENERATION RULES:\n"
                                        f"- Selected mark types: {', '.join(selected_prediction_marks)}\n"
                                        f"- Generate exactly {prediction_questions_per_mark} "
                                        "questions for EACH selected mark type.\n"
                                        "- Do not generate any unselected mark type.\n"
                                        "- Separate questions clearly by marks.\n"
                                        "- Rank each question as Very Important, Important, "
                                        "or Possible.\n"
                                        "- Mention the supporting topic/reason when useful.\n"
                                        "- Use ONLY information supported by the retrieved "
                                        "study material.\n\n"
                                        + ai_question
                                    )

                                pdf_answer = (
                                    generate_answer(
                                        pdf_prompt,
                                        context
                                    )
                                )

                                st.session_state.last_answer = (
                                    pdf_answer
                                )

                                print(
                                    f"GEMINI TIME: "
                                    f"{time.time() - generation_start:.2f} seconds"
                                )

                            except Exception as error:

                                st.error(
                                    f"Gemini error: {error}"
                                )

                    else:

                        if answer_mode == "📚 PDF / RAG":

                            st.warning(
                                "I could not find relevant "
                                "information in your study material."
                            )

            # =================================================
            # ONLINE AI
            # =================================================

            if answer_mode in [
                "🌐 Online AI",
                "🔀 PDF + Online"
            ]:

                if ask_online is None:

                    st.error(
                        "Online AI module is not available."
                    )

                else:

                    online_start = time.time()

                    with st.spinner(
                        "🌐 Asking Online Gemini..."
                    ):

                        try:

                            online_result = (
                                ask_online(
                                    ai_question
                                )
                            )

                            if online_result.get(
                                "success",
                                False
                            ):

                                online_answer = (
                                    online_result.get(
                                        "answer",
                                        ""
                                    )
                                )

                                st.session_state.last_online_answer = (
                                    online_answer
                                )

                                print(
                                    f"ONLINE GEMINI TIME: "
                                    f"{time.time() - online_start:.2f} seconds"
                                )

                            else:

                                error_message = (
                                    online_result.get(
                                        "answer",
                                        "Online AI could not generate an answer."
                                    )
                                )

                                st.warning(
                                    f"🌐 Online AI: "
                                    f"{error_message}"
                                )

                        except Exception as error:

                            st.error(
                                f"Online AI error: {error}"
                            )

            # =================================================
            # SAVE QUESTION
            # =================================================

            final_answer_for_history = ""

            if pdf_answer:

                final_answer_for_history += (
                    "📚 PDF / RAG ANSWER:\n\n"
                    + pdf_answer
                )

            if online_answer:

                if final_answer_for_history:

                    final_answer_for_history += (
                        "\n\n"
                        "🌐 ONLINE AI ANSWER:\n\n"
                        + online_answer
                    )

                else:

                    final_answer_for_history = (
                        "🌐 ONLINE AI ANSWER:\n\n"
                        + online_answer
                    )

            if final_answer_for_history:

                st.session_state.questions_asked += 1

                save_question(
                    CURRENT_USER,
                    question,
                    final_answer_for_history
                )

                update_progress(
                    CURRENT_USER,
                    st.session_state.questions_asked
                )


                # ------------------------------------------------
                # SAVE TO CURRENT SUBJECT CHAT
                # ------------------------------------------------

                add_subject_message(
                    st.session_state.current_subject_id,
                    question,
                    final_answer_for_history,
                    study_mode,
                    answer_mode
                )

                # ------------------------------------------------
                # UPDATE LONG-TERM SUBJECT MEMORY
                # ------------------------------------------------

                update_subject_memory(
                    st.session_state.current_subject_id,
                    question,
                    final_answer_for_history
                )


                # Refresh subject data.
                st.session_state.subjects = (
                    load_subjects()
                )


# ============================================================
# LEARNING DASHBOARD
# ============================================================

with col2:

    st.header(
        "📊 Learning Dashboard"
    )

    document_count = len(
        saved_documents
    )

    st.metric(
        "Documents",
        document_count
    )

    questions = get_user_questions(
        CURRENT_USER
    )

    st.metric(
        "Questions Asked",
        len(questions)
    )

    progress_data = get_progress(
        CURRENT_USER
    )

    if progress_data:

        progress = min(
            progress_data["questions_asked"] * 10,
            100
        )

    else:

        progress = 0

    st.metric(
        "Learning Progress",
        f"{progress}%"
    )


# ============================================================
# CURRENT SUBJECT CHAT
# ============================================================

current_subject = get_subject(
    st.session_state.current_subject_id
)

if current_subject:

    st.divider()

    st.header(
        f"💬 {current_subject['name']} — Chat"
    )

    subject_messages = (
        current_subject.get(
            "messages",
            []
        )
    )

    if subject_messages:

        st.caption(
            f"Showing the latest "
            f"{min(len(subject_messages), 20)} "
            f"conversation(s). "
            f"Total saved: {len(subject_messages)}"
        )

        # Show newest conversations first.
        st.markdown(
            """
            <style>
            .ta-past-question {
                background: linear-gradient(135deg,#eef2ff,#f8f9ff);
                border: 1px solid #c7d2fe;
                border-left: 6px solid #6366f1;
                border-radius: 15px;
                padding: 13px 16px;
                margin: 12px 0 7px 0;
                box-shadow: 0 3px 10px rgba(15,23,42,.05);
            }

            .ta-past-question-label {
                color: #4338ca;
                font-size: .75rem;
                font-weight: 850;
                letter-spacing: .05em;
                text-transform: uppercase;
                margin-bottom: 5px;
            }

            .ta-past-question-text {
                color: #1e293b;
                font-weight: 650;
                line-height: 1.55;
            }

            .ta-past-saved {
                border-radius: 15px;
                padding: 14px 16px;
                margin: 0 0 10px 24px;
                line-height: 1.65;
                box-shadow: 0 2px 8px rgba(15,23,42,.04);
            }

            .ta-past-saved-label {
                font-size: .75rem;
                font-weight: 850;
                letter-spacing: .05em;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            .ta-past-pdf {
                background: #eaf4ff;
                border: 1px solid #bfdbfe;
                border-left: 6px solid #1976d2;
                color: #17324d;
            }

            .ta-past-pdf .ta-past-saved-label {
                color: #125aa0;
            }

            .ta-past-online {
                background: #eaf8ee;
                border: 1px solid #bbf7d0;
                border-left: 6px solid #16a05d;
                color: #173b28;
            }

            .ta-past-online .ta-past-saved-label {
                color: #08783f;
            }

            .ta-past-normal {
                background: #f3edff;
                border: 1px solid #ddd6fe;
                border-left: 6px solid #7c3aed;
                color: #33204f;
            }

            .ta-past-normal .ta-past-saved-label {
                color: #6d28d9;
            }

            .ta-past-note {
                background: #fff8e1;
                border: 1px solid #fde68a;
                border-left: 5px solid #f59e0b;
                border-radius: 12px;
                padding: 10px 13px;
                margin: 5px 0 14px 24px;
                color: #78350f;
                font-size: .9rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        for message in reversed(subject_messages[-20:]):

            past_question = _inline_html(
                message.get("question", "")
            )

            raw_answer = str(
                message.get("answer", "")
            )

            st.markdown(
                f"""
                <div class="ta-past-question">
                    <div class="ta-past-question-label">
                        📝 Past Question
                    </div>
                    <div class="ta-past-question-text">
                        {past_question}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # Split saved comparison answers into their own
            # colored cards.
            # ------------------------------------------------

            pdf_marker = "📚 PDF / RAG ANSWER:"
            online_marker = "🌐 ONLINE AI ANSWER:"

            pdf_part = ""
            online_part = ""

            if pdf_marker in raw_answer:
                pdf_part = raw_answer.split(
                    pdf_marker,
                    1
                )[1]

                if online_marker in pdf_part:
                    pdf_part = pdf_part.split(
                        online_marker,
                        1
                    )[0]

            if online_marker in raw_answer:
                online_part = raw_answer.split(
                    online_marker,
                    1
                )[1]

            # If the saved answer is an old single-answer format,
            # show it as a normal saved answer.
            if not pdf_part and not online_part:

                saved_answer = _inline_html(
                    raw_answer
                ).replace("\n", "<br>")

                st.markdown(
                    f"""
                    <div class="ta-past-saved ta-past-normal">
                        <div class="ta-past-saved-label">
                            🤖 Saved Answer
                        </div>
                        <div>{saved_answer}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                if pdf_part.strip():

                    pdf_html = _inline_html(
                        pdf_part.strip()
                    ).replace("\n", "<br>")

                    st.markdown(
                        f"""
                        <div class="ta-past-saved ta-past-pdf">
                            <div class="ta-past-saved-label">
                                📚 PDF / RAG ANSWER
                            </div>
                            <div>{pdf_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if online_part.strip():

                    online_html = _inline_html(
                        online_part.strip()
                    ).replace("\n", "<br>")

                    st.markdown(
                        f"""
                        <div class="ta-past-saved ta-past-online">
                            <div class="ta-past-saved-label">
                                🌐 ONLINE AI ANSWER
                            </div>
                            <div>{online_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Old saved conversations cannot be regenerated
            # automatically. Tell the student when an old PDF
            # result was the previous "not available" response.
            if (
                "The answer is not available in the uploaded document."
                in raw_answer
            ):

                st.markdown(
                    """
                    <div class="ta-past-note">
                        🔄 This is an older saved PDF result.
                        Ask this question again to run the improved
                        PDF retrieval and check the uploaded material again.
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    else:

        st.info(
            "No questions in this subject yet. "
            "Ask your first question below."
        )


# ============================================================
# AI ANSWERS
# ============================================================

if (
    st.session_state.last_answer
    or st.session_state.last_online_answer
):

    st.divider()
    st.header("🤖 AI Assistant Answers")
    st.caption(f"🎯 Study Mode: **{study_mode}**")

    if answer_mode == "🔀 PDF + Online":

        st.info(
            "🔀 Comparison Mode: Both answers are generated independently "
            "so the student can compare the PDF-based answer with the Online AI answer."
        )

        compare_pdf, compare_online = st.columns(2)

        with compare_pdf:
            st.subheader("📚 Answer from Your PDF")
            st.caption("Grounded in the uploaded study material")
            if st.session_state.last_answer:
                render_beautiful_answer(
                    st.session_state.last_answer,
                    "PDF / RAG Answer"
                )
            else:
                st.warning("No PDF-based answer was generated.")

        with compare_online:
            st.subheader("🌐 Online AI Answer")
            st.caption("Generated independently using Online Gemini")
            if st.session_state.last_online_answer:
                render_beautiful_answer(
                    st.session_state.last_online_answer,
                    "Online Gemini Answer"
                )
            else:
                st.warning("No Online AI answer was generated.")

        st.divider()
        st.subheader("🔍 Student Comparison")

        comparison1, comparison2, comparison3 = st.columns(3)
        with comparison1:
            st.markdown("**📖 PDF Accuracy**")
            st.caption("Does the answer match your study material?")
        with comparison2:
            st.markdown("**🌐 Additional Knowledge**")
            st.caption("Does Online AI provide useful extra explanation?")
        with comparison3:
            st.markdown("**🎯 Exam Usefulness**")
            st.caption("Which explanation is easier to understand and revise?")

    else:

        if st.session_state.last_answer:
            st.subheader("📚 Answer from Your PDF")
            render_beautiful_answer(
                st.session_state.last_answer,
                "PDF / RAG Answer"
            )

        if st.session_state.last_online_answer:
            st.subheader("🌐 Online AI Answer")
            render_beautiful_answer(
                st.session_state.last_online_answer,
                "Online Gemini Answer"
            )


# ============================================================
# RETRIEVED CONTEXT
# ============================================================

if st.session_state.last_results:

    st.header(
        "📚 Retrieved Context"
    )

    for number, result in enumerate(
        st.session_state.last_results,
        start=1
    ):

        with st.expander(
            f"Context {number} "
            f"(Similarity: "
            f"{result['score']:.3f})"
        ):

            st.write(
                result["text"]
            )


# ============================================================
# RECENT QUESTIONS
# ============================================================

questions = get_user_questions(
    CURRENT_USER
)


if questions:

    st.divider()

    st.header(
        "📝 Recent Questions"
    )

    st.markdown("""
    <style>
        .ta-recent-question {
            background:linear-gradient(135deg,#fff7ed,#fffbf5);
            border:1px solid #fed7aa;
            border-left:6px solid #f97316;
            border-radius:14px;
            padding:12px 15px;
            margin:9px 0 4px 0;
        }
        .ta-recent-question-label {
            color:#c2410c;
            font-size:.74rem;
            font-weight:850;
            text-transform:uppercase;
            letter-spacing:.05em;
            margin-bottom:4px;
        }
        .ta-recent-question-text {
            color:#431407;
            font-weight:700;
        }
    </style>
    """, unsafe_allow_html=True)

    for item in questions[:5]:
        with st.expander(item["question"]):
            st.markdown(
                f"""
                <div class="ta-recent-question">
                    <div class="ta-recent-question-label">📝 Past Question</div>
                    <div class="ta-recent-question-text">
                        {_inline_html(item["question"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            saved_answer_html = _inline_html(item["answer"]).replace("\n", "<br>")
            st.markdown(
                f"""
                <div style="
                    background:#f8fafc;
                    border:1px solid #e2e8f0;
                    border-radius:14px;
                    padding:14px 16px;
                    margin-top:8px;
                    line-height:1.65;
                    color:#334155;
                ">
                    <b>🤖 Saved Answer</b><br><br>
                    {saved_answer_html}
                </div>
                """,
                unsafe_allow_html=True
            )



# ============================================================
# RAG PIPELINE
# ============================================================

st.divider()

st.header(
    "🔄 RAG Pipeline"
)

pipeline1, pipeline2, pipeline3, pipeline4, pipeline5 = (
    st.columns(5)
)


with pipeline1:

    st.write(
        "📄 **1. PDF**"
    )

    st.caption(
        "Upload study material"
    )


with pipeline2:

    st.write(
        "✂️ **2. Chunks**"
    )

    st.caption(
        "Split document text"
    )


with pipeline3:

    st.write(
        "🧠 **3. Embeddings**"
    )

    st.caption(
        "Convert text into vectors"
    )


with pipeline4:

    st.write(
        "🔎 **4. Retrieval**"
    )

    st.caption(
        "Find relevant content"
    )


with pipeline5:

    st.write(
        "🤖 **5. Gemini**"
    )

    st.caption(
        "Generate grounded answer"
    )


# ============================================================
# FUTURE FEATURES
# ============================================================

st.divider()

st.header(
    "🚀 Coming Modules"
)

feature1, feature2, feature3, feature4 = (
    st.columns(4)
)


with feature1:

    st.write(
        "🌐 **Online Q&A**"
    )

    st.caption(
        "Ask questions using online AI."
    )


with feature2:

    st.write(
        "📝 **Study Modes**"
    )

    st.caption(
        "Exam answers, revision notes, simple explanations, MCQs and predicted questions."
    )


with feature3:

    st.write(
        "🎯 **Question Prediction**"
    )

    st.caption(
        "Find likely and important exam questions."
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
# STARTUP PERFORMANCE
# ============================================================

startup_time = (
    time.time()
    - APP_START_TIME
)

print(
    f"APP READY: "
    f"{startup_time:.2f} seconds"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"AI Teaching Assistant Using RAG | "
    f"Startup: {startup_time:.1f}s | "
    f"Capstone Project | "
    f"2026–2027"
)