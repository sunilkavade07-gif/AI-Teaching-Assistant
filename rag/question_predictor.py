# ============================================================
# QUESTION PREDICTOR
# Historical question-paper analysis
# ============================================================

import re
import math
from collections import Counter
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# MARK PATTERNS
# ============================================================

MARK_PATTERNS = [
    (10, re.compile(r"\b10\s*(?:marks?|m)\b", re.I)),
    (6, re.compile(r"\b6\s*(?:marks?|m)\b", re.I)),
    (5, re.compile(r"\b5\s*(?:marks?|m)\b", re.I)),
    (4, re.compile(r"\b4\s*(?:marks?|m)\b", re.I)),
    (2, re.compile(r"\b2\s*(?:marks?|m)\b", re.I)),
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_question(text):
    """Normalize a question for comparison."""
    if not text:
        return ""

    text = str(text).strip()

    # Remove common question numbering.
    text = re.sub(
        r"^\s*(?:Q(?:uestion)?\.?\s*)?\d+\s*[\)\.\-:]?\s*",
        "",
        text,
        flags=re.I,
    )

    # Remove marks at the end.
    text = re.sub(
        r"\(?\s*\d+\s*(?:marks?|m)\s*\)?\s*$",
        "",
        text,
        flags=re.I,
    )

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_marks(text):
    """Detect a supported mark value from a question."""
    if not text:
        return None

    for marks, pattern in MARK_PATTERNS:
        if pattern.search(text):
            return marks

    return None


def split_question_lines(text):
    """
    Extract possible question lines from plain extracted text.

    This intentionally stays conservative. It does not claim that
    every line is a question.
    """
    if not text:
        return []

    lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in str(text).splitlines()
    ]

    candidates = []

    for line in lines:

        if not line:
            continue

        if len(line) < 15:
            continue

        # Skip obvious page/header/footer noise.
        lower = line.lower()

        if lower.startswith(("page ", "semester ", "subject code")):
            continue

        looks_like_question = (
            "?" in line
            or re.match(
                r"^(?:q(?:uestion)?\.?\s*)?\d+[\)\.\-:]\s*",
                line,
                flags=re.I,
            )
            or re.search(
                r"\b(?:explain|define|describe|discuss|write|state|"
                r"compare|differentiate|list|what|why|how|draw|"
                r"calculate|distinguish|mention)\b",
                lower,
            )
        )

        if looks_like_question:
            candidates.append(line)

    return candidates


# ============================================================
# QUESTION EXTRACTION
# ============================================================

def extract_questions(text, year=None):
    """
    Extract question records from one previous-paper text.

    Expected record:
        {
            question,
            normalized,
            marks,
            year
        }
    """
    questions = []

    for raw in split_question_lines(text):

        normalized = normalize_question(raw)

        if len(normalized) < 10:
            continue

        questions.append(
            {
                "question": raw,
                "normalized": normalized,
                "marks": detect_marks(raw),
                "year": year,
            }
        )

    return questions


# ============================================================
# SIMILARITY HELPERS
# ============================================================

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def recency_score(year, current_year=None):
    """
    Convert year into a recency score.

    Current year = 100.
    Each year older reduces the score.
    """
    if year is None:
        return 50.0

    if current_year is None:
        current_year = datetime.now().year

    age = max(0, current_year - int(year))

    return max(0.0, 100.0 - (age * 15.0))


# ============================================================
# QUESTION PREDICTOR
# ============================================================

class QuestionPredictor:

    def __init__(self, model=None):

        self.model = model

        if self.model is None:
            self.model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )


    # ========================================================
    # GROUP SIMILAR QUESTIONS
    # ========================================================

    def group_questions(
        self,
        questions,
        similarity_threshold=0.78,
    ):
        """
        Group semantically similar historical questions.
        """
        if not questions:
            return []

        texts = [
            item["normalized"]
            for item in questions
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        groups = []

        for index, question in enumerate(questions):

            best_group = None
            best_score = 0.0

            for group_index, group in enumerate(groups):

                representative = group["embedding"]

                score = cosine_similarity(
                    embeddings[index],
                    representative,
                )

                if score > best_score:
                    best_score = score
                    best_group = group_index

            if (
                best_group is not None
                and best_score >= similarity_threshold
            ):

                group = groups[best_group]

                group["questions"].append(question)
                group["indices"].append(index)

                # Keep the strongest representative.
                if best_score > group["representative_score"]:
                    group["embedding"] = embeddings[index]
                    group["representative_score"] = best_score

            else:

                groups.append(
                    {
                        "representative": question["normalized"],
                        "embedding": embeddings[index],
                        "representative_score": 1.0,
                        "questions": [question],
                        "indices": [index],
                    }
                )

        return groups


    # ========================================================
    # SCORE QUESTION GROUPS
    # ========================================================

    def score_groups(
        self,
        groups,
        current_year=None,
    ):
        """
        Calculate project prediction score.

        Frequency: 40%
        Similarity: 30%
        Recency: 20%
        Topic/coverage proxy: 10%

        This is a ranking score, NOT a guarantee of exam appearance.
        """
        if not groups:
            return []

        max_frequency = max(
            len(group["questions"])
            for group in groups
        )

        results = []

        for group in groups:

            questions = group["questions"]

            frequency = len(questions)

            frequency_score = (
                frequency / max_frequency
            ) * 100.0

            similarity_values = []

            for question in questions:
                similarity_values.append(
                    group["representative_score"]
                    * 100.0
                )

            semantic_score = (
                sum(similarity_values)
                / len(similarity_values)
                if similarity_values
                else 0.0
            )

            years = [
                q["year"]
                for q in questions
                if q.get("year") is not None
            ]

            if years:
                latest_year = max(years)
            else:
                latest_year = None

            recency = recency_score(
                latest_year,
                current_year,
            )

            # Topic/coverage proxy:
            # repeated semantic grouping itself indicates
            # that the concept recurs across papers.
            topic_coverage = min(
                100.0,
                50.0 + frequency * 10.0
            )

            prediction_score = (
                frequency_score * 0.40
                + semantic_score * 0.30
                + recency * 0.20
                + topic_coverage * 0.10
            )

            if prediction_score >= 80:
                importance = "🔥 VERY IMPORTANT"
            elif prediction_score >= 60:
                importance = "⭐ IMPORTANT"
            else:
                importance = "📌 POSSIBLE"

            mark_counts = Counter(
                q["marks"]
                for q in questions
                if q.get("marks") in {
                    2, 4, 5, 6, 10
                }
            )

            results.append(
                {
                    "question": group["representative"],
                    "marks": (
                        mark_counts.most_common(1)[0][0]
                        if mark_counts
                        else None
                    ),
                    "prediction_score": round(
                        prediction_score,
                        1,
                    ),
                    "importance": importance,
                    "frequency": frequency,
                    "latest_year": latest_year,
                    "frequency_score": round(
                        frequency_score,
                        1,
                    ),
                    "semantic_score": round(
                        semantic_score,
                        1,
                    ),
                    "recency_score": round(
                        recency,
                        1,
                    ),
                    "topic_coverage": round(
                        topic_coverage,
                        1,
                    ),
                    "source_questions": [
                        q["question"]
                        for q in questions
                    ],
                }
            )

        results.sort(
            key=lambda item: item["prediction_score"],
            reverse=True,
        )

        return results


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        paper_texts,
        selected_marks=None,
        questions_per_mark=3,
        current_year=None,
    ):
        """
        Generate ranked historical-question predictions.

        paper_texts:
            list of:
                {"text": "...", "year": 2025}
            or plain strings.

        selected_marks:
            e.g. [2, 4, 6]
        """
        all_questions = []

        for paper in paper_texts:

            if isinstance(paper, dict):

                text = paper.get(
                    "text",
                    ""
                )

                year = paper.get(
                    "year"
                )

            else:

                text = str(paper)
                year = None

            all_questions.extend(
                extract_questions(
                    text,
                    year=year,
                )
            )

        if selected_marks:

            selected_marks = {
                int(mark)
                for mark in selected_marks
            }

            filtered = [
                q
                for q in all_questions
                if q.get("marks") in selected_marks
            ]

        else:

            filtered = all_questions

        if not filtered:
            return []

        groups = self.group_questions(
            filtered
        )

        ranked = self.score_groups(
            groups,
            current_year=current_year,
        )

        # Respect requested quantity per mark type.
        if selected_marks:

            final_results = []

            for mark in sorted(
                selected_marks
            ):

                mark_results = [
                    item
                    for item in ranked
                    if item.get("marks") == mark
                ]

                final_results.extend(
                    mark_results[
                        :questions_per_mark
                    ]
                )

            return final_results

        return ranked


# ============================================================
# SIMPLE FUNCTION
# ============================================================

def predict_questions(
    paper_texts,
    selected_marks=None,
    questions_per_mark=3,
):
    predictor = QuestionPredictor()

    return predictor.predict(
        paper_texts=paper_texts,
        selected_marks=selected_marks,
        questions_per_mark=questions_per_mark,
    )