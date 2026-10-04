# ============================================================
# GEMINI RAG GENERATOR
# FAST + RELIABLE VERSION
# ============================================================

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GOOGLE CLOUD CONFIGURATION
# ============================================================

PROJECT_ID = os.getenv(
    "GOOGLE_CLOUD_PROJECT",
    "project-c2e13393-0412-4513-bca"
)

LOCATION = os.getenv(
    "GOOGLE_CLOUD_LOCATION",
    "us-central1"
)

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

# Optional fallback model.
# If the primary model is temporarily unavailable,
# we can try this model after retries.
FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-2.5-flash"
)

# Number of retries for temporary errors.
MAX_RETRIES = int(
    os.getenv(
        "GEMINI_MAX_RETRIES",
        "2"
    )
)


# ============================================================
# INITIALIZE GEMINI
# ============================================================

print("=" * 60)
print("INITIALIZING GEMINI RAG GENERATOR")
print("=" * 60)

gemini_start = time.time()

try:

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )

    print(
        "Gemini Vertex AI client initialized successfully."
    )

except Exception as e:

    print(
        "GEMINI INITIALIZATION ERROR:"
    )

    print(e)

    client = None


print(
    f"GEMINI INITIALIZATION TIME: "
    f"{time.time() - gemini_start:.2f} seconds"
)

print(
    f"PRIMARY MODEL: {MODEL_NAME}"
)

print(
    f"FALLBACK MODEL: {FALLBACK_MODEL}"
)

print("=" * 60)


# ============================================================
# ERROR HELPERS
# ============================================================

def is_temporary_error(error_text):
    """
    Detect errors that are normally worth retrying.
    """

    text = error_text.upper()

    temporary_errors = [
        "429",
        "RESOURCE_EXHAUSTED",
        "500",
        "502",
        "503",
        "504",
        "INTERNAL",
        "UNAVAILABLE",
        "SERVICE UNAVAILABLE",
        "HIGH DEMAND",
        "TIMEOUT",
        "TIMED OUT",
        "DEADLINE EXCEEDED",
    ]

    return any(
        error in text
        for error in temporary_errors
    )


def is_configuration_error(error_text):
    """
    Detect errors where retrying will not help.
    """

    text = error_text.upper()

    configuration_errors = [
        "401",
        "403",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "INVALID_ARGUMENT",
        "NOT_FOUND",
    ]

    return any(
        error in text
        for error in configuration_errors
    )


# ============================================================
# ONE GEMINI REQUEST
# ============================================================

def _generate_with_model(
    prompt,
    model_name
):
    """
    Send one request to Gemini.

    Thinking remains disabled for speed.
    """

    start_time = time.time()

    response = client.models.generate_content(

        model=model_name,

        contents=prompt,

        config=types.GenerateContentConfig(

            # ------------------------------------------------
            # SPEED
            # ------------------------------------------------

            thinking_config=(
                types.ThinkingConfig(
                    thinking_budget=0
                )
            ),

            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------

            max_output_tokens=2048,

            # ------------------------------------------------
            # STUDY ASSISTANT STYLE
            # ------------------------------------------------

            temperature=0.2
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"GEMINI RESPONSE TIME: "
        f"{elapsed:.2f} seconds"
    )

    if response is None:

        raise RuntimeError(
            "Gemini returned no response."
        )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


# ============================================================
# SIMPLE GEMINI FUNCTION
# ============================================================

def generate_answer_simple(prompt):
    """
    Fast and reliable Gemini text generation.

    Temporary errors:
        Retry with exponential backoff.

    Permanent/configuration errors:
        Stop immediately.

    If the primary model still fails:
        Try the fallback model.
    """

    if client is None:

        return (
            "Gemini client is not initialized."
        )


    models_to_try = [
        MODEL_NAME
    ]

    # Do not duplicate the same model.
    if (
        FALLBACK_MODEL
        and FALLBACK_MODEL
        not in models_to_try
    ):

        models_to_try.append(
            FALLBACK_MODEL
        )


    last_error = None


    # ========================================================
    # PRIMARY + FALLBACK MODELS
    # ========================================================

    for model_index, model_name in enumerate(
        models_to_try
    ):

        print()
        print("=" * 60)
        print(
            f"TRYING GEMINI MODEL: "
            f"{model_name}"
        )
        print("=" * 60)


        # ----------------------------------------------------
        # RETRIES
        # ----------------------------------------------------

        for attempt in range(
            MAX_RETRIES + 1
        ):

            try:

                if attempt > 0:

                    # 1 second, then 2 seconds.
                    wait_time = min(
                        2 ** (attempt - 1),
                        4
                    )

                    print(
                        f"Waiting "
                        f"{wait_time} seconds "
                        f"before retry..."
                    )

                    time.sleep(
                        wait_time
                    )


                print(
                    f"Gemini attempt "
                    f"{attempt + 1}/"
                    f"{MAX_RETRIES + 1}"
                )


                answer = _generate_with_model(
                    prompt,
                    model_name
                )


                print(
                    "GEMINI ANSWER RECEIVED"
                )

                print("=" * 60)


                return answer


            except Exception as e:

                last_error = e

                error_text = str(e)

                print()
                print("=" * 60)
                print("GEMINI ERROR")
                print("=" * 60)

                print(
                    f"MODEL: {model_name}"
                )

                print(
                    f"ATTEMPT: "
                    f"{attempt + 1}"
                )

                print(
                    error_text
                )

                print("=" * 60)


                # --------------------------------------------
                # CONFIGURATION ERROR
                # --------------------------------------------

                if is_configuration_error(
                    error_text
                ):

                    print(
                        "Configuration/"
                        "authentication error."
                    )

                    return (
                        "Gemini configuration error. "
                        "Please check the Google Cloud "
                        "project, location, model, and "
                        "permissions."
                    )


                # --------------------------------------------
                # TEMPORARY ERROR
                # --------------------------------------------

                if is_temporary_error(
                    error_text
                ):

                    print(
                        "Temporary Gemini error. "
                        "Retrying..."
                    )

                    continue


                # --------------------------------------------
                # UNKNOWN ERROR
                # --------------------------------------------

                print(
                    "Non-temporary Gemini error."
                )

                break


        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if model_index < len(
            models_to_try
        ) - 1:

            print()
            print("=" * 60)

            print(
                "PRIMARY GEMINI MODEL FAILED."
            )

            print(
                f"Trying fallback model: "
                f"{models_to_try[model_index + 1]}"
            )

            print("=" * 60)


    # ========================================================
    # ALL ATTEMPTS FAILED
    # ========================================================

    if last_error is not None:

        error_text = str(
            last_error
        )

        if is_temporary_error(
            error_text
        ):

            return (
                "Gemini is temporarily busy. "
                "The request was retried several times "
                "but Gemini did not become available. "
                "Please try again in a few seconds."
            )


    return (
        "Gemini could not generate an answer right now. "
        "Please try again."
    )


# ============================================================
# RAG ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    context
):

    print()
    print("=" * 60)
    print("GENERATING RAG ANSWER")
    print("=" * 60)

    try:

        # ====================================================
        # CONVERT CONTEXT TO TEXT
        # ====================================================

        context_text = ""

        if isinstance(
            context,
            list
        ):

            for i, item in enumerate(
                context,
                start=1
            ):

                if isinstance(
                    item,
                    dict
                ):

                    text = item.get(
                        "text",
                        ""
                    )

                    score = item.get(
                        "score",
                        0
                    )

                    if text:

                        context_text += (
                            f"\n--- Chunk {i} "
                            f"(score: {score:.4f}) ---\n"
                        )

                        context_text += (
                            text
                        )

                        context_text += (
                            "\n"
                        )

                else:

                    context_text += (
                        f"\n--- Chunk {i} ---\n"
                    )

                    context_text += (
                        str(item)
                    )

                    context_text += (
                        "\n"
                    )

        elif isinstance(
            context,
            str
        ):

            context_text = context

        else:

            context_text = str(
                context
            )


        # ====================================================
        # CONTEXT LIMIT
        # ====================================================

        MAX_CONTEXT_CHARS = 14000

        if len(
            context_text
        ) > MAX_CONTEXT_CHARS:

            context_text = (
                context_text[
                    :MAX_CONTEXT_CHARS
                ]
            )

            context_text += (
                "\n\n"
                "[Context shortened for speed.]"
            )


        print(
            f"CONTEXT CHARACTERS: "
            f"{len(context_text)}"
        )


        # ====================================================
        # RAG PROMPT
        # ====================================================

        prompt = f"""
You are an AI Teaching Assistant.

Answer the student's question using ONLY the
retrieved study material below.

RETRIEVED STUDY MATERIAL
========================
{context_text}
========================

STUDENT QUESTION
================
{question}
================

RULES:

1. Use only the retrieved study material.
2. Do not invent information.
3. Do not use outside knowledge.
4. Explain in simple student-friendly language.
5. Use headings or bullet points when useful.
6. Give examples only when supported by the material.
7. For technical topics, explain step-by-step.
8. Keep the answer concise but useful.
9. If the answer is genuinely not present in the
   retrieved material, say exactly:

"The answer is not available in the uploaded document."

ANSWER:
"""


        # ====================================================
        # GEMINI
        # ====================================================

        result = generate_answer_simple(
            prompt
        )


        print(
            "RAG ANSWER GENERATED."
        )

        print("=" * 60)


        return result.strip()


    except Exception as e:

        print()
        print("=" * 60)
        print("RAG GENERATION ERROR")
        print("=" * 60)

        print(
            e
        )

        print("=" * 60)

        return (
            "Gemini could not generate the "
            "PDF-based answer right now. "
            "Please try again."
        )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("GEMINI RAG GENERATOR TEST")
    print("=" * 60)

    test_result = generate_answer(
        "What is an operating system?",
        "An operating system is system software that manages "
        "computer hardware and provides services to programs."
    )

    print()
    print("=" * 60)
    print("TEST RESULT")
    print("=" * 60)

    print(
        test_result
    )