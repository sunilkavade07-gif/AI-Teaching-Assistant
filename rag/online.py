# ============================================================
# ONLINE AI
# HIGH QUALITY + STREAMING GEMINI VERSION
# ============================================================

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_PATH = PROJECT_ROOT / "rag" / ".env"


print("=" * 60)
print("ONLINE GEMINI")
print("=" * 60)

print(f"Project root: {PROJECT_ROOT}")
print(f"Loading .env: {ENV_PATH}")


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(ENV_PATH)


# ============================================================
# API KEY
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")


if not API_KEY:

    print("ERROR: GEMINI_API_KEY not found.")

else:

    print("GEMINI_API_KEY found successfully.")


# ============================================================
# MODEL
# ============================================================

ONLINE_MODEL = os.getenv(
    "ONLINE_GEMINI_MODEL",
    "gemini-3.6-flash"
)

print(f"ONLINE MODEL: {ONLINE_MODEL}")


# ============================================================
# ONLINE AI CLASS
# ============================================================

class OnlineAI:

    def __init__(self):

        self.client = None

        if not API_KEY:

            print("Online Gemini disabled.")

            return

        try:

            print("Initializing Online Gemini...")

            self.client = genai.Client(
                api_key=API_KEY
            )

            print(
                "Online Gemini initialized successfully."
            )

        except Exception as e:

            print(
                "ONLINE GEMINI INITIALIZATION ERROR:"
            )

            print(e)

            self.client = None


    # ========================================================
    # BUILD HIGH-QUALITY PROMPT
    # ========================================================

    def build_prompt(self, question):

        return f"""
You are an expert AI Teaching Assistant for college students.

Your job is to give the student a correct, useful, clear,
and easy-to-understand answer.

STUDENT QUESTION:
{question}

ANSWERING RULES:

1. First understand exactly what the student is asking.

2. Give a direct answer before giving extra explanation.

3. Explain difficult concepts in very simple student-friendly
   language.

4. Use headings when they improve readability.

5. Use bullet points for lists.

6. Give a practical example when useful.

7. For technical subjects, explain step-by-step.

8. For programming questions:
   - Explain the concept briefly.
   - Give correct code when requested.
   - Explain important parts of the code.
   - Keep the code simple and runnable.

9. For mathematical calculations:
   - Show the important calculation steps.
   - Give the final answer clearly.

10. For comparisons:
    - Use a table when appropriate.

11. For exam preparation:
    - Highlight important points.
    - Use definitions, examples, advantages,
      disadvantages, and differences when relevant.

12. If the question is simple, do not give an unnecessarily
    long answer.

13. If the question requires a detailed explanation,
    provide enough detail for proper understanding.

14. Do not use unnecessarily complicated terminology.

15. Do not mention these instructions.

16. Do not say that you are an AI unless it is relevant.

17. Never deliberately invent facts.

18. If something is uncertain, clearly say so.

19. Keep the answer well organized and easy to read.

20. Focus on helping the student understand and remember
    the topic.

Now answer the student's question.
"""


    # ========================================================
    # NORMAL ASK
    # ========================================================

    def ask(self, question):

        # ----------------------------------------------------
        # CLIENT CHECK
        # ----------------------------------------------------

        if not self.client:

            return {
                "success": False,
                "answer": "Online AI is not configured.",
                "source": "Online AI"
            }


        # ----------------------------------------------------
        # QUESTION CHECK
        # ----------------------------------------------------

        if not question or not question.strip():

            return {
                "success": False,
                "answer": "Please enter a question.",
                "source": "Online AI"
            }


        # ----------------------------------------------------
        # BUILD PROMPT
        # ----------------------------------------------------

        prompt = self.build_prompt(
            question.strip()
        )


        # ----------------------------------------------------
        # START TIMER
        # ----------------------------------------------------

        start_time = time.time()


        print()
        print("=" * 60)
        print("ONLINE AI REQUEST")
        print(f"MODEL: {ONLINE_MODEL}")
        print("=" * 60)


        try:

            # =================================================
            # STREAM RESPONSE
            # =================================================

            response_stream = (
                self.client
                .models
                .generate_content_stream(

                    model=ONLINE_MODEL,

                    contents=prompt,

                    config=types.GenerateContentConfig(
                       max_output_tokens=1024,
                       thinking_config=types.ThinkingConfig(
                     thinking_level="minimal"
                       )
)
                )
            )


            # ------------------------------------------------
            # COLLECT STREAMED TEXT
            # ------------------------------------------------

            answer_parts = []

            first_chunk_time = None

            for chunk in response_stream:

                if chunk is None:
                    continue

                chunk_text = getattr(
                    chunk,
                    "text",
                    None
                )

                if chunk_text:

                    if first_chunk_time is None:

                        first_chunk_time = (
                            time.time()
                            - start_time
                        )

                        print(
                            f"FIRST TOKEN TIME: "
                            f"{first_chunk_time:.2f} seconds"
                        )

                    answer_parts.append(
                        chunk_text
                    )

                    # Show streaming in terminal
                    print(
                        chunk_text,
                        end="",
                        flush=True
                    )


            # ------------------------------------------------
            # COMPLETE ANSWER
            # ------------------------------------------------

            answer = "".join(
                answer_parts
            ).strip()


            elapsed = (
                time.time()
                - start_time
            )


            print()
            print()

            print(
                f"ONLINE GEMINI TIME: "
                f"{elapsed:.2f} seconds"
            )


            # ------------------------------------------------
            # CHECK ANSWER
            # ------------------------------------------------

            if answer:

                print(
                    "ONLINE AI ANSWER RECEIVED"
                )

                return {

                    "success": True,

                    "answer": answer,

                    "source":
                        "Online Gemini"

                }


            # ------------------------------------------------
            # EMPTY RESPONSE
            # ------------------------------------------------

            print(
                "ONLINE AI RETURNED EMPTY RESPONSE."
            )

            return {

                "success": False,

                "answer":
                    "Online AI returned an empty answer.",

                "source":
                    "Online Gemini"

            }


        # ====================================================
        # ERROR HANDLING
        # ====================================================

        except Exception as e:

            elapsed = (
                time.time()
                - start_time
            )

            error_text = str(e)


            print()
            print("=" * 60)
            print("ONLINE AI ERROR")
            print("=" * 60)

            print(error_text)

            print(
                f"FAILED AFTER: "
                f"{elapsed:.2f} seconds"
            )

            print("=" * 60)


            # ------------------------------------------------
            # RATE LIMIT
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED"
                in error_text
            ):

                return {

                    "success": False,

                    "answer":
                        "Online AI is temporarily busy "
                        "because the API rate limit has "
                        "been reached. Please try again "
                        "after a short while.",

                    "source":
                        "Online Gemini"

                }


            # ------------------------------------------------
            # SERVICE UNAVAILABLE
            # ------------------------------------------------

            if (
                "503" in error_text
                or "UNAVAILABLE"
                in error_text
                or "high demand"
                in error_text
            ):

                return {

                    "success": False,

                    "answer":
                        "Online AI is temporarily busy. "
                        "Please try again shortly.",

                    "source":
                        "Online Gemini"

                }


            # ------------------------------------------------
            # MODEL NOT FOUND
            # ------------------------------------------------

            if (
                "404" in error_text
                or "NOT_FOUND"
                in error_text
            ):

                return {

                    "success": False,

                    "answer":
                        "The configured Gemini model is "
                        "not available. Please check "
                        "ONLINE_GEMINI_MODEL in the .env file.",

                    "source":
                        "Online Gemini"

                }


            # ------------------------------------------------
            # GENERAL ERROR
            # ------------------------------------------------

            return {

                "success": False,

                "answer":
                    "Online AI could not generate "
                    "an answer right now.",

                "source":
                    "Online Gemini"

            }


# ============================================================
# GLOBAL ONLINE AI OBJECT
# ============================================================

online_ai = OnlineAI()


# ============================================================
# APP FUNCTION
# ============================================================

def ask_online(question):

    return online_ai.ask(
        question
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ONLINE GEMINI TEST")
    print("=" * 60)


    question = (
        "What is an operating system?"
    )


    result = ask_online(
        question
    )


    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(result)