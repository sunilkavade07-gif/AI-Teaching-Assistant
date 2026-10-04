import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# ONLINE GEMINI TEST
# AI TEACHING ASSISTANT
# ============================================================

print("=" * 60)
print("ONLINE GEMINI TEST")
print("=" * 60)


# ============================================================
# FIND PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Your .env file is inside the rag folder
ENV_FILE = PROJECT_ROOT / "rag" / ".env"

print(f"Project root: {PROJECT_ROOT}")
print(f"Loading .env: {ENV_FILE}")


# ============================================================
# CHECK .ENV
# ============================================================

if not ENV_FILE.exists():

    print()
    print("ERROR: .env file not found!")
    print(f"Expected location: {ENV_FILE}")

    exit()


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(ENV_FILE)


# ============================================================
# GET GEMINI API KEY
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")


if not API_KEY:

    print()
    print("ERROR: GEMINI_API_KEY not found in rag/.env")
    print()
    print("Your rag/.env should contain:")
    print()
    print("GEMINI_API_KEY=YOUR_API_KEY")
    print()

    exit()


print("GEMINI_API_KEY found successfully.")


# ============================================================
# INITIALIZE GEMINI
# ============================================================

try:

    print()
    print("Initializing Gemini...")

    client = genai.Client(
        api_key=API_KEY
    )

    print("Gemini client initialized successfully.")


    # ========================================================
    # ONLINE QUESTION
    # ========================================================

    print()
    print("Sending online question...")


    question = """
Explain Operating System in very simple language
for a college student.

Include:

1. Definition
2. Main functions
3. Simple real-life example
4. Short conclusion
"""


    # ========================================================
    # GENERATE ONLINE ANSWER
    # ========================================================

    response = client.models.generate_content(

        model="gemini-3.6-flash",

        contents=question

    )


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    print()
    print("=" * 60)
    print("ONLINE GEMINI ANSWER")
    print("=" * 60)

    if response and response.text:

        print(response.text)

    else:

        print("Gemini returned an empty answer.")


    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("=" * 60)
    print("ONLINE GEMINI TEST SUCCESSFUL")
    print("=" * 60)


# ============================================================
# ERROR HANDLING
# ============================================================

except Exception as e:

    print()
    print("=" * 60)
    print("ONLINE GEMINI ERROR")
    print("=" * 60)

    print("Error type:")
    print(type(e).__name__)

    print()
    print("Error message:")
    print(e)

    print()
    print("=" * 60)