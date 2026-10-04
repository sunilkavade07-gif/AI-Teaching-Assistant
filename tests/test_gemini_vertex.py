from google import genai
from google.genai.types import HttpOptions


def main():
    print("Connecting to Gemini through Vertex AI...")

    client = genai.Client(
        vertexai=True,
        project="project-c2e13393-0412-4513-bca",
        location="us-central1",
        http_options=HttpOptions(api_version="v1"),
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello to my AI Teaching Assistant in one short sentence."
    )

    print("\nGemini Response:")
    print(response.text)


if __name__ == "__main__":
    main()