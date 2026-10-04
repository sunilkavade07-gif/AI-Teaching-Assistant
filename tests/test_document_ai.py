from pathlib import Path

from google.cloud import documentai_v1 as documentai


PROJECT_ID = "ai-teaching-assistant-505717"
LOCATION = "us"
PROCESSOR_ID = "413207ab9191b063"

PDF_PATH = Path("data/test.pdf")


def main():

    print("=" * 60)
    print("DOCUMENT AI REAL PDF TEST")
    print("=" * 60)

    if not PDF_PATH.exists():

        print(
            f"ERROR: PDF not found: {PDF_PATH}"
        )

        return

    pdf_bytes = PDF_PATH.read_bytes()

    print(
        f"PDF size: "
        f"{len(pdf_bytes) / (1024 * 1024):.2f} MB"
    )

    client = documentai.DocumentProcessorServiceClient()

    processor_name = client.processor_path(
        PROJECT_ID,
        LOCATION,
        PROCESSOR_ID
    )

    print(
        f"Processor: {processor_name}"
    )

    raw_document = documentai.RawDocument(
        content=pdf_bytes,
        mime_type="application/pdf"
    )

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document
    )

    print("Sending PDF to Document AI...")
    print("Please wait...")

    result = client.process_document(
        request=request
    )

    document = result.document

    text = document.text or ""

    print("=" * 60)
    print("DOCUMENT AI TEST RESULT")
    print("=" * 60)

    print(
        f"Pages detected: "
        f"{len(document.pages)}"
    )

    print(
        f"Characters extracted: "
        f"{len(text)}"
    )

    if text.strip():

        print("\nFIRST 2000 CHARACTERS:")
        print("-" * 60)
        print(text[:2000])
        print("-" * 60)

        print(
            "\n✅ DOCUMENT AI EXTRACTION SUCCESSFUL"
        )

    else:

        print(
            "\n❌ DOCUMENT AI RETURNED NO TEXT"
        )


if __name__ == "__main__":
    main()