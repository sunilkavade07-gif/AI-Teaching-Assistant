from pypdf import PdfReader
from io import BytesIO


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from uploaded PDF.
    """

    print("=" * 50)
    print("PDF PROCESSING")
    print("=" * 50)

    try:
        # Read uploaded file completely
        pdf_bytes = uploaded_file.getvalue()

        # Create a fresh PDF stream
        pdf_stream = BytesIO(pdf_bytes)

        reader = PdfReader(pdf_stream)

        print(f"Total pages: {len(reader.pages)}")

        pages_text = []

        for page_number, page in enumerate(reader.pages, start=1):

            print(f"Processing page {page_number}...")

            try:
                text = page.extract_text()

                if text and text.strip():

                    text = text.strip()

                    pages_text.append(
                        f"\n--- Page {page_number} ---\n{text}"
                    )

                    print(
                        f"Page {page_number}: "
                        f"{len(text)} characters extracted."
                    )

                else:
                    print(
                        f"Page {page_number}: "
                        "NO TEXT DETECTED"
                    )

            except Exception as e:

                print(
                    f"Page {page_number} extraction error: {e}"
                )

        final_text = "\n".join(pages_text)

        print("=" * 50)
        print(
            f"TOTAL CHARACTERS EXTRACTED: "
            f"{len(final_text)}"
        )
        print("=" * 50)

        return final_text

    except Exception as e:

        print("PDF ERROR:", e)

        return ""


def get_pdf_page_count(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    pdf_stream = BytesIO(pdf_bytes)

    reader = PdfReader(pdf_stream)

    return len(reader.pages)