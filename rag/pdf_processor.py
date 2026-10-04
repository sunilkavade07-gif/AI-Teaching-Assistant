import time
from io import BytesIO

from pypdf import PdfReader, PdfWriter

from google.api_core.client_options import ClientOptions

from google.cloud import documentai_v1 as documentai
from rag.cloud_auth import get_google_credentials

# ============================================================
# GOOGLE CLOUD CONFIGURATION
# ============================================================

PROJECT_ID = "project-c2e13393-0412-4513-bca"

LOCATION = "us"

PROCESSOR_ID = "9c5d0c841f90d81"


# ============================================================
# CREATE DOCUMENT AI CLIENT
# ============================================================

def create_document_ai_client():

    client_options = ClientOptions(
        api_endpoint=f"{LOCATION}-documentai.googleapis.com"
    )

    google_credentials = get_google_credentials()

    client = (
        documentai.DocumentProcessorServiceClient(
            client_options=client_options,
            credentials=google_credentials
        )
    )

    return client


# ============================================================
# DOCUMENT AI OCR
# ============================================================

def extract_text_with_document_ai(uploaded_file):
    """
    Extract text from scanned/image-based PDFs
    using Google Cloud Document AI OCR.

    Large PDFs are divided into smaller PDF batches
    before sending them to Document AI.
    """

    print("=" * 60)
    print("DOCUMENT AI PDF PROCESSING")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # Reset file pointer
        # ----------------------------------------------------

        uploaded_file.seek(0)

        pdf_bytes = uploaded_file.read()

        if not pdf_bytes:

            print("ERROR: Empty PDF file.")

            return ""

        print(
            f"Original PDF size: "
            f"{len(pdf_bytes) / (1024 * 1024):.2f} MB"
        )

        # ----------------------------------------------------
        # Read original PDF
        # ----------------------------------------------------

        reader = PdfReader(
            BytesIO(pdf_bytes)
        )

        total_pages = len(
            reader.pages
        )

        print(
            f"Total pages: {total_pages}"
        )

        # ----------------------------------------------------
        # Create Document AI client
        # ----------------------------------------------------

        client = create_document_ai_client()

        processor_name = (
            client.processor_path(
                PROJECT_ID,
                LOCATION,
                PROCESSOR_ID
            )
        )

        print(
            f"Processor: {processor_name}"
        )

        # ----------------------------------------------------
        # Store extracted text
        # ----------------------------------------------------

        all_text = []

        # ----------------------------------------------------
        # Maximum 5 pages per request
        #
        # Document AI online OCR limit is 15 pages.
        # We use 5 for extra safety.
        # ----------------------------------------------------

        pages_per_batch = 15

        # ----------------------------------------------------
        # Process PDF batches
        # ----------------------------------------------------

        for start in range(
            0,
            total_pages,
            pages_per_batch
        ):

            end = min(
                start + pages_per_batch,
                total_pages
            )

            print()
            print("=" * 60)

            print(
                f"Processing pages "
                f"{start + 1}-{end}"
            )

            print("=" * 60)

            # ------------------------------------------------
            # Create small PDF
            # ------------------------------------------------

            writer = PdfWriter()

            for page_number in range(
                start,
                end
            ):

                writer.add_page(
                    reader.pages[page_number]
                )

            batch_stream = BytesIO()

            writer.write(
                batch_stream
            )

            batch_bytes = (
                batch_stream.getvalue()
            )

            batch_size_mb = (
                len(batch_bytes)
                / (1024 * 1024)
            )

            print(
                f"Batch size: "
                f"{batch_size_mb:.2f} MB"
            )

            # ------------------------------------------------
            # Check Document AI 40 MB online limit
            # ------------------------------------------------

            if len(batch_bytes) >= 40 * 1024 * 1024:

                print(
                    "WARNING: 15-page batch is too large."
                )

                print(
                    "Falling back to 5-page batches "
                    "for this section."
                )

                fallback_size = 5

                for fallback_start in range(
                    start,
                    end,
                    fallback_size
                ):

                    fallback_end = min(
                        fallback_start + fallback_size,
                        end
                    )

                    fallback_writer = PdfWriter()

                    for page_number in range(
                        fallback_start,
                        fallback_end
                    ):

                        fallback_writer.add_page(
                            reader.pages[page_number]
                        )

                    fallback_stream = BytesIO()

                    fallback_writer.write(
                        fallback_stream
                    )

                    fallback_bytes = (
                        fallback_stream.getvalue()
                    )

                    if (
                        len(fallback_bytes)
                        >= 40 * 1024 * 1024
                    ):

                        print(
                            f"ERROR: Pages "
                            f"{fallback_start + 1}-"
                            f"{fallback_end} are still too large."
                        )

                        continue

                    fallback_document = (
                        documentai.RawDocument(
                            content=fallback_bytes,
                            mime_type="application/pdf"
                        )
                    )

                    fallback_request = (
                        documentai.ProcessRequest(
                            name=processor_name,
                            raw_document=fallback_document,
                            process_options=process_options
                        )
                    )

                    print(
                        f"Sending fallback pages "
                        f"{fallback_start + 1}-"
                        f"{fallback_end}..."
                    )

                    fallback_start_time = time.time()

                    fallback_result = (
                        client.process_document(
                            request=fallback_request
                        )
                    )

                    print(
                        f"Fallback batch time: "
                        f"{time.time() - fallback_start_time:.2f} seconds"
                    )

                    fallback_text = (
                        fallback_result.document.text
                        or ""
                    )

                    if fallback_text.strip():

                        all_text.append(
                            "\n--- Pages "
                            f"{fallback_start + 1}-"
                            f"{fallback_end} ---\n"
                            f"{fallback_text.strip()}"
                        )

                        print(
                            f"Characters extracted: "
                            f"{len(fallback_text)}"
                        )

                continue

            # ------------------------------------------------
            # Create raw document
            # ------------------------------------------------

            raw_document = (
                documentai.RawDocument(
                    content=batch_bytes,
                    mime_type="application/pdf"
                )
            )

            # ------------------------------------------------
            # OCR configuration
            # ------------------------------------------------

            process_options = (
                documentai.ProcessOptions(
                    ocr_config=
                    documentai.OcrConfig(
                        enable_native_pdf_parsing=True
                    )
                )
            )

            # ------------------------------------------------
            # Create request
            # ------------------------------------------------

            request = (
                documentai.ProcessRequest(
                    name=processor_name,
                    raw_document=raw_document,
                    process_options=process_options
                )
            )

            print(
                "Sending batch to "
                "Google Document AI..."
            )

            # ------------------------------------------------
            # Send to Document AI
            # ------------------------------------------------

            batch_start_time = time.time()

            result = (
                client.process_document(
                    request=request
                )
            )

            print(
                f"Document AI batch time: "
                f"{time.time() - batch_start_time:.2f} seconds"
            )

            # ------------------------------------------------
            # Get OCR text
            # ------------------------------------------------

            batch_text = (
                result.document.text
                or ""
            )

            if batch_text.strip():

                all_text.append(
                    "\n--- Pages "
                    f"{start + 1}-{end} ---\n"
                    f"{batch_text.strip()}"
                )

                print(
                    f"Characters extracted: "
                    f"{len(batch_text)}"
                )

                print(
                    "Batch extraction successful."
                )

            else:

                print(
                    "NO TEXT DETECTED "
                    "in this batch."
                )

        # ----------------------------------------------------
        # Combine all batches
        # ----------------------------------------------------

        final_text = "\n".join(
            all_text
        )

        print()
        print("=" * 60)

        print(
            "TOTAL DOCUMENT AI CHARACTERS:",
            len(final_text)
        )

        print("=" * 60)

        if final_text.strip():

            print(
                "TEXT EXTRACTION SUCCESSFUL"
            )

            print(
                "=" * 60
            )

            return final_text.strip()

        print(
            "Document AI returned no text."
        )

        print(
            "=" * 60
        )

        return ""

    except Exception as e:

        print("=" * 60)

        print(
            "DOCUMENT AI ERROR"
        )

        print(
            str(e)
        )

        print("=" * 60)

        return ""


# ============================================================
# PYPDF TEXT EXTRACTION
# ============================================================

def extract_text_with_pypdf(uploaded_file):
    """
    Try to extract selectable text directly from PDF.

    This is NOT AI.

    It is used first because normal text PDFs are faster
    and cheaper to process this way.
    """

    print("=" * 60)
    print("PYPDF PDF PROCESSING")
    print("=" * 60)

    try:

        # IMPORTANT:
        # Reset file pointer to beginning.

        uploaded_file.seek(0)

        pdf_bytes = uploaded_file.read()

        if not pdf_bytes:

            print(
                "ERROR: Empty PDF file."
            )

            return ""

        pdf_stream = BytesIO(
            pdf_bytes
        )

        reader = PdfReader(
            pdf_stream
        )

        total_pages = len(
            reader.pages
        )

        print(
            f"Total pages: {total_pages}"
        )

        pages_text = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            print(
                f"Processing page "
                f"{page_number}..."
            )

            try:

                text = page.extract_text()

                if text and text.strip():

                    text = text.strip()

                    pages_text.append(
                        f"\n--- Page "
                        f"{page_number} ---\n"
                        f"{text}"
                    )

                    print(
                        f"Page {page_number}: "
                        f"{len(text)} characters "
                        "extracted."
                    )

                else:

                    print(
                        f"Page {page_number}: "
                        "NO TEXT DETECTED"
                    )

            except Exception as e:

                print(
                    f"Page {page_number} "
                    f"extraction error: {e}"
                )

        final_text = "\n".join(
            pages_text
        )

        print()
        print(
            "Total characters extracted:",
            len(final_text)
        )

        print("=" * 60)

        return final_text.strip()

    except Exception as e:

        print(
            "PYPDF ERROR:",
            e
        )

        return ""


# ============================================================
# MAIN PDF EXTRACTION
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """
    Main PDF extraction pipeline.

    STEP 1:
        Try normal selectable PDF text using PyPDF.

    STEP 2:
        If little/no text exists,
        use Google Document AI OCR.

    STEP 3:
        Return extracted text.
    """

    print()
    print("#" * 60)

    print(
        "STARTING PDF TEXT EXTRACTION"
    )

    print("#" * 60)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print()

    print(
        "STEP 1: Trying normal PDF text extraction..."
    )

    text = extract_text_with_pypdf(
        uploaded_file
    )

    # --------------------------------------------------------
    # If enough text was found
    # --------------------------------------------------------

    if (
        text
        and
        len(text.strip()) >= 50
    ):

        print()

        print(
            "Normal PDF text extraction successful."
        )

        print(
            "Document AI is not required."
        )

        print(
            "#" * 60
        )

        return text

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print()

    print(
        "Little or no readable text found."
    )

    print(
        "This may be a scanned/image PDF."
    )

    print()

    print(
        "STEP 2: Switching to "
        "Google Document AI OCR..."
    )

    document_ai_text = (
        extract_text_with_document_ai(
            uploaded_file
        )
    )

    # --------------------------------------------------------
    # Document AI successful
    # --------------------------------------------------------

    if document_ai_text:

        print()

        print(
            "Document AI OCR extraction successful."
        )

        print(
            "#" * 60
        )

        return document_ai_text

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print()

    print(
        "ERROR: No readable text found "
        "using either method."
    )

    print(
        "#" * 60
    )

    return ""


# ============================================================
# PDF PAGE COUNT
# ============================================================

def get_pdf_page_count(uploaded_file):
    """
    Return the number of pages in a PDF.
    """

    try:

        # Reset file pointer

        uploaded_file.seek(0)

        pdf_bytes = uploaded_file.read()

        if not pdf_bytes:

            return 0

        pdf_stream = BytesIO(
            pdf_bytes
        )

        reader = PdfReader(
            pdf_stream
        )

        return len(
            reader.pages
        )

    except Exception as e:

        print(
            "PAGE COUNT ERROR:",
            e
        )

        return 0