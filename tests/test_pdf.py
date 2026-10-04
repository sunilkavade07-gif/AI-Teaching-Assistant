import sys
from pathlib import Path
from io import BytesIO

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rag.pdf_processor import extract_text_from_pdf


# PDF location
pdf_path = project_root / "data" / "test.pdf"


print("===================================")
print("PDF EXTRACTION TEST")
print("===================================")

# Read PDF as bytes
with open(pdf_path, "rb") as pdf_file:
    pdf_bytes = pdf_file.read()

# Convert bytes into an object that supports getvalue()
uploaded_file = BytesIO(pdf_bytes)

# Extract text
text = extract_text_from_pdf(uploaded_file)


print("===================================")
print("PDF EXTRACTION RESULT")
print("===================================")

print("Characters extracted:", len(text))

print("\nFirst 1000 characters:")
print(text[:1000])

print("\n===================================")
print("TEST COMPLETE")
print("===================================")