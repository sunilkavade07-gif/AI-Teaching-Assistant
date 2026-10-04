import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rag.chunker import create_chunks


sample_text = """
Government Polytechnic Pune offers diploma-level technical education.
Students study various engineering subjects.
The institute provides practical and theoretical learning.
Students can participate in projects, practicals and technical activities.
"""


chunks = create_chunks(
    sample_text,
    chunk_size=100,
    chunk_overlap=20
)


print("==============================")
print("CHUNKING TEST")
print("==============================")

print("Total chunks:", len(chunks))

for number, chunk in enumerate(chunks, start=1):

    print(f"\n--- Chunk {number} ---")
    print(chunk)