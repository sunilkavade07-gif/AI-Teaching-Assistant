import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rag.retriever import Retriever


chunks = [
    "The OSI model has seven layers.",
    "TCP is a connection-oriented protocol.",
    "Python is a high-level programming language.",
    "IPv4 uses 32-bit addresses."
]


print("Creating retriever...")

retriever = Retriever()

retriever.add_documents(chunks)


question = "How many layers are there in the OSI model?"

print("\nQuestion:", question)

results = retriever.search(
    question,
    top_k=2
)


print("\n==============================")
print("SEARCH RESULTS")
print("==============================")


for result in results:

    print("\nScore:", result["score"])

    print("Text:", result["text"])