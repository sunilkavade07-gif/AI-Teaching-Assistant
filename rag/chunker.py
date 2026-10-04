def clean_text(text):
    """
    Clean extracted PDF text.
    """

    if not text:
        return ""

    # Remove excessive spaces
    lines = []

    for line in text.splitlines():
        line = line.strip()

        if line:
            lines.append(line)

    # Join lines
    cleaned_text = "\n".join(lines)

    return cleaned_text


def create_chunks(
    text,
    chunk_size=1000,
    chunk_overlap=150
):
    """
    Split document text into overlapping chunks.

    Args:
        text: Full extracted document text
        chunk_size: Maximum approximate characters per chunk
        chunk_overlap: Characters repeated between chunks

    Returns:
        list: List of text chunks
    """

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        # Move forward while keeping overlap
        start = end - chunk_overlap

    return chunks