from rag.retriever import Retriever
from rag.generator import generate_answer


def main():

    print("\n==============================")
    print(" AI TEACHING ASSISTANT - RAG")
    print("==============================\n")

    # ---------------------------------
    # 1. Sample study material
    # ---------------------------------

    chunks = [
        """
        An operating system is system software that manages
        computer hardware and software resources. It provides
        services to application programs and acts as an interface
        between the user and computer hardware.
        """,

        """
        The main functions of an operating system include process
        management, memory management, file management, device
        management and security.
        """,

        """
        Examples of operating systems are Windows, Linux, macOS,
        Android and iOS.
        """,

        """
        A process is a program in execution. The operating system
        manages processes and allocates CPU time to them.
        """
    ]


    # ---------------------------------
    # 2. Create retriever
    # ---------------------------------

    print("Loading retriever...")

    retriever = Retriever()


    # ---------------------------------
    # 3. Create embeddings
    # ---------------------------------

    retriever.add_documents(chunks)


    # ---------------------------------
    # 4. Ask student question
    # ---------------------------------

    question = input(
        "\nAsk your question: "
    )


    # ---------------------------------
    # 5. Retrieve relevant chunks
    # ---------------------------------

    results = retriever.search(
        question,
        top_k=3
    )


    if not results:

        print(
            "\nNo relevant study material found."
        )

        return


    # ---------------------------------
    # 6. Display retrieved information
    # ---------------------------------

    print("\n==============================")
    print("RETRIEVED STUDY MATERIAL")
    print("==============================")

    context_parts = []

    for i, result in enumerate(results, start=1):

        print(
            f"\nResult {i} "
            f"(Similarity: {result['score']:.3f})"
        )

        print(result["text"])

        context_parts.append(
            result["text"]
        )


    # ---------------------------------
    # 7. Combine retrieved chunks
    # ---------------------------------

    context = "\n\n".join(
        context_parts
    )


    # ---------------------------------
    # 8. Send context + question
    #    to Gemini
    # ---------------------------------

    print("\n==============================")
    print("ASKING GEMINI...")
    print("==============================\n")

    answer = generate_answer(
        question,
        context
    )


    # ---------------------------------
    # 9. Display final answer
    # ---------------------------------

    print("==============================")
    print("AI TEACHING ASSISTANT")
    print("==============================\n")

    print(answer)

    print("\n==============================")
    print("END")
    print("==============================")


if __name__ == "__main__":
    main()