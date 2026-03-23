from src.rag_pipeline import generate_answer

print("=== RAG LegalAssist ===")
print("Ask legal questions. Type exit to quit.\n")

while True:
    q = input("Enter legal question: ")

    if q.lower() in ["exit", "quit", "stop"]:
        break

    print("\nAnswer:\n")
    print(generate_answer(q))
    print("\n------------------------------------\n")
