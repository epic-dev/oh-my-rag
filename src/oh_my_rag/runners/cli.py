from oh_my_rag.rag import Rag


def main() -> None:
    rag = Rag()
    while True:
        q = input("\n\nQuery> ")
        if q.lower() in {"/exit", "/quit"}:
            break
        print(f"\n\nResponse:\n{rag.answer(q)}")


if __name__ == "__main__":
    main()
