import argparse

from oh_my_rag.rag import Rag


def parse_args():
    parser = argparse.ArgumentParser(description="Chat against the oh-my-rag index.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the query expansion, which retriever found each chunk, and the "
             "fused ranking before the answer.",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Skip the query-expansion LLM call and search the question as typed.",
    )
    return parser.parse_args()


def _print_debug(result) -> None:
    eq = result.expanded
    if eq.expanded:
        print(f"\n  expansion   : keywords={eq.keywords}")
        for i, alt in enumerate(eq.alternates, start=1):
            print(f"                ALT{i}: {alt}")
    else:
        print(f"\n  expansion   : (not expanded — {eq.reason})")
    print(f"  fts terms   : {eq.fts_terms()}")
    print(f"  fused top {len(result.contexts)}:")
    for i, hit in enumerate(result.contexts, start=1):
        found = ", ".join(f"{name}@{rank}" for name, rank in sorted(hit.sources.items()))
        print(f"    {i}. {hit.chunk_uid}  score={hit.score:.5f}  [{found}]")


def main() -> None:
    args = parse_args()
    rag = Rag(expand_queries=not args.no_expand)

    while True:
        q = input("\n\nQuery> ")
        if q.lower() in {"/exit", "/quit"}:
            break

        if args.debug:
            result = rag.answer_verbose(q)
            _print_debug(result)
            print(f"\n\nResponse:\n{result.answer}")
        else:
            print(f"\n\nResponse:\n{rag.answer(q)}")


if __name__ == "__main__":
    main()
