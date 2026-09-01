import argparse
import sys
import time

from oh_my_rag.config import CANDIDATE_K, RRF_K, TOP_K
from oh_my_rag.rag import Rag

WIDTH = 78


def parse_args():
    parser = argparse.ArgumentParser(description="Chat against the oh-my-rag index.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show every stage: expansion, each retriever's ranking, the fusion "
             "arithmetic, prompt size and per-stage timings.",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Skip the query-expansion LLM call and search the question as typed.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for the whole answer instead of printing it as it generates.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        metavar="N",
        help="How many hits to list per retriever in --debug (default 5).",
    )
    return parser.parse_args()


# --- rendering ---------------------------------------------------------------

def rule(label: str = "", right: str = "") -> str:
    left = f"── {label} " if label else "─" * 3
    tail = f" {right} " if right else ""
    fill = max(3, WIDTH - len(left) - len(tail))
    return f"\n{left}{'─' * fill}{tail}"


def ms(value: float) -> str:
    return f"{value / 1000:.2f}s" if value >= 1000 else f"{value:.0f}ms"


def short(uid: str) -> str:
    return uid.replace("data/incoming/", "")


def render_expansion(eq, trace) -> None:
    took = trace.timings.get("expansion", 0.0)
    print(rule("expansion", ms(took)))
    if not eq.expanded:
        print(f"  fell back to the original query")
        print(f"  reason     : {eq.reason}")
    else:
        print(f"  keywords   : {', '.join(eq.keywords) or '(none)'}")
        for i, alt in enumerate(eq.alternates, start=1):
            print(f"  ALT{i}       : {alt}")
    print(f"  dense gets : {len(eq.vector_queries())} phrasing(s)")
    print(f"  lexical gets: {', '.join(eq.fts_terms()) or '(nothing)'}")


def render_branches(trace, show: int) -> None:
    wall = trace.timings.get("  retrieval (both branches, concurrent)", 0.0)
    print(rule("retrieval", f"{ms(wall)} wall, 2 threads"))
    for b in trace.branches:
        head = f"  {b.name:<12} weight={b.weight:<4} {len(b.hits):>2} hits"
        if b.candidates is not None:
            head += f"   {b.candidates} chunk(s) matched before LIMIT {CANDIDATE_K}"
        if b.query:
            head += f'   "{b.query[:44]}"'
        print(("  [FAILED] " + head.strip()) if b.failed else head)
        if b.detail:
            print(f"      MATCH {b.detail[:WIDTH - 12]}")
        for hit in b.hits[:show]:
            print(f"      {hit.rank:>2}. {short(hit.chunk_uid):<44} score={hit.score:.4f}")
        if len(b.hits) > show:
            print(f"      … {len(b.hits) - show} more")


def render_fusion(trace) -> None:
    weights = {b.name: b.weight for b in trace.branches}
    print(rule("fusion", f"RRF k={RRF_K} → top {TOP_K}"))
    for i, hit in enumerate(trace.fused, start=1):
        parts = [
            f"{name}@{rank} ({weights.get(name, 1.0) / (RRF_K + rank):.5f})"
            for name, rank in sorted(hit.sources.items(), key=lambda kv: kv[1])
        ]
        print(f"  {i}. {short(hit.chunk_uid):<44} {hit.score:.5f}")
        print(f"       = {' + '.join(parts)}")


def render_trace(result, show: int) -> None:
    trace = result.trace
    render_expansion(result.expanded, trace)
    render_branches(trace, show)
    render_fusion(trace)
    print(rule("prompt"))
    print(f"  {trace.prompt_chars} chars across {len(result.contexts)} document block(s)")
    for w in trace.warnings:
        print(f"  ! {w}")


# --- loop --------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rag = Rag(expand_queries=not args.no_expand)

    # Load the embedding model now rather than inside the first question. It is
    # lazy by design (an MCP host wants a fast handshake), but leaving it lazy
    # here charges ~3s of one-off model loading to the first query's retrieval
    # timing, which makes --debug lie about steady-state cost.
    print("Loading models...", end="", flush=True)
    rag.search.vector.search(["warmup"], 1)
    print(" ready.  Ask a question, or /exit.")

    while True:
        try:
            q = input("\n\nQuery> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q.strip():
            continue
        if q.lower() in {"/exit", "/quit"}:
            break

        started = time.perf_counter()
        result = rag.prepare(q, trace=args.debug)
        if args.debug:
            render_trace(result, args.show)

        if args.no_stream:
            gen_started = time.perf_counter()
            answer = rag.ask_llm(result.prompt)
            gen_ms = (time.perf_counter() - gen_started) * 1000
            print(rule("answer", ms(gen_ms)) if args.debug else "\nResponse:")
            print(answer)
        else:
            print(rule("answer", "streaming") if args.debug else "\nResponse:")
            gen_started = time.perf_counter()
            first_token_ms = None
            chars = 0
            for piece in rag.stream_answer(result.prompt):
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - gen_started) * 1000
                chars += len(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
            gen_ms = (time.perf_counter() - gen_started) * 1000
            print()
            if args.debug:
                rate = chars / (gen_ms / 1000) if gen_ms else 0
                print(
                    f"\n  first token {ms(first_token_ms or 0)}  ·  "
                    f"{chars} chars in {ms(gen_ms)}  ·  {rate:.0f} char/s"
                )

        if args.debug:
            total = (time.perf_counter() - started) * 1000
            retrieval = result.trace.timings.get(
                "  retrieval (both branches, concurrent)", 0.0
            )
            expansion = result.trace.timings.get("expansion", 0.0)
            print(rule("total", ms(total)))
            print(
                f"  expansion {ms(expansion)}  ·  retrieval {ms(retrieval)}  ·  "
                f"generation {ms(gen_ms)}"
            )


if __name__ == "__main__":
    main()
