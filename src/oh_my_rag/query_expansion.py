import sys
import re
from dataclasses import dataclass, field

from oh_my_rag.config import (
    EXPANSION_ENABLED,
    EXPANSION_MAX_ALTERNATES,
    EXPANSION_MAX_KEYWORDS,
    EXPANSION_MODEL,
    EXPANSION_MAX_TOKENS,
    EXPANSION_NO_THINK,
    EXPANSION_REASONING_EFFORT,
    EXPANSION_TIMEOUT,
    LLM_BASE_URL,
)
from oh_my_rag.llm_client import chat

# Small stoplist. BM25 already discounts common terms by IDF, but leaving them
# in an OR'd MATCH inflates the candidate set for no ranking benefit.
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from had has have how i if in into is it
    its me my not of on or should that the their them there these they this to was we
    were what when where which who why will with would you your""".split()
)

_STRIP = ".,;:!?()[]{}<>\"'`"


def tokenize(text: str) -> list[str]:
    """Content words from a raw question, for when expansion produced no keywords.

    Splits on whitespace and strips surrounding punctuation only, so identifiers
    that matter to a lexical search survive intact: `C++`, `AP2`, `bge-small`.
    """
    seen: set[str] = set()
    out: list[str] = []
    for token in re.split(r"\s+", text):
        token = token.strip(_STRIP)
        if not token or not any(ch.isalnum() for ch in token):
            continue
        folded = token.casefold()
        if folded in _STOPWORDS or folded in seen:
            continue
        seen.add(folded)
        out.append(token)
    return out


@dataclass(frozen=True)
class ExpandedQuery:
    """What the search layer consumes.

    `expanded=False` means STEP 6 fired and this is the untouched question, so
    `--debug` and MCP responses can tell a real expansion from a fallback
    instead of degrading silently.
    """

    original: str
    keywords: list[str] = field(default_factory=list)
    alternates: list[str] = field(default_factory=list)
    expanded: bool = False
    reason: str | None = None

    @classmethod
    def passthrough(cls, query: str, reason: str | None = None) -> "ExpandedQuery":
        return cls(original=query, expanded=False, reason=reason)

    def vector_queries(self) -> list[str]:
        """The user's words first — fusion weights that list highest."""
        return [self.original, *self.alternates]

    def fts_terms(self) -> list[str]:
        return self.keywords or tokenize(self.original)


# Three lines, because every extra instruction is another thing a 4-bit 1.7B
# model can drift on. "key nouns" rather than "terms" because porter stemming
# merges inflections but not -al/-ment derivations, so base nouns match more.
SYSTEM_PROMPT = """You expand search queries. Reply with exactly these 3 lines, nothing else:
KEYWORDS: comma-separated key nouns from the question
ALT1: the question, rephrased
ALT2: the question, rephrased differently"""

# Independent per line: a mangled ALT2 costs ALT2 and nothing else.
_KEYWORDS_RE = re.compile(r"^[^\S\n]*KEYWORDS[^\S\n]*:[^\S\n]*(.+)$", re.IGNORECASE | re.MULTILINE)
_ALT1_RE = re.compile(r"^[^\S\n]*ALT[^\S\n]*1[^\S\n]*:[^\S\n]*(.+)$", re.IGNORECASE | re.MULTILINE)
_ALT2_RE = re.compile(r"^[^\S\n]*ALT[^\S\n]*2[^\S\n]*:[^\S\n]*(.+)$", re.IGNORECASE | re.MULTILINE)

_MAX_KEYWORD_CHARS = 40
_MAX_ALTERNATE_CHARS = 300


def _clean_keywords(raw: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for part in raw.split(","):
        term = part.strip().strip("\"'`*-").strip()
        if not term or not any(ch.isalnum() for ch in term):
            continue
        term = term[:_MAX_KEYWORD_CHARS]
        folded = term.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(term)
        if len(out) >= EXPANSION_MAX_KEYWORDS:
            break
    return out


def _clean_alternates(raws: list[str], original: str) -> list[str]:
    original_folded = original.strip().casefold()
    seen: set[str] = set()
    out: list[str] = []
    for raw in raws:
        alt = raw.strip().strip("\"'`*").strip()
        if not alt or not any(ch.isalnum() for ch in alt):
            continue
        alt = alt[:_MAX_ALTERNATE_CHARS]
        folded = alt.casefold()
        # A paraphrase identical to the question adds a duplicate ranked list
        # and no recall.
        if folded == original_folded or folded in seen:
            continue
        seen.add(folded)
        out.append(alt)
        if len(out) >= EXPANSION_MAX_ALTERNATES:
            break
    return out


def _looks_like_a_keyword_list(reply: str) -> bool:
    """Is this a bare keyword list that simply lost its tag?

    Qwen3-1.7B often answers this prompt with the right content in the wrong
    shape — `AP2, UCP` instead of `KEYWORDS: AP2, UCP`. Throwing that away
    costs the whole lexical branch, so it is salvaged when the reply is
    structurally a list: short, comma-separated, and no part long enough to be
    a sentence. Prose without commas is not salvaged.
    """
    text = reply.strip()
    if not text or len(text) > 150 or text.count("\n") > 1 or "," not in text:
        return False
    parts = [p.strip() for p in text.replace("\n", " ").split(",")]
    return 2 <= len(parts) <= EXPANSION_MAX_KEYWORDS and all(
        0 < len(p) <= _MAX_KEYWORD_CHARS for p in parts
    )


def parse_expansion(reply: str, original: str) -> tuple[list[str], list[str]]:
    """Pull keywords and alternates out of a reply. Never raises."""
    kw_match = _KEYWORDS_RE.search(reply)
    keywords = _clean_keywords(kw_match.group(1)) if kw_match else []

    raw_alts = []
    for pattern in (_ALT1_RE, _ALT2_RE):
        match = pattern.search(reply)
        if match:
            raw_alts.append(match.group(1))
    alternates = _clean_alternates(raw_alts, original)

    if not keywords and not alternates and _looks_like_a_keyword_list(reply):
        keywords = _clean_keywords(reply)

    return keywords, alternates


def expand(
    query: str,
    *,
    base_url: str = LLM_BASE_URL,
    model: str = EXPANSION_MODEL,
    timeout: float = EXPANSION_TIMEOUT,
    enabled: bool = EXPANSION_ENABLED,
    no_think: bool = EXPANSION_NO_THINK,
    reasoning_effort: str | None = EXPANSION_REASONING_EFFORT,
) -> ExpandedQuery:
    """STEP 1, with STEP 6 built in.

    Guarantees an ExpandedQuery. Never raises, never returns None, never
    propagates an HTTP error — every failure collapses to the original query,
    which leaves the pipeline exactly where it was before expansion existed.
    """
    if not enabled:
        return ExpandedQuery.passthrough(query, reason="expansion disabled")
    if not query.strip():
        return ExpandedQuery.passthrough(query, reason="empty query")

    # Qwen3 soft switch. Harmless to other models, which see it as trailing text.
    user_content = f"{query} /no_think" if no_think else query

    try:
        reply = chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            base_url=base_url,
            model=model,
            timeout=timeout,
            temperature=0,
            max_tokens=EXPANSION_MAX_TOKENS,
            reasoning_effort=reasoning_effort,
            kind="expansion",
        )
    except Exception as exc:
        return _fallback(query, f"{type(exc).__name__}: {exc}")

    if not reply.strip():
        # A reasoning model can spend its whole token budget inside <think>,
        # leaving nothing once the block is stripped.
        return _fallback(query, "empty reply after stripping control tokens")

    keywords, alternates = parse_expansion(reply, query)
    if not keywords and not alternates:
        return _fallback(query, f"no tagged lines in reply: {reply[:80]!r}")

    return ExpandedQuery(
        original=query,
        keywords=keywords,
        alternates=alternates,
        expanded=True,
    )


def _fallback(query: str, reason: str) -> ExpandedQuery:
    print(f"[expansion] falling back to the original query ({reason})", file=sys.stderr)
    return ExpandedQuery.passthrough(query, reason=reason)
