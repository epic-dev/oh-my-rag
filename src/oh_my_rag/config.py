# --- LLM (answering) ---------------------------------------------------------
# OpenAI-compatible /chat/completions. Ollama and mlx-lm both serve this shape on
# :11434, so switching backends is a matter of changing LLM_MODEL to match
# whichever is running (`ollama list`, or the mlx-community repo id).
LLM_BASE_URL = "http://localhost:11434/v1/chat/completions"
LLM_MODEL = "qwen3.5:9b"
# Previously unbounded: a hung mlx-lm server hung the CLI forever.
LLM_TIMEOUT = 120.0
# mlx-lm defaults to 512 completion tokens when the request omits max_tokens,
# which silently truncated answers mid-sentence — and on a reasoning model the
# <think> block eats part of that budget before the answer even starts.
LLM_MAX_TOKENS = 2048
# Answering from retrieved context is extraction and synthesis, not a reasoning
# problem — and a reasoning model's scratchpad shares the LLM_MAX_TOKENS budget.
# Measured on qwen3.5:9b over this corpus: 23.0s mean with reasoning off vs
# 87.7s with it on, for answers of equivalent quality (one question exceeded
# LLM_TIMEOUT entirely). Honoured by Ollama; harmless elsewhere since the field
# is only sent when set. None lets the model think.
LLM_REASONING_EFFORT = "none"

# --- Embeddings --------------------------------------------------------------
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Prepended to queries only — documents are embedded from their raw text. This
# string belongs to EMBEDDING_MODEL and must be changed with it: each family has
# its own convention (this is Qwen3-Embedding's, copied verbatim from the
# model's own config_sentence_transformers.json — note there is no space
# after "Query:"; bge instead wants "Represent this sentence for searching
# relevant passages: "). A mismatch raises nothing, it just quietly degrades
# retrieval, so the two constants sit together to keep them in step.
EMBEDDING_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query:"
)

# --- Chunking ----------------------------------------------------------------
CHUNKS_PATH = "chunks.pkl"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# --- Stores ------------------------------------------------------------------
CHROMADB_PATH = "./chromadb_base"
DATA_SOURCE_DIR = "data/incoming"

# --- Query expansion (STEP 1 / STEP 6) ---------------------------------------
EXPANSION_ENABLED = True
# Same mlx-lm server as answering; separate constant so a smaller/faster model
# can be pointed at expansion without touching the answering path.
EXPANSION_MODEL = LLM_MODEL
EXPANSION_TIMEOUT = 20.0
EXPANSION_MAX_TOKENS = 200
EXPANSION_MAX_KEYWORDS = 8
EXPANSION_MAX_ALTERNATES = 2
# Expansion is a formatting task, so a reasoning model's scratchpad is pure cost.
# Two mechanisms, because the two backends honour different ones:
#
#   Ollama    -> reasoning_effort="none".  Measured on qwen3.5:9b: 4.1s and a
#                correct answer, vs 12.1s and an EMPTY reply at max_tokens=200
#                (the scratchpad consumed the whole budget before any content).
#   mlx-lm    -> the "/no_think" suffix. Measured on Qwen3-1.7B-4bit: 5/5
#                expansions at 369ms vs 4/5 at 887ms. Ollama ignores it.
#
# Set EXPANSION_REASONING_EFFORT to None to let the model think.
EXPANSION_REASONING_EFFORT = "none"
EXPANSION_NO_THINK = False

# --- Full-text search (STEP 2) -----------------------------------------------
FTS_PATH = "./fts_index.sqlite3"
# Porter stemming merges inflections reliably ("retrieve" matches retrieves/
# retrieved/retrieving, "policy" matches policies) and some derivations
# ("retrieval", "connection", "computation") but not all ("deployment",
# "accrual", "runner" do NOT merge). Verified against this SQLite build.
# Changing this requires `uv run oh-my-rag --rebuild`.
FTS_TOKENIZER = "porter unicode61 remove_diacritics 2"

# --- Hybrid search and fusion (STEP 3) ---------------------------------------
TOP_K = 3
# Depth pulled from each retriever before fusion. Fusion needs more candidates
# than it returns, or there is nothing to fuse.
CANDIDATE_K = 20
RRF_K = 60
RRF_WEIGHT_VECTOR = 1.0
# Alternates are the model's paraphrase, not the user's words: a vote, quieter.
RRF_WEIGHT_ALTERNATE = 0.7
RRF_WEIGHT_FTS = 1.0
SEARCH_TIMEOUT = 15.0

# --- Usage log ---------------------------------------------------------------
# Every call to the LLM is appended here: tokens in/out, latency, and failures.
# Its own database, so wiping or rotating it never touches the retrieval indexes.
USAGE_DB_PATH = "./llm_usage.sqlite3"
USAGE_LOG_ENABLED = True

# --- MCP (STEP 5) ------------------------------------------------------------
MCP_SERVER_NAME = "oh-my-rag"
MCP_HTTP_PORT = 8000
