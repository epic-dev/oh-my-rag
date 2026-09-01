# --- LLM (answering) ---------------------------------------------------------
LLM_BASE_URL = "http://localhost:11434/v1/chat/completions"
LLM_MODEL = "mlx-community/Qwen3-1.7B-4bit"
# Previously unbounded: a hung mlx-lm server hung the CLI forever.
LLM_TIMEOUT = 120.0

# --- Embeddings --------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

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
# Qwen3 reasoning suppression, verified through mlx-lm: 5/5 expansions at 369ms
# mean vs 4/5 at 887ms with reasoning on. Faster AND more compliant — but the
# model then rarely emits ALT lines, so expansion contributes keywords more than
# extra dense queries. Set False to trade latency for alternates.
EXPANSION_NO_THINK = True

# --- Full-text search (STEP 2) -----------------------------------------------
FTS_PATH = "./fts_index.sqlite3"
# Porter stemming merges inflections reliably ("retrieve" matches retrieves/
# retrieved/retrieving, "policy" matches policies) and some derivations
# ("retrieval", "connection", "computation") but not all ("deployment",
# "accrual", "runner" do NOT merge). Verified against this SQLite build.
# Changing this requires `uv run oh-my-rag --rebuild`.
FTS_TOKENIZER = "porter unicode61 remove_diacritics 2"

# --- Hybrid search and fusion (STEP 3) ---------------------------------------
TOP_K = 5
# Depth pulled from each retriever before fusion. Fusion needs more candidates
# than it returns, or there is nothing to fuse.
CANDIDATE_K = 20
RRF_K = 60
RRF_WEIGHT_VECTOR = 1.0
# Alternates are the model's paraphrase, not the user's words: a vote, quieter.
RRF_WEIGHT_ALTERNATE = 0.7
RRF_WEIGHT_FTS = 1.0
SEARCH_TIMEOUT = 15.0

# --- MCP (STEP 5) ------------------------------------------------------------
MCP_SERVER_NAME = "oh-my-rag"
MCP_HTTP_PORT = 8000
