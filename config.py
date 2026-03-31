"""
RAG Chatbot Configuration — Dual-AI Architecture
Points to your existing llama-server proxy at localhost:8000
"""

# ─── Your existing API server ────────────────────────────────────────────────
API_BASE_URL    = "http://localhost:8000"          # your proxy port

# Researcher (Qwen3 with thinking mode)
RESEARCHER_MODEL       = "qwen-3.5vl-Q4-Claude-thinking"         # must match a key in your MODEL_CONFIGS
RESEARCHER_MAX_TOKENS  = 4096                             # more tokens for reasoning
RESEARCHER_TEMPERATURE = 0.0                              # deterministic for reasoning

# Chatbot (fast model for conversation)
CHATBOT_MODEL       = "qwen-3.5vl-Q4-Claude-thinking"         # can use same or faster model
CHATBOT_MAX_TOKENS  = 2048
CHATBOT_TEMPERATURE = 0.7                              # more creative for conversation

FALLBACK_MODEL  = "mistral-7b"

# ─── Embedding model (runs locally via sentence-transformers, ~90MB) ─────────
EMBED_MODEL     = "D:/models/nomic-embed-v2-moe"             # fast, good quality, 384-dim

# ─── ChromaDB ────────────────────────────────────────────────────────────────
CHROMA_DIR      = "./vectorstore"                  # persisted on disk
COLLECTION_NAME = "rag_documents"

# ─── PDF ingestion ───────────────────────────────────────────────────────────
PDF_DIR         = "./pdfs"                         # drop your PDFs here
CHUNK_SIZE      = 512                              # tokens per chunk (approx chars/4)
CHUNK_OVERLAP   = 150                              # overlap between chunks

# ─── Retrieval ───────────────────────────────────────────────────────────────
TOP_K           = 8                                # chunks injected per query
SCORE_THRESHOLD = 0.0                              # min similarity (0 = no filter)

# ─── Researcher Loop budget ──────────────────────────────────────────────────
LOOP_BUDGET     = 5                                # max reasoning iterations per question

# ─── Terminal UI ─────────────────────────────────────────────────────────────
UI_WIDTH        = 80                               # terminal panel width
CHUNK_TRUNCATE  = 80                               # truncate chunk text to this length

# ─── System prompt injected with every RAG query ────────────────────────────
SYSTEM_PROMPT = """
You are a precise research assistant.

STRICT RULES:
- Answer ONLY from the provided excerpts.
- You MUST prioritize explicitly stated facts over interpretation.
- If a concept is clearly defined in the text (e.g., “three main levels”), you MUST include it.
- Do NOT ignore exact phrases that match the question.

GROUNDING:
- Every key statement MUST be directly supported by the excerpts.
- Prefer quoting exact phrases when they directly answer the question.
- Do NOT generalize or reinterpret if exact wording exists.

RELEVANCE FILTER:
- Focus ONLY on content directly related to the question.
- Ignore unrelated technical details even if they appear in the excerpts.

UNCERTAINTY:
- If the answer is partially found, say: "Based on the provided excerpts..."
- If NOT found, say exactly: "Not found in the provided documents"

FORBIDDEN:
- No external knowledge
- No assumptions
- No adding structure not present in text

"""
