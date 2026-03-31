"""
Researcher Module — Qwen3 with thinking mode
Performs iterative retrieval and gap analysis with streaming output
Features smart chunk management to prevent context overflow
"""

import json
import httpx
import re
import time
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    API_BASE_URL, RESEARCHER_MODEL, RESEARCHER_MAX_TOKENS, RESEARCHER_TEMPERATURE,
    CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL, TOP_K, LOOP_BUDGET
)
from terminal_ui import (
    console,
    print_chunks_retrieved,
    print_thinking,
    print_thinking_header,
    print_thinking_footer,
    print_final_answer_header,
    print_budget_warning,
    stream_token,
    stream_thinking_token,
    print_loop_decision,
    print_researcher_loop_header,
    print_researcher_json_output,
    print_expand_chunks,
    COLOR_TEXT, COLOR_SUCCESS, COLOR_MUTED, COLOR_RESEARCHER
)

# ─── Startup ──────────────────────────────────────────────────────────────────

print("📦 Loading embedding model...", end=" ", flush=True)
embed_model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
print(f"✅ (dim={embed_model.get_sentence_embedding_dimension()})")

print("📂 Connecting to ChromaDB...", end=" ", flush=True)
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)
print(f"✅ ({collection.count():,} chunks)")


# ─── Chunk Registry ───────────────────────────────────────────────────────────

class ChunkRegistry:
    """
    Manages chunks outside the LLM context window.
    - Stores full chunk content internally
    - Provides summarized references for previous loops
    - Only returns full text for newly retrieved chunks
    """
    
    def __init__(self):
        self.chunks = {}  # id -> full chunk data
        self.order = []   # list of chunk ids in retrieval order
        self.annotations = {}  # id -> annotation string (USEFUL/IRRELEVANT + reason)
    
    def add_chunks(self, chunks: list[dict]):
        """Add new chunks to the registry."""
        for chunk in chunks:
            chunk_id = chunk["id"]
            if chunk_id not in self.chunks:
                self.chunks[chunk_id] = chunk
                self.order.append(chunk_id)
    
    def annotate_chunk(self, chunk_id: str, annotation: str):
        """Mark a chunk as USEFUL or IRRELEVANT with optional reason."""
        self.annotations[chunk_id] = annotation
    
    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Get full chunk data by ID."""
        return self.chunks.get(chunk_id)
    
    def get_all_ids(self) -> list[str]:
        """Get all chunk IDs in retrieval order."""
        return list(self.order)
    
    def build_context(self, new_chunk_ids: list[str], max_prev_refs: int = 10) -> str:
        """
        Build context for LLM:
        - Previous chunks: one-line references only
        - New chunks: full text
        """
        parts = []
        
        # Previous chunks summary
        prev_ids = [cid for cid in self.order if cid not in new_chunk_ids]
        if prev_ids:
            parts.append("=== PREVIOUS LOOPS SUMMARY ===")
            for i, cid in enumerate(prev_ids[-max_prev_refs:], 1):
                chunk = self.chunks[cid]
                ref = f"[CHUNK {i}] {chunk['source']} p.{chunk['page']}"
                annotation = self.annotations.get(cid, "")
                if annotation:
                    ref += f" — {annotation}"
                parts.append(ref)
            if len(prev_ids) > max_prev_refs:
                parts.append(f"... and {len(prev_ids) - max_prev_refs} more chunks from earlier loops")
            parts.append("")
        
        # New chunks with full text
        if new_chunk_ids:
            parts.append("=== NEW CHUNKS THIS LOOP ===")
            for i, cid in enumerate(new_chunk_ids, 1):
                chunk = self.chunks[cid]
                parts.append(f"[CHUNK {i}] {chunk['source']} p.{chunk['page']} score:{chunk['score']}")
                parts.append(chunk['text'])
                parts.append("")
        
        return "\n".join(parts)
    
    def get_chunks_by_page_range(self, source: str, page: int, range_pages: int = 1) -> list[dict]:
        """Get chunks for pages page-range_pages to page+range_pages from a source."""
        results = []
        for cid in self.order:
            chunk = self.chunks[cid]
            if chunk['source'] == source:
                chunk_page = int(chunk['page']) if str(chunk['page']).isdigit() else 0
                if abs(chunk_page - page) <= range_pages:
                    results.append(chunk)
        return results


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve_chunks(keywords: list[str], exclude_ids: set[str], top_k: int = TOP_K, source_filter: str = None) -> list[dict]:
    """Embed keywords and retrieve chunks from ChromaDB, excluding already-seen IDs."""
    query_text = " ".join(keywords)
    query_embed = embed_model.encode([query_text]).tolist()

    query_params = {
        "query_embeddings": query_embed,
        "n_results": top_k * 2,  # Get extra to filter out excluded
        "include": ["documents", "metadatas", "distances"],
    }
    if source_filter:
        query_params["where"] = {"source": source_filter}

    results = collection.query(**query_params)

    chunks = []
    if results["ids"] and results["ids"][0]:
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            if doc_id not in exclude_ids:
                chunks.append({
                    "id": doc_id,
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", "?"),
                    "chunk": meta.get("chunk", "?"),
                    "score": round(1 - dist, 3),
                })
            if len(chunks) >= top_k:
                break

    return chunks


def fetch_adjacent_pages(source: str, page: int, range_size: int = 1) -> list[dict]:
    """Fetch chunks from pages adjacent to the specified page."""
    min_page = page - range_size
    max_page = page + range_size
    
    # Build query for adjacent content
    query_text = f"{source} page {page}"
    query_embed = embed_model.encode([query_text]).tolist()
    
    # Get more results to find adjacent pages
    results = collection.query(
        query_embeddings=query_embed,
        n_results=50,
        include=["documents", "metadatas", "distances"],
    )
    
    adjacent = []
    seen_ids = set()
    if results["ids"] and results["ids"][0]:
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            chunk_page = int(meta.get("page", 0)) if str(meta.get("page", "")).isdigit() else 0
            if meta.get("source") == source and min_page <= chunk_page <= max_page:
                adjacent.append({
                    "id": doc_id,
                    "text": doc,
                    "source": source,
                    "page": chunk_page,
                    "chunk": meta.get("chunk", "?"),
                    "score": round(1 - dist, 3),
                })
    
    return sorted(adjacent, key=lambda c: (c['page'], c['chunk']))


# ─── Researcher System Prompt ───────────────────────────────────────────────────────

RESEARCHER_SYSTEM_PROMPT = """
You are Researcher, a precise reasoning assistant with iterative retrieval capabilities.

YOUR TASK:
1. Analyze the user's question and the retrieved chunks
2. Perform gap analysis to identify missing information
3. Decide whether you need more retrieval loops or can answer
4. When ready, provide a complete answer based ONLY on the provided chunks

STRICT RULES:
- Answer ONLY from the provided excerpts
- Every key statement MUST be directly supported by the chunks
- If NOT found in chunks, say: "Not found in the provided documents"
- No external knowledge, no assumptions

READING FORMATTED TABLES:
- Some chunks contain vendor list tables as unformatted space-separated text
- Example: "Air cooled Chiller  Trane  USA  York  USA  Daikin  Japan"
- Read these as columns: Description | Manufacturer | Country

EXPANDING CONTEXT:
- If a chunk contains the START of useful information but appears cut off, add its reference to expand_chunks
- Format: "source.pdf:p.1468" to retrieve pages p.1467, p.1468, p.1469
- Use this when you see incomplete tables, formulas, or specifications

OUTPUT FORMAT:
You MUST respond with valid JSON only:
{
    "answer": "your complete answer here",
    "confidence": "high|medium|low",
    "gaps": ["list of information gaps if any"],
    "need_more": true|false,
    "loops_used": N,
    "new_keywords": ["keywords for next retrieval if needed"],
    "expand_chunks": ["source.pdf:p.123", "source.pdf:p.456"]
}

CONFIDENCE LEVELS:
- high: All information found, well-supported answer
- medium: Most information found, some minor gaps  
- low: Significant gaps or conflicting information

THINKING MODE (concise):
Include a brief <think> section before the JSON where you:

- Identify whether the retrieved chunks contain the required information
- List any missing fields required by the question
- Decide whether more retrieval is needed
- Avoid repeating previous analysis
- Keep this section under 5 lines


IMPORTANT: Output the <think> section first, then the JSON. Do not mix them.
""".strip()


# ─── LLM Call with Streaming ──────────────────────────────────────────────────

def call_researcher_stream(messages: list[dict], model: str = RESEARCHER_MODEL):
    """
    Call Researcher with streaming. Yields (is_thinking, token) tuples.
    Thinking tokens come from <think>...</think> tags or reasoning_content field.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": RESEARCHER_MAX_TOKENS,
        "temperature": RESEARCHER_TEMPERATURE,
        "stream": True,
    }

    in_thinking = False
    thinking_buffer = ""

    with httpx.Client(timeout=httpx.Timeout(300, connect=10)) as client:
        with client.stream("POST", f"{API_BASE_URL}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"]

                        # Check for reasoning_content field (some APIs send thinking separately)
                        reasoning_token = delta.get("reasoning_content", "")
                        if reasoning_token:
                            yield (True, reasoning_token)
                            continue

                        token = delta.get("content", "")
                        if token:
                            # Check for thinking tags in content
                            if "<<think>>" in token or "<think>" in token:
                                in_thinking = True
                                token = re.sub(r'<?/?<think>>', '', token)
                            elif "</<think>>" in token or "</think>" in token:
                                in_thinking = False
                                token = re.sub(r'</?</think>>', '', token)

                            if in_thinking:
                                thinking_buffer += token
                                yield (True, token)
                            else:
                                yield (False, token)
                    except Exception as e:
                        console.print(f"\n  [Error parsing stream chunk: {e}]", style="dim red")
                        continue


def parse_researcher_response(full_response: str) -> dict:
    """Parse Researcher's JSON response, handling potential markdown wrapping."""
    # Try to extract JSON from the response
    json_match = re.search(r'\{[\s\S]*\}', full_response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return structured default
    return {
        "answer": full_response.strip(),
        "confidence": "medium",
        "gaps": [],
        "need_more": False,
        "loops_used": 1,
        "new_keywords": [],
        "expand_chunks": []
    }


# ─── Researcher Main Loop ───────────────────────────────────────────────────────────

def run_researcher_reasoning(user_question: str, initial_keywords: list[str], source_filter: str = None, stream_callback=None) -> dict:
    """
    Run Researcher's iterative reasoning loop with smart chunk management.
    Returns structured result with answer, confidence, and metadata.
    """
    registry = ChunkRegistry()
    new_keywords = list(initial_keywords)
    loops_used = 0

    researcher_start_time = time.perf_counter()
    loop_times = []
    chunk_annotations = []  # Track useful/irrelevant for final synthesis

    for loop_num in range(1, LOOP_BUDGET + 1):
        loops_used = loop_num
        loop_start = time.perf_counter()

        new_chunk_ids = []

        # ── Step 1: Retrieve chunks ─────────────────────────────────────────
        if stream_callback:
            stream_callback("status", f"Loop {loop_num}/{LOOP_BUDGET} — searching: {', '.join(new_keywords)}")
        chunks = retrieve_chunks(new_keywords, set(registry.get_all_ids()), source_filter=source_filter)

        if chunks:
            print_chunks_retrieved(chunks, new_start_idx=len(registry.chunks))
            registry.add_chunks(chunks)
            new_chunk_ids = [c["id"] for c in chunks]

        # ── Step 2: Build context & show Researcher input header ─────────────────
        context = registry.build_context(new_chunk_ids)

        print_researcher_loop_header(
            loop_num=loop_num,
            max_loops=LOOP_BUDGET,
            keywords=new_keywords,
            total_chunks=len(registry.chunks),
        )

        messages = [
            {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"USER QUESTION: {user_question}\n\n"
                f"{context}\n\n"
                f"LOOP: {loop_num}/{LOOP_BUDGET}\n"
                f"TOTAL CHUNKS IN REGISTRY: {len(registry.chunks)}\n\n"
                "Provide your gap analysis in <think>...</think> tags, then your JSON response."
            )}
        ]

        # ── Step 3: Stream Researcher thinking + response ─────────────────────────
        thinking_lines = []
        json_response = ""

        print_thinking_header()

        for is_thinking, token in call_researcher_stream(messages):
            if stream_callback and is_thinking:
                stream_callback("thinking", token)
                
            if is_thinking:
                stream_thinking_token(token)
                thinking_lines.append(token)
            else:
                json_response += token

        print_thinking_footer()

        loop_elapsed = time.perf_counter() - loop_start
        loop_times.append(loop_elapsed)

        # ── Step 4: Parse and display Researcher output ──────────────────────────
        result = parse_researcher_response(json_response)
        result["loops_used"] = loops_used
        if stream_callback:
            stream_callback("reasoning", result)

        print_researcher_json_output(result, loop_num, loop_elapsed)

        # ── Step 5: Handle expand_chunks ────────────────────────────────────
        need_more = result.get("need_more", False)
        expand_chunks = result.get("expand_chunks", [])

        if expand_chunks and loop_num < LOOP_BUDGET:
            print_expand_chunks(expand_chunks)
            expanded_ids = []
            for ref in expand_chunks:
                if ":" in ref:
                    source, page_str = ref.rsplit(":", 1)
                    try:
                        page = int(page_str)
                        adjacent = fetch_adjacent_pages(source, page)
                        if adjacent:
                            registry.add_chunks(adjacent)
                            expanded_ids.extend([c["id"] for c in adjacent])
                            console.print(
                                f"    ✓ {source}: fetched pages around p.{page}",
                                style=f"dim {COLOR_RESEARCHER}"
                            )
                    except ValueError:
                        pass
            if expanded_ids:
                new_chunk_ids.extend(expanded_ids)

        # ── Step 6: Final or continue ────────────────────────────────────────
        if not need_more or loop_num >= LOOP_BUDGET:
            researcher_elapsed = time.perf_counter() - researcher_start_time

            answer = result.get("answer", "").strip()
            if not answer:
                answer = "The requested information was not found in the available documents after thorough search."
                result["confidence"] = "low"

            if loop_num >= LOOP_BUDGET and need_more:
                print_budget_warning(loops_used, LOOP_BUDGET)

            print_final_answer_header(
                result.get("confidence", "medium"),
                loops_used,
                LOOP_BUDGET,
            )

            return {
                "answer": answer,
                "confidence": result.get("confidence", "medium"),
                "loops_used": loops_used,
                "chunks_used": len(registry.chunks),
                "chunk_ids": registry.get_all_ids(),
                "researcher_time": researcher_elapsed,
                "loop_times": loop_times,
            }

        # Prepare for next loop
        new_keywords = result.get("new_keywords", [])
        if not new_keywords:
            new_keywords = result.get("gaps", [])

        print_loop_decision("continue retrieval", f"new keywords: {new_keywords}")

    # Should not reach here, but just in case
    researcher_elapsed = time.perf_counter() - researcher_start_time
    answer = result.get("answer", "").strip()
    if not answer:
        answer = "The requested information was not found in the available documents after thorough search."

    return {
        "answer": answer,
        "confidence": "low",
        "loops_used": loops_used,
        "chunks_used": len(registry.chunks),
        "chunk_ids": registry.get_all_ids(),
        "researcher_time": researcher_elapsed,
        "loop_times": loop_times,
    }
