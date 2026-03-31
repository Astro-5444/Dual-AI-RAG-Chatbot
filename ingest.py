#!/usr/bin/env python3
"""
PDF Ingestion Pipeline — GPU Accelerated
Parses PDFs → detects sections via font metadata → chunks → embeds on GPU → stores in ChromaDB

Usage:
    python ingest.py                        # ingest all PDFs in ./pdfs/
    python ingest.py path/to/file.pdf       # ingest a specific PDF
    python ingest.py --reset                # wipe vectorstore and re-ingest all
    python ingest.py --list                 # list ingested documents
    python ingest.py --sections             # preview detected sections per PDF
"""

import sys
import os
import argparse
import hashlib
import time
from pathlib import Path
from statistics import mode

import fitz  # PyMuPDF
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL,
    PDF_DIR, CHUNK_SIZE, CHUNK_OVERLAP
)

# ─── GPU Setup ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024**3
        print(f"🖥️  GPU: {props.name}  |  VRAM: {vram:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("⚠️  No GPU found — falling back to CPU")
    return dev


def optimal_batch_size(device: torch.device, base: int = 64) -> int:
    """Scale batch size to available VRAM so the GPU stays fed."""
    if device.type != "cuda":
        return base
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb >= 16:
        return base * 4       # 256
    elif vram_gb >= 8:
        return base * 2       # 128
    elif vram_gb >= 4:
        return base           # 64
    else:
        return base // 2      # 32


# ─── Init ─────────────────────────────────────────────────────────────────────

def get_chroma_collection(reset: bool = False):
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Wiped collection: {COLLECTION_NAME}")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_embed_model(device: torch.device) -> SentenceTransformer:
    print(f"📦 Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)

    # Warm up — avoids a slow first batch in production
    if device.type == "cuda":
        _ = model.encode(["warmup"], show_progress_bar=False)
        torch.cuda.empty_cache()

    dim = model.get_sentence_embedding_dimension()
    print(f"✅ Embedding model ready  |  dim={dim}  |  device={device}")
    return model


# ─── PDF Parsing ──────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text page-by-page using PyMuPDF's dict mode.
    Each page yields a list of spans with font metadata:
        text, size, bold, x_origin
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        spans  = []

        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    spans.append({
                        "text":     text,
                        "size":     round(span["size"], 1),
                        "bold":     bool(span["flags"] & (1 << 4)),  # bit 4 = bold
                        "x_origin": span["origin"][0],
                    })

        if spans:
            pages.append({"page": i + 1, "spans": spans})

    doc.close()
    total_words = sum(len(s["text"].split()) for p in pages for s in p["spans"])
    print(f"  📄 Pages: {len(pages)} | Words: {total_words:,}")
    return pages


# ─── Section Detection ────────────────────────────────────────────────────────

def detect_sections(pages: list[dict]) -> list[dict]:
    """
    Detect headings from visual properties — font size and bold flag —
    rather than text patterns.  This handles PDFs where heading levels are
    visually consistent but use arbitrary numbering schemes.

    Algorithm
    ---------
    1. Collect every font size in the document.
    2. Body size  = the statistical mode (most frequent size).
    3. A span is a heading candidate when:
         - It is noticeably larger than body (size > body + 0.5), OR
         - It is bold AND at least body size.
       AND it is short enough to be a heading (< 120 chars).
    4. Walk pages in order; the first heading candidate on a page updates
       the running section label for all subsequent chunks until the next
       heading appears.
    5. Pages are flattened to plain text for downstream chunking, with
       section metadata attached.
    """
    # Step 1 — find body size
    all_sizes = [s["size"] for p in pages for s in p["spans"]]
    if not all_sizes:
        # Degenerate PDF — no spans at all
        return [
            {**p, "text": "", "section": "Unknown", "section_page": p["page"]}
            for p in pages
        ]

    body_size = mode(all_sizes)

    def is_heading(span: dict) -> bool:
        text = span["text"].strip()
        if not text or len(text) > 120:
            return False
        larger = span["size"] > body_size + 0.5
        bold   = span["bold"] and span["size"] >= body_size
        # Reject lines that are just numbers, punctuation, or single characters
        if len(text) < 3:
            return False
        return larger or bold

    current_section      = "Preamble"
    current_section_page = 1
    enriched             = []

    for p in pages:
        # Look for the first heading-like span on this page
        for span in p["spans"]:
            if is_heading(span):
                current_section      = span["text"]
                current_section_page = p["page"]
                break   # one heading per page is enough to track context

        # Flatten spans to a single string (preserves reading order)
        full_text = " ".join(s["text"] for s in p["spans"])

        enriched.append({
            "page":         p["page"],
            "text":         full_text,
            "section":      current_section,
            "section_page": current_section_page,
        })

    return enriched


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict], pdf_name: str) -> list[dict]:
    """
    Slide a character window over the full document text.
    Each chunk carries:
        source, page, chunk index, char count, section, section_page
    """
    # Enrich pages with section labels first
    pages = detect_sections(pages)

    char_size    = CHUNK_SIZE    * 4
    char_overlap = CHUNK_OVERLAP * 4

    # Flatten all pages into one stream with position bookmarks
    full_text      = ""
    page_bookmarks = []   # (char_start, page_num, section, section_page)

    for p in pages:
        page_bookmarks.append((len(full_text), p["page"], p["section"], p["section_page"]))
        full_text += p["text"] + "\n\n"

    def get_page_info(pos: int):
        page, section, section_page = 1, "Preamble", 1
        for start, pnum, sec, sec_pg in page_bookmarks:
            if pos >= start:
                page, section, section_page = pnum, sec, sec_pg
            else:
                break
        return page, section, section_page

    chunks = []
    start  = 0
    idx    = 0

    while start < len(full_text):
        end  = min(start + char_size, len(full_text))
        text = full_text[start:end].strip()
        if text:
            page, section, section_page = get_page_info(start)
            chunk_id = hashlib.md5(
                f"{pdf_name}:{idx}:{text[:50]}".encode()
            ).hexdigest()
            chunks.append({
                "id":   chunk_id,
                "text": text,
                "metadata": {
                    "source":       pdf_name,
                    "page":         page,
                    "chunk":        idx,
                    "chars":        len(text),
                    "section":      section,       # ← which section this chunk is from
                    "section_page": section_page,  # ← page where that section started
                },
            })
            idx += 1
        start += char_size - char_overlap

    return chunks


# ─── Ingestion ────────────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: str,
    collection,
    embed_model: SentenceTransformer,
    device: torch.device,
    batch_size: int,
    force: bool = False,
):
    pdf_name = Path(pdf_path).name
    print(f"\n{'─'*60}")
    print(f"📂 Ingesting: {pdf_name}")
    t0 = time.time()

    # Skip if already ingested
    existing = collection.get(where={"source": pdf_name}, limit=1)
    if existing["ids"] and not force:
        print(f"  ⏭️  Already ingested (use --reset to re-ingest)")
        return 0

    # Parse
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        print(f"  ⚠️  No text extracted — is the PDF scanned? (OCR not supported)")
        return 0

    # Chunk (section detection happens inside)
    chunks = chunk_pages(pages, pdf_name)
    print(f"  ✂️  Chunks: {len(chunks):,} (size≈{CHUNK_SIZE} tokens, overlap≈{CHUNK_OVERLAP})")

    # Preview unique sections found
    seen_sections = []
    for c in chunks:
        sec = c["metadata"]["section"]
        if not seen_sections or seen_sections[-1] != sec:
            seen_sections.append(sec)
    print(f"  📑 Sections detected: {len(seen_sections)}")
    for s in seen_sections[:8]:
        print(f"       • {s[:80]}")
    if len(seen_sections) > 8:
        print(f"       … and {len(seen_sections) - 8} more")

    # Embed + store in batches on GPU
    total = 0
    for i in tqdm(range(0, len(chunks), batch_size), desc="  🔢 Embedding", unit="batch"):
        batch  = chunks[i : i + batch_size]
        texts  = [c["text"]     for c in batch]
        ids    = [c["id"]       for c in batch]
        metas  = [c["metadata"] for c in batch]

        embeds = embed_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,   # ChromaDB needs numpy/list, not tensors
            device=device,
        ).tolist()

        collection.upsert(
            ids        = ids,
            documents  = texts,
            embeddings = embeds,
            metadatas  = metas,
        )
        total += len(batch)

    elapsed = time.time() - t0
    speed   = total / elapsed
    print(f"  ✅ Done: {total:,} chunks in {elapsed:.1f}s  ({speed:.1f} chunks/s)")
    return total


def ingest_all(
    pdf_dir: str,
    collection,
    embed_model: SentenceTransformer,
    device: torch.device,
    batch_size: int,
    reset: bool = False,
):
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDFs found in '{pdf_dir}' — drop files there and re-run.")
        return

    print(f"\n🗂️  Found {len(pdf_files)} PDF(s) in '{pdf_dir}'")
    grand_total = 0
    t0 = time.time()

    for pdf_path in pdf_files:
        grand_total += ingest_pdf(
            str(pdf_path), collection, embed_model, device, batch_size, force=reset
        )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"🎉 Ingestion complete — {grand_total:,} chunks in {elapsed:.1f}s")
    print(f"📁 Vectorstore: {CHROMA_DIR}")


# ─── Utilities ────────────────────────────────────────────────────────────────

def list_documents(collection):
    results = collection.get(include=["metadatas"])
    sources = {}
    for meta in results["metadatas"]:
        src = meta.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    if not sources:
        print("📭 No documents ingested yet.")
        return
    print(f"\n📚 Ingested documents ({len(sources)} files):")
    for src, count in sorted(sources.items()):
        print(f"  • {src}  ({count:,} chunks)")


def preview_sections(pdf_path: str):
    """Dry-run section detection and print results without ingesting."""
    pdf_name = Path(pdf_path).name
    print(f"\n📂 Previewing sections: {pdf_name}")
    pages  = extract_text_from_pdf(pdf_path)
    pages  = detect_sections(pages)
    seen   = []
    for p in pages:
        entry = (p["section"], p["section_page"])
        if not seen or seen[-1] != entry:
            seen.append(entry)
    print(f"\n  Found {len(seen)} section(s):\n")
    for sec, pg in seen:
        print(f"  p.{pg:>4}  {sec[:100]}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PDF → ChromaDB ingestion pipeline (GPU-accelerated)"
    )
    parser.add_argument("pdf",       nargs="?",            help="Path to a specific PDF")
    parser.add_argument("--reset",   action="store_true",  help="Wipe vectorstore and re-ingest")
    parser.add_argument("--list",    action="store_true",  help="List ingested documents")
    parser.add_argument("--sections",action="store_true",  help="Preview detected sections (no ingest)")
    args = parser.parse_args()

    os.makedirs(PDF_DIR,    exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # --sections preview — no GPU / model needed
    if args.sections:
        targets = [args.pdf] if args.pdf else list(Path(PDF_DIR).glob("*.pdf"))
        for t in targets:
            preview_sections(str(t))
        return

    device     = get_device()
    batch_size = optimal_batch_size(device)
    print(f"⚡ Batch size: {batch_size}")

    collection  = get_chroma_collection(reset=args.reset)
    embed_model = get_embed_model(device)

    if args.list:
        list_documents(collection)
        return

    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"❌ File not found: {args.pdf}")
            sys.exit(1)
        ingest_pdf(args.pdf, collection, embed_model, device, batch_size, force=args.reset)
    else:
        ingest_all(PDF_DIR, collection, embed_model, device, batch_size, reset=args.reset)


if __name__ == "__main__":
    main()