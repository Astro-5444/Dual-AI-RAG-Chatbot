"""
Chatbot Module — Fast model for conversation and keyword translation
Holds conversation history and translates follow-ups into search keywords
"""

import json
import httpx
import re
import time
from typing import Optional

from config import API_BASE_URL, CHATBOT_MODEL, CHATBOT_MAX_TOKENS, CHATBOT_TEMPERATURE
from terminal_ui import (
    console,
    print_chatbot_input_keyword,
    print_chatbot_translating,
    print_chatbot_response_input,
    print_chatbot_response_footer,
    COLORAccent2,
    COLOR_TEXT,
)

# ─── Conversation History ─────────────────────────────────────────────────────

class ConversationHistory:
    """Manages conversation history for Chatbot."""
    
    def __init__(self):
        self.messages = []
        self.questions = []
        self.answers = []
    
    def add_turn(self, question: str, answer: str, chunk_ids: list[str]):
        """Add a Q&A turn to history."""
        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": answer})
        self.questions.append(question)
        self.answers.append(answer)
    
    def get_recent(self, n: int = 6) -> list[dict]:
        """Get last n messages (n/2 turns)."""
        return self.messages[-n:] if len(self.messages) > n else self.messages
    
    def get_all_questions(self) -> list[str]:
        """Get all previous questions."""
        return self.questions
    
    def clear(self):
        """Clear history."""
        self.messages = []
        self.questions = []
        self.answers = []


# ─── Keyword Extraction ───────────────────────────────────────────────────────

CHATBOT_KEYWORD_PROMPT = """
You are Chatbot, a conversation assistant that translates user questions into search keywords.

YOUR TASK:
Given the user's question and conversation history, extract 3-6 key search terms.

RULES:
- Focus on nouns, technical terms, and specific concepts
- Include synonyms and related terms that might appear in documents
- For follow-up questions, include context from previous turns
- Output ONLY a JSON array of keywords

EXAMPLE:
User: "What are the side effects of ibuprofen?"
Output: ["ibuprofen", "side effects", "adverse reactions", "NSAIDs"]

User: "What about cardiovascular risks?"
Output: ["cardiovascular", "heart", "risks", "ibuprofen", "NSAIDs"]
""".strip()


def extract_keywords(question: str, history: ConversationHistory) -> list[str]:
    """Extract search keywords from a user question."""
    # Show what Chatbot is receiving
    print_chatbot_input_keyword(question, len(history.get_all_questions()))
    recent_history = history.get_recent(6)

    messages = [
        {"role": "system", "content": CHATBOT_KEYWORD_PROMPT},
    ]

    # Add context from recent history
    if recent_history:
        context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history[-4:]])
        messages.append({"role": "user", "content": f"Conversation history:\n{context}"})

    messages.append({"role": "user", "content": f"Extract keywords for: {question}"})

    try:
        payload = {
            "model": CHATBOT_MODEL,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False,
        }

        with httpx.Client(timeout=30) as client:
            resp = client.post(f"{API_BASE_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON array from response
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                keywords = json.loads(json_match.group())
                if isinstance(keywords, list) and keywords:
                    return keywords
    except Exception:
        pass  # Fall through to fallback

    # Fallback: simple keyword extraction from raw question
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
    stop_words = {'what', 'are', 'the', 'and', 'for', 'how', 'does', 'did', 'about', 'into',
                  'with', 'from', 'that', 'this', 'have', 'has', 'was', 'were', 'been', 'being'}
    keywords = [w for w in words if w not in stop_words][:6]

    # Ensure we never return empty - use all significant words
    if not keywords:
        keywords = [w for w in words if len(w) > 2][:6]

    # Last resort: use the whole question as a single keyword
    if not keywords:
        keywords = [question.strip()[:100]]

    return keywords


# ─── Chatbot Response Streaming ──────────────────────────────────────────────────

CHATBOT_RESPONSE_PROMPT = """
You are Chatbot, a friendly conversation assistant.

YOUR TASK:
Provide a natural, conversational response based on the answer from Researcher.

RULES:
- Researcher has already analyzed the documents and provided an answer
- Your job is to present it conversationally
- You only see Researcher's answer, NOT the raw chunks
- Add context from conversation history when relevant
- Be concise but complete
- If Researcher expressed uncertainty, reflect that appropriately

Researcher's ANSWER:
{ai1_answer}

CONFIDENCE: {confidence}
""".strip()


def stream_chatbot_response(
    question: str,
    ai1_answer: str,
    confidence: str,
    history: ConversationHistory
):
    """Stream Chatbot's conversational response token by token."""
    import sys
    
    recent_history = history.get_recent(8)

    messages = [
        {"role": "system", "content": CHATBOT_RESPONSE_PROMPT.format(
            ai1_answer=ai1_answer[:2000] + "..." if len(ai1_answer) > 2000 else ai1_answer,
            confidence=confidence
        )},
    ]

    # Add conversation history
    if recent_history:
        for msg in recent_history:
            messages.append(msg)

    messages.append({"role": "user", "content": question})

    payload = {
        "model": CHATBOT_MODEL,
        "messages": messages,
        "max_tokens": CHATBOT_MAX_TOKENS,
        "temperature": CHATBOT_TEMPERATURE,
        "stream": True,
    }

    try:
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
                            token = delta.get("content", "")
                            if token:
                                # Print token immediately with flush
                                console.print(token, end="", style="bright_white", highlight=False)
                                console.file.flush()
                                sys.stdout.flush()
                        except Exception:
                            continue
    except Exception as e:
        console.print(f"\n[Error: {e}]", style="red")


# ─── Main Chatbot Interface ──────────────────────────────────────────────────────

def process_follow_up(
    question: str,
    history: ConversationHistory,
    chunk_ids_from_researcher: list[str]
) -> tuple[list[str], list[str]]:
    """
    Process a follow-up question: extract keywords and return previously relevant chunk IDs.
    Returns: (keywords, previous_chunk_ids_to_exclude)
    """
    # extract_keywords already shows the input header; show result after
    keywords = extract_keywords(question, history)
    print_chatbot_translating(keywords, len(chunk_ids_from_researcher))
    return keywords, chunk_ids_from_researcher


def generate_response(
    question: str,
    ai1_result: dict,
    history: ConversationHistory
) -> float:
    """Generate and stream Chatbot's conversational response. Returns Chatbot time in seconds."""
    answer = ai1_result.get("answer", "")
    confidence = ai1_result.get("confidence", "medium")

    # Show what Chatbot receives as input
    print_chatbot_response_input(
        question=question,
        ai1_answer=answer,
        confidence=confidence,
        history_turns=len(history.get_all_questions()),
    )

    # Handle empty or null answer gracefully
    if not answer or answer.strip() == "" or "not found" in answer.lower():
        console.print("  I wasn't able to find that specific information in the available documents.", style="yellow")
        console.print("  Could you rephrase the question or check if the relevant document has been ingested?", style="yellow")
        console.print()
        return 0.0

    chatbot_start = time.perf_counter()

    # Stream response directly
    stream_chatbot_response(question, answer, confidence, history)

    chatbot_elapsed = time.perf_counter() - chatbot_start
    print_chatbot_response_footer(chatbot_elapsed)
    return chatbot_elapsed
