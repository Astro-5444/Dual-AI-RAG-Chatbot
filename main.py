#!/usr/bin/env python3
"""
Dual-AI RAG Chatbot — CLI Entry Point
Ties together Researcher and Chatbot with transparent terminal UI

Usage:
    python main.py           # Start interactive chat
    python main.py --reset   # Clear conversation history and start fresh
"""

import argparse
import sys
import time

from config import LOOP_BUDGET
from terminal_ui import (
    print_welcome, print_user_question, print_separator, print_divider,
    print_error, print_success, print_status, print_timing_summary,
    print_chatbot_translating,
)
from Chatbot import ConversationHistory, process_follow_up, generate_response
from Researcher import run_researcher_reasoning, collection

# ─── Global State ─────────────────────────────────────────────────────────────

history = ConversationHistory()


# ─── Main Chat Loop ───────────────────────────────────────────────────────────

def chat_loop():
    """Run the interactive chat loop."""
    print_welcome()
    
    # Check if we have documents
    chunk_count = collection.count()
    if chunk_count == 0:
        print_error("No documents indexed yet. Run `python ingest.py` first.")
        return
    
    print_status(f"Vector DB: {chunk_count:,} chunks ready", "success")
    print_status(f"Loop budget: {LOOP_BUDGET} iterations max")
    print_separator()
    
    while True:
        try:
            # Get user input
            print()
            user_input = input("❯ ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                print()
                print_status("Goodbye!", "info")
                print_separator()
                break
            
            if user_input.lower() == "clear":
                history.clear()
                print_success("Conversation history cleared")
                continue
            
            if user_input.lower() == "help":
                print_help()
                continue
            
            # Display user question
            print_user_question(user_input)

            # Start total timer
            total_start = time.perf_counter()

            # Chatbot: Extract keywords from question
            previous_chunk_ids = []
            if history.messages:
                # Follow-up — process_follow_up shows input + output headers
                keywords, previous_chunk_ids = process_follow_up(
                    user_input, history, []
                )
            else:
                # First question — extract_keywords shows the input header;
                # we show the output (keywords) manually after
                from Chatbot import extract_keywords
                keywords = extract_keywords(user_input, history)
                print_chatbot_translating(keywords, 0)

            # Researcher: Run iterative reasoning
            ai1_result = run_researcher_reasoning(user_input, keywords)

            # Chatbot: Generate conversational response (Researcher's answer is passed but not shown directly)
            ai2_time = generate_response(user_input, ai1_result, history)

            # Update history with this turn
            history.add_turn(
                user_input,
                ai1_result["answer"],
                ai1_result.get("chunk_ids", [])
            )

            # Calculate and display timing
            total_elapsed = time.perf_counter() - total_start
            ai1_time = ai1_result.get("researcher_time", total_elapsed - ai2_time)

            print_divider()
            print_timing_summary(total_elapsed, ai1_time, ai2_time)
            print_divider()
            
        except KeyboardInterrupt:
            print()
            print_status("Interrupted. Type 'exit' to quit.", "warning")
            continue
        except Exception as e:
            print_error(f"An error occurred: {e}")
            print_status("Try again or type 'exit' to quit.", "info")


def print_help():
    """Print help information."""
    print()
    print("  Commands:")
    print("    clear  - Clear conversation history")
    print("    help   - Show this help")
    print("    exit   - Quit the chatbot")
    print()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dual-AI RAG Chatbot with transparent terminal UI"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear conversation history and start fresh"
    )
    args = parser.parse_args()
    
    if args.reset:
        history.clear()
        print_success("Conversation history cleared")
    
    try:
        chat_loop()
    except Exception as e:
        print_error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
