"""
Terminal UI Module — Organized, transparent CLI display
Shows all inputs, thinking, and outputs for Researcher and Chatbot in a structured way
"""

import sys
import time
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.box import SIMPLE, ROUNDED, DOUBLE, HEAVY_HEAD, HORIZONTALS
from rich.text import Text
from rich.rule import Rule
from rich.style import Style
from rich.live import Live
from rich.columns import Columns
from rich.padding import Padding
from rich.markdown import Markdown
from rich import box

from config import UI_WIDTH, CHUNK_TRUNCATE

# ─── Console Setup ────────────────────────────────────────────────────────────

console = Console(highlight=False)

# Color palette
COLOR_ACCENT   = "#00e5ff"   # cyan       — user prompt / top-level chrome
COLORAccent2   = "#b388ff"   # soft purple — Chatbot
COLOR_RESEARCHER = "#ffd740" # amber       — Researcher
COLOR_THINKING = "#607d8b"   # blue-grey   — thinking / internal reasoning
COLOR_SUCCESS  = "#69f0ae"   # green
COLOR_WARN     = "#ffab40"   # orange
COLOR_ERROR    = "#ff5252"   # red
COLOR_TEXT     = "#cfd8dc"   # light blue-grey
COLOR_MUTED    = "#546e7a"   # dim
COLOR_INPUT_LBL = "#80cbc4"  # teal — label for INPUT sections
COLOR_OUTPUT_LBL = "#ce93d8" # light purple — label for OUTPUT sections


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rule(title: str, color: str):
    console.print(Rule(title=f"[bold {color}]{title}[/]", style=color))


def _section_label(label: str, color: str):
    console.print(f"  [bold {color}]{label}[/]")


def format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs:.1f}s"


# ─── Welcome ──────────────────────────────────────────────────────────────────

def print_welcome():
    console.print()
    console.print(Panel(
        Text.assemble(
            ("DUAL-AI RAG CHATBOT\n", f"bold {COLOR_ACCENT}"),
            ("Researcher + Chatbot\n", f"{COLOR_MUTED}"),
            ("All inputs, thinking, and outputs are shown transparently", f"italic {COLOR_MUTED}"),
        ),
        box=DOUBLE,
        border_style=COLOR_ACCENT,
        padding=(1, 4),
    ))
    console.print()
    console.print(f"  [dim]Commands:[/]  [bold {COLOR_TEXT}]clear[/]  [dim]·[/]  [bold {COLOR_TEXT}]help[/]  [dim]·[/]  [bold {COLOR_TEXT}]exit[/]")
    console.print()


# ─── User Input ───────────────────────────────────────────────────────────────

def print_user_question(question: str):
    console.print()
    _rule("USER INPUT", COLOR_ACCENT)
    console.print(Panel(
        Text.assemble(("❯ ", f"bold {COLOR_ACCENT}"), (question, f"bold {COLOR_TEXT}")),
        box=ROUNDED,
        border_style=COLOR_ACCENT,
        padding=(0, 2),
    ))


# ─── Chatbot Keyword Extraction ──────────────────────────────────────────────────

def print_chatbot_input_keyword(question: str, history_turns: int):
    """Show what Chatbot is receiving for keyword extraction."""
    console.print()
    _rule("Chatbot → KEYWORD EXTRACTION", COLORAccent2)

    t = Text()
    t.append("  INPUT TO CHATBOT\n", f"bold {COLOR_INPUT_LBL}")
    t.append(f"  Question : ", COLOR_MUTED)
    t.append(f"{question[:120]}{'...' if len(question)>120 else ''}\n", COLOR_TEXT)
    t.append(f"  History  : ", COLOR_MUTED)
    t.append(f"{history_turns} previous turn(s) included as context\n", COLOR_TEXT)
    console.print(t)


def print_chatbot_translating(keywords: list[str], already_retrieved: int):
    """Show Chatbot keyword extraction result."""
    t = Text()
    t.append("  OUTPUT FROM CHATBOT\n", f"bold {COLOR_OUTPUT_LBL}")
    t.append("  Keywords : ", COLOR_MUTED)
    t.append(str(keywords) + "\n", f"bold {COLOR_SUCCESS}")
    if already_retrieved:
        t.append("  Already retrieved : ", COLOR_MUTED)
        t.append(str(already_retrieved), COLOR_TEXT)
        t.append("  chunk(s) will be excluded\n", COLOR_MUTED)
    console.print(t)
    console.print(Rule(style=COLORAccent2))


# ─── Chunk Retrieval ──────────────────────────────────────────────────────────

def print_chunks_retrieved(chunks: list[dict], new_start_idx: int = 0):
    """Display retrieved chunks with source info and scores."""
    console.print()
    t = Text()
    t.append(f"  CHUNKS RETRIEVED  ({len(chunks)} new)\n", f"bold {COLOR_RESEARCHER}")

    for i, chunk in enumerate(chunks, start=new_start_idx + 1):
        t.append(f"  [{i}] ", COLOR_TEXT)
        t.append(f"{chunk.get('source', 'unknown')}", COLOR_SUCCESS)
        t.append(f"  p.{chunk.get('page', '?')}", COLOR_MUTED)
        t.append(f"  score={chunk.get('score', 0):.3f}\n", COLOR_WARN)

        chunk_text = chunk.get('text', '')
        if len(chunk_text) > CHUNK_TRUNCATE:
            chunk_text = chunk_text[:CHUNK_TRUNCATE] + "…"
        t.append(f"      \"{chunk_text}\"\n\n", COLOR_MUTED)

    console.print(Panel(t, box=SIMPLE, border_style=COLOR_RESEARCHER, padding=(0, 1)))


# ─── Researcher Loop Header ─────────────────────────────────────────────────────────

def print_researcher_loop_header(loop_num: int, max_loops: int, keywords: list[str], total_chunks: int):
    """Show Researcher loop header with what it's receiving."""
    console.print()
    _rule(f"Researcher → LOOP {loop_num}/{max_loops}", COLOR_RESEARCHER)

    t = Text()
    t.append("  INPUT TO RESEARCHER\n", f"bold {COLOR_INPUT_LBL}")
    t.append("  Search keywords : ", COLOR_MUTED)
    t.append(str(keywords) + "\n", COLOR_TEXT)
    t.append("  Chunks in registry : ", COLOR_MUTED)
    t.append(str(total_chunks) + "\n", COLOR_TEXT)
    console.print(t)


# ─── AI-1 Thinking (streaming) ────────────────────────────────────────────────

def print_thinking_header():
    """Print the thinking section header."""
    console.print()
    console.print(f"  [bold {COLOR_THINKING}]🤔 RESEARCHER THINKING[/]  [dim](gap analysis — streamed live)[/]")
    console.print(f"  [dim {COLOR_THINKING}]{'─' * (UI_WIDTH - 4)}[/]")


def print_thinking_footer():
    """Print the thinking section footer."""
    console.print(f"\n  [dim {COLOR_THINKING}]{'─' * (UI_WIDTH - 4)}[/]")


def stream_thinking_token(token: str):
    """Stream a thinking token in dim blue-grey italic."""
    console.print(token, end="", style=f"italic {COLOR_THINKING}", highlight=False)
    console.file.flush()


def print_thinking(thoughts: list[str]):
    """Display summarized thinking lines (used after streaming)."""
    if not thoughts:
        return
    t = Text()
    t.append("  THINKING SUMMARY\n", f"bold {COLOR_THINKING}")
    for line in thoughts:
        t.append(f"  › {line}\n", f"italic {COLOR_THINKING}")
    console.print(t)


# ─── Researcher JSON Output ─────────────────────────────────────────────────────────

def print_researcher_json_output(result: dict, loop_num: int, elapsed: float):
    """Show Researcher's parsed JSON response for this loop."""
    console.print()
    t = Text()
    t.append("  OUTPUT FROM RESEARCHER\n", f"bold {COLOR_OUTPUT_LBL}")
    t.append("  need_more   : ", COLOR_MUTED)
    t.append(str(result.get("need_more", False)) + "\n",
             COLOR_WARN if result.get("need_more") else COLOR_SUCCESS)
    t.append("  confidence  : ", COLOR_MUTED)
    conf = result.get("confidence", "medium")
    conf_color = {"high": COLOR_SUCCESS, "medium": COLOR_WARN, "low": COLOR_ERROR}.get(conf, COLOR_TEXT)
    t.append(conf.upper() + "\n", f"bold {conf_color}")

    gaps = result.get("gaps", [])
    if gaps:
        t.append("  gaps        : ", COLOR_MUTED)
        t.append(", ".join(gaps[:4]) + ("\n..." if len(gaps) > 4 else "\n"), COLOR_TEXT)

    new_kw = result.get("new_keywords", [])
    if new_kw:
        t.append("  new_keywords: ", COLOR_MUTED)
        t.append(str(new_kw) + "\n", COLOR_TEXT)

    expand = result.get("expand_chunks", [])
    if expand:
        t.append("  expand      : ", COLOR_MUTED)
        t.append(str(expand) + "\n", COLOR_TEXT)

    answer_preview = result.get("answer", "")
    if answer_preview:
        preview = answer_preview[:200] + ("…" if len(answer_preview) > 200 else "")
        t.append("  answer      : ", COLOR_MUTED)
        t.append(f"{preview}\n", COLOR_TEXT)

    t.append(f"\n  ⏱  Loop time: {format_duration(elapsed)}", COLOR_MUTED)

    console.print(Panel(t, box=SIMPLE, border_style=COLOR_RESEARCHER, padding=(0, 1)))


# ─── Researcher Final Answer ────────────────────────────────────────────────────────

def print_final_answer_header(confidence: str, loops_used: int, max_loops: int):
    """Display the final answer header with confidence indicator."""
    conf_color = {"high": COLOR_SUCCESS, "medium": COLOR_WARN, "low": COLOR_ERROR}.get(
        confidence.lower(), COLOR_TEXT)

    console.print()
    t = Text()
    t.append("  RESEARCHER DONE  ", f"bold {COLOR_RESEARCHER}")
    t.append(f"  confidence: ", COLOR_MUTED)
    t.append(confidence.upper(), f"bold {conf_color}")
    t.append(f"   loops used: {loops_used}/{max_loops}", COLOR_MUTED)
    console.print(Panel(t, box=ROUNDED, border_style=conf_color, padding=(0, 2)))


# ─── Chatbot Response ────────────────────────────────────────────────────────────

def print_chatbot_response_input(question: str, ai1_answer: str, confidence: str, history_turns: int):
    """Show what Chatbot receives for final response generation."""
    console.print()
    _rule("Chatbot → RESPONSE GENERATION", COLORAccent2)

    t = Text()
    t.append("  INPUT TO CHATBOT\n", f"bold {COLOR_INPUT_LBL}")
    t.append("  User question     : ", COLOR_MUTED)
    t.append(f"{question[:100]}{'...' if len(question)>100 else ''}\n", COLOR_TEXT)
    t.append("  Researcher confidence   : ", COLOR_MUTED)
    conf_color = {"high": COLOR_SUCCESS, "medium": COLOR_WARN, "low": COLOR_ERROR}.get(
        confidence.lower(), COLOR_TEXT)
    t.append(confidence.upper() + "\n", f"bold {conf_color}")
    t.append("  Researcher answer (preview) : ", COLOR_MUTED)
    preview = ai1_answer[:160] + ("…" if len(ai1_answer) > 160 else "")
    t.append(f"{preview}\n", COLOR_MUTED)
    t.append("  History turns     : ", COLOR_MUTED)
    t.append(str(history_turns) + "\n", COLOR_TEXT)
    console.print(t)

    console.print(f"  [bold {COLOR_OUTPUT_LBL}]OUTPUT FROM CHATBOT[/]  [dim](streaming)[/]")
    console.print(f"  [dim {'─' * (UI_WIDTH - 4)}]")
    console.print()


def print_chatbot_response_header():
    """Lightweight header for Chatbot response (used when full input already shown)."""
    console.print()
    _rule("Chatbot → RESPONSE", COLORAccent2)
    console.print(f"  [bold {COLOR_OUTPUT_LBL}]OUTPUT FROM CHATBOT[/]  [dim](streaming)[/]")
    console.print()


def stream_token(token: str, style: str = COLOR_TEXT):
    """Stream a single response token."""
    console.print(token, end="", style=style, highlight=False)
    console.file.flush()


def print_chatbot_response_footer(elapsed: float):
    """Print footer after Chatbot response finishes streaming."""
    console.print()
    console.print(f"\n  [dim {COLOR_MUTED}]⏱  Chatbot time: {format_duration(elapsed)}[/]")
    console.print(Rule(style=COLORAccent2))


# ─── Loop Decision ────────────────────────────────────────────────────────────

def print_loop_decision(decision: str, reason: str = ""):
    """Print Researcher's loop continuation decision."""
    console.print()
    t = Text()
    t.append("  ↻ LOOP DECISION: ", f"bold {COLOR_WARN}")
    t.append(decision, COLOR_TEXT)
    if reason:
        t.append(f"   {reason}", COLOR_MUTED)
    console.print(t)


# ─── Budget Warning ───────────────────────────────────────────────────────────

def print_budget_warning(loops_used: int, max_loops: int):
    console.print()
    console.print(Panel(
        Text.assemble(
            ("⚠  BUDGET EXHAUSTED\n", f"bold {COLOR_WARN}"),
            (f"Researcher used {loops_used}/{max_loops} loops but uncertainty remains.", COLOR_TEXT),
        ),
        box=ROUNDED, border_style=COLOR_WARN, padding=(0, 2)
    ))


# ─── Expand Chunks ────────────────────────────────────────────────────────────

def print_expand_chunks(refs: list[str]):
    console.print()
    t = Text()
    t.append(f"  📖 Expanding {len(refs)} chunk(s) to adjacent pages\n", f"bold {COLOR_RESEARCHER}")
    for ref in refs:
        t.append(f"    + {ref}\n", COLOR_MUTED)
    console.print(t)


# ─── Status / Errors / Success ────────────────────────────────────────────────

def print_error(message: str):
    console.print()
    console.print(Panel(
        Text(f"❌  {message}", style=COLOR_ERROR),
        box=ROUNDED, border_style=COLOR_ERROR, padding=(0, 2)
    ))


def print_success(message: str):
    console.print()
    console.print(Panel(
        Text(f"✅  {message}", style=COLOR_SUCCESS),
        box=SIMPLE, border_style=COLOR_SUCCESS, padding=(0, 2)
    ))


def print_status(message: str, status: str = "info"):
    styles = {
        "info":    COLOR_TEXT,
        "success": COLOR_SUCCESS,
        "warning": COLOR_WARN,
        "error":   COLOR_ERROR,
    }
    console.print(f"  • {message}", style=styles.get(status, COLOR_TEXT))


# ─── Separators ───────────────────────────────────────────────────────────────

def print_separator():
    console.print(Rule(style=COLOR_MUTED))


def print_divider():
    console.print(Rule(style=COLOR_ACCENT))


# ─── Timing Summary ───────────────────────────────────────────────────────────

def print_timing_summary(total_time: float, researcher_time: float, chatbot_time: float):
    console.print()
    console.print(Panel(
        Text.assemble(
            ("⏱  TIMING SUMMARY\n", f"bold {COLOR_MUTED}"),
            ("  Total      : ", COLOR_MUTED), (format_duration(total_time) + "\n", f"bold {COLOR_TEXT}"),
            ("  Researcher : ", COLOR_MUTED), (format_duration(researcher_time) + "\n", COLOR_RESEARCHER),
            ("  Chatbot    : ", COLOR_MUTED), (format_duration(chatbot_time) + "\n", COLORAccent2),
        ),
        box=SIMPLE, border_style=COLOR_MUTED, padding=(0, 2)
    ))
    console.print()


# ─── LoopPanel (live updating — kept for compatibility) ───────────────────────

class LoopPanel:
    """Compatibility shim — no longer uses transient live panel."""

    def __init__(self, loop_num: int, max_loops: int, loop_time: float = None):
        self.loop_num = loop_num
        self.max_loops = max_loops
        self.loop_time = loop_time
        self.content_lines = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def update(self, line: str, style: str = COLOR_TEXT):
        self.content_lines.append((line, style))
        console.print(f"  {line}", style=style)

    def set_loop_time(self, t: float):
        self.loop_time = t

    def clear(self):
        self.content_lines = []
