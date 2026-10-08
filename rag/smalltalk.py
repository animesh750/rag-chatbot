"""Answer greetings and other small talk directly, without searching the documents.

Rule-based on purpose: instant, free, never fails, and short enough to audit. It only fires
on very short messages, so real questions ("his skills", "help me summarise this") are
never mistaken for chit-chat.
"""

from __future__ import annotations

import difflib
import re

_GREETINGS = {"hi", "hii", "hiii", "hello", "hellow", "helo", "hey", "heya", "hlo", "hola", "yo", "namaste"}
_GREETING_PHRASES = ("good morning", "good afternoon", "good evening", "good night")
_THANKS = {"thanks", "thank", "thx", "ty", "thanku", "shukriya", "dhanyavad"}
_BYE = {"bye", "goodbye", "cya", "tata"}
_HOW_ARE_YOU = {"how are you", "how are you doing", "how r u", "how are u", "whats up", "what's up", "sup"}
_IDENTITY = {"who are you", "what are you", "what is your name", "whats your name", "what's your name"}
_HELP = {"help", "what can you do", "how do you work", "how can you help", "what can i ask", "how to use"}


def _fuzzy(word: str, vocab: set[str]) -> bool:
    if word in vocab:
        return True
    # typo tolerance only for longer words, so real words like "his" never match "hi"
    return len(word) >= 5 and bool(difflib.get_close_matches(word, vocab, n=1, cutoff=0.85))


def classify(text: str) -> str | None:
    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > 5:
        return None
    joined = " ".join(words)
    if joined in _HOW_ARE_YOU:
        return "how_are_you"
    if joined in _IDENTITY:
        return "identity"
    if joined in _HELP:
        return "help"
    if any(joined.startswith(p) for p in _GREETING_PHRASES):
        return "greeting"
    if len(words) <= 3:
        if _fuzzy(words[0], _GREETINGS):
            return "greeting"
        if _fuzzy(words[0], _THANKS):
            return "thanks"
        if _fuzzy(words[0], _BYE):
            return "bye"
    return None


def reply(kind: str, n_docs: int) -> str:
    ready = (
        f"I'm ready to answer questions about your {n_docs} document{'s' if n_docs != 1 else ''}. "
        "What would you like to know?"
        if n_docs
        else "Upload a PDF in the sidebar and I'll answer questions about it."
    )
    return {
        "greeting": f"Hi there! 👋 {ready}",
        "how_are_you": f"I'm doing well, thanks for asking! {ready}",
        "thanks": "You're welcome! Ask me anything else about your documents.",
        "bye": "Goodbye! Come back any time. 👋",
        "identity": (
            "I'm a document Q&A assistant. I search your uploaded PDFs with keyword and semantic "
            "search, then answer with citations showing the file and page."
        ),
        "help": (
            "Upload one or more PDFs, then ask questions like “What projects are listed?” or "
            "“Summarise page 2”. Each answer shows its sources, and you can change the search mode "
            "in the settings."
        ),
    }[kind]
