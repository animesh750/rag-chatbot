import pytest

from rag.smalltalk import classify, reply


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("hi", "greeting"),
        ("Hello!", "greeting"),
        ("hellow", "greeting"),
        ("heyy there", None),
        ("good morning", "greeting"),
        ("hello there", "greeting"),
        ("how are you?", "how_are_you"),
        ("thanks", "thanks"),
        ("thank you", "thanks"),
        ("thanku", "thanks"),
        ("bye", "bye"),
        ("who are you", "identity"),
        ("what can you do", "help"),
        ("help", "help"),
    ],
)
def test_small_talk_is_recognised(text, kind):
    assert classify(text) == kind


@pytest.mark.parametrize(
    "text",
    [
        "his skills",
        "what are his projects",
        "help me summarise the document",
        "high school marks",
        "thanks for the project list and the skills too",
        "How many parameters does GPT3 have?",
        "hire date",
        "Summarise this document",
        "",
        "   ",
    ],
)
def test_real_questions_are_never_treated_as_small_talk(text):
    assert classify(text) is None


def test_replies_adapt_to_whether_documents_exist():
    assert "Upload a PDF" in reply("greeting", 0)
    assert "2 documents" in reply("greeting", 2) and "1 document." in reply("greeting", 1)
