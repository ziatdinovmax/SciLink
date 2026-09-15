"""The white-paper and technical-document prompts carry writing-style rules:
one idea per sentence, no stock rhetorical phrases, and audience-matched
vocabulary (no software jargon for natural-science readers)."""

import pytest

from scilink.agents.planning_agents.instruct import (
    TECHNICAL_DOCUMENT_INSTRUCTIONS, WHITE_PAPER_INSTRUCTIONS)


@pytest.mark.parametrize("prompt", [WHITE_PAPER_INSTRUCTIONS,
                                    TECHNICAL_DOCUMENT_INSTRUCTIONS],
                         ids=["white_paper", "technical_document"])
def test_sentence_discipline_is_stated(prompt):
    assert "one idea per sentence" in prompt.lower()
    assert "split" in prompt.lower()


@pytest.mark.parametrize("prompt", [WHITE_PAPER_INSTRUCTIONS,
                                    TECHNICAL_DOCUMENT_INSTRUCTIONS],
                         ids=["white_paper", "technical_document"])
def test_stock_phrases_are_banned(prompt):
    assert 'never "load-bearing"' in prompt
    assert 'not merely X, but Y' in prompt


@pytest.mark.parametrize("prompt", [WHITE_PAPER_INSTRUCTIONS,
                                    TECHNICAL_DOCUMENT_INSTRUCTIONS],
                         ids=["white_paper", "technical_document"])
def test_audience_vocabulary_rule_present(prompt):
    assert "software-engineering jargon" in prompt
    assert "natural scientists" in prompt


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
