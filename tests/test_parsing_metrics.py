from src.llm_judge.parsing import parse_pairwise_label, parse_score
from src.llm_judge.mitigations import invert_label, majority_vote


def test_parse_pairwise_label():
    assert parse_pairwise_label("A") == "A"
    assert parse_pairwise_label("FINAL: B") == "B"
    assert parse_pairwise_label("It is a tie.") == "TIE"
    assert parse_pairwise_label("Answer A is better") == "A"


def test_parse_score():
    assert parse_score("5") == 5.0
    assert parse_score("Score: 3.5") == 3.5
    assert parse_score("10") == 1.0  # first valid digit is 1


def test_mitigation_helpers():
    assert invert_label("A") == "B"
    assert invert_label("B") == "A"
    assert majority_vote(["A", "A", "B"]) == "A"
    assert majority_vote(["A", "B"]) == "TIE"
