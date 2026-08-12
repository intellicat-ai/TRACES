from __future__ import annotations

from traces.influence.linguistic import LinguisticDetector
from traces.influence.matching import build_nlp


def test_positive_responsible_sentence_is_not_a_refusal() -> None:
    nlp = build_nlp()
    detector = LinguisticDetector(nlp)
    doc = nlp("It is responsible to design a follow-up study to test the hypothesis.")
    matches = detector.detect(doc, probe_authors=set(), debug=True)

    assert matches["R1_refusal_with_reason"] == []
    assert matches["R2_refusal_no_reason"] == []


def test_negated_think_responsible_without_study_is_not_a_refusal() -> None:
    nlp = build_nlp()
    detector = LinguisticDetector(nlp)
    doc = nlp("I don't think it would be responsible for me to do that.")
    matches = detector.detect(doc, probe_authors=set(), debug=True)

    assert matches["R1_refusal_with_reason"] == []
    assert matches["R2_refusal_no_reason"] == []
