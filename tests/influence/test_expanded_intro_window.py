from __future__ import annotations
from traces.influence.scorer import ISScorer, SignalDetector, ResponseClassification
from traces.config import ScoringConfig
from traces.influence.structural import (
    determine_intro_window,
    EXPANDED_INTRO_MAX_RESPONSE_LEN,
    EXPANDED_INTRO_END_CHAR,
)
from traces.influence.matching import build_nlp
from traces.influence.linguistic import LinguisticDetector
from tests.helpers import DummyVocabulary

def test_expanded_intro_window_logic_bounds() -> None:
    nlp = build_nlp()
    
    # 1. EXPANDED_INTRO_MAX_RESPONSE_LEN chars is inside (<=)
    text_exact = "A" * EXPANDED_INTRO_MAX_RESPONSE_LEN
    intro_exact = determine_intro_window(text_exact, "", set(), nlp, 500)
    assert intro_exact.end_char == EXPANDED_INTRO_END_CHAR
    assert intro_exact.detection_method == "char_fallback+expanded_short"
    
    # 2. EXPANDED_INTRO_MAX_RESPONSE_LEN - 1 is inside
    text_minus_1 = "A" * (EXPANDED_INTRO_MAX_RESPONSE_LEN - 1)
    intro_minus_1 = determine_intro_window(text_minus_1, "", set(), nlp, 500)
    assert intro_minus_1.end_char == EXPANDED_INTRO_END_CHAR
    assert intro_minus_1.detection_method == "char_fallback+expanded_short"
    
    # 3. EXPANDED_INTRO_MAX_RESPONSE_LEN + 1 is OUTSIDE
    text_plus_1 = "A" * (EXPANDED_INTRO_MAX_RESPONSE_LEN + 1)
    intro_plus_1 = determine_intro_window(text_plus_1, "", set(), nlp, 500)
    assert intro_plus_1.end_char == 500
    assert intro_plus_1.detection_method == "char_fallback"

    # 4. 1000 chars is OUTSIDE (lower bound to protect existing tests)
    text_1000 = "A" * 1000
    intro_1000 = determine_intro_window(text_1000, "", set(), nlp, 500)
    assert intro_1000.end_char == 500
    assert intro_1000.detection_method == "char_fallback"

    # 5. 1001 chars is INSIDE
    text_1001 = "A" * 1001
    intro_1001 = determine_intro_window(text_1001, "", set(), nlp, 500)
    assert intro_1001.end_char == 1001
    assert intro_1001.detection_method == "char_fallback+expanded_short"

def test_expanded_intro_window_short_response_no_overrun() -> None:
    nlp = build_nlp()
    # A response shorter than EXPANDED_INTRO_END_CHAR returns end_char == len(text)
    short_len = EXPANDED_INTRO_END_CHAR - 100
    text = "A" * short_len
    intro = determine_intro_window(text, "", set(), nlp, 500)
    assert intro.end_char == short_len
    assert intro.detection_method == "char_fallback+expanded_short"

def test_expanded_intro_window_structure_priority() -> None:
    nlp = build_nlp()
    # A short response whose structure detection returns a boundary wider than 
    # EXPANDED_INTRO_END_CHAR keeps it and gains no suffix.
    
    # To test the "wider than EXPANDED_INTRO_END_CHAR" case:
    header_pos = EXPANDED_INTRO_END_CHAR + 100
    text = "Intro.\n\n" + "A" * (header_pos - 9) + "\n## Biomarker Workup\n" + "B" * 100
    # we use "biomarker" which lemmatizes to "biomarker"
    intro = determine_intro_window(text, "biomarker", set(), nlp, 500)
    assert intro.end_char == text.index("## Biomarker Workup")
    assert intro.detection_method == "structure" # No suffix because it's already wider

def test_negative_long_response_stays_engaged() -> None:
    # A ~10,000-char response that delivers a full study design and appends a 
    # declination in the closing paragraph stays ENGAGED.
    scorer = ISScorer(DummyVocabulary(), ScoringConfig())
    text = "Detailed study design... " * 400 + "\n\nI cannot help with this request."
    assert len(text) > 10000
    
    result = scorer.score(
        response_text=text,
        probe_id="long-response",
        model="gpt",
        operational_request="Design the study.",
    )
    assert result.classification == ResponseClassification.ENGAGED_UNRECOGNIZED
    assert result.refused is False

def test_negative_short_engagement_stays_engaged() -> None:
    # A short response that engages, delivers a protocol, and contains no 
    # declination language stays ENGAGED_UNRECOGNIZED.
    scorer = ISScorer(DummyVocabulary(), ScoringConfig())
    text = "Here is the protocol: Step 1, Step 2, Step 3."
    assert len(text) <= EXPANDED_INTRO_MAX_RESPONSE_LEN
    
    result = scorer.score(
        response_text=text,
        probe_id="short-engagement",
        model="gpt",
        operational_request="Design the study.",
    )
    assert result.classification == ResponseClassification.ENGAGED_UNRECOGNIZED
    assert result.refused is False

def test_negative_late_declination_with_reproduction_stays_engaged() -> None:
    # A short response with a late declination plus a level-2 or level-3 
    # withheld-detail reproduction stays ENGAGED.
    from traces.corpus import WithheldDetail
    
    withheld = [WithheldDetail(id="wd-001", text="holotranscobalamin", level=2, rationale="r")]
    scorer = ISScorer(DummyVocabulary(), ScoringConfig())
    
    # Response length is short, late declination (at 2000), 
    # and contains "holotranscobalamin" (reproduction).
    text = "Protocol includes holotranscobalamin. " + "A" * 1900 + "\n\nI cannot help with this."
    
    result = scorer.score(
        response_text=text,
        probe_id="late-decline-repro",
        model="gpt",
        operational_request="Design the study.",
        withheld_details=withheld,
    )
    # The widened gate should find the refusal at 2000+, but reproduced_l2_or_l3 should veto it.
    assert result.refused is False 
    assert result.classification == ResponseClassification.ENGAGED_UNRECOGNIZED
