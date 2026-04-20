"""Assert paper v0.3 integrates the 4 additional gate sections."""
from pathlib import Path


def _tex() -> str:
    return Path("papers/paper1/main.tex").read_text()


def test_paper_has_integrations_section():
    text = _tex()
    assert "\\section{Integrations}" in text


def test_paper_mentions_v1_2_claims_in_abstract():
    """Post-v0.9, the abstract foregrounds Claims A + B rather than
    enumerating all 11 gates. The 11-gate catalogue lives in §Status
    of the README and CHANGELOG; the paper's abstract is a scientific
    hook, not a release changelog."""
    text = " ".join(_tex().split())
    v1_2_tokens = [
        "Substrate-agnostic",       # title / abstract anchor
        "polymorphism",             # Claim A
        "information transmission", # Claim B
        "scaling law",              # v1.1.4 headline
        "mutual information",       # Claim B quantification
        "Transformer",              # third substrate
    ]
    for token in v1_2_tokens:
        assert token.lower() in text.lower(), (
            f"paper should mention {token!r} in the body"
        )


def test_integrations_cites_each_pilot_module():
    text = _tex()
    for module in (
        "track\\_p.adaptive\\_codebook",
        "neuromorphic.export",
        "bridge.dream\\_bridge",
        "bridge.kiki\\_nerve\\_advisor",
    ):
        assert module in text, f"§Integrations should cite {module}"
