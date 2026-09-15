"""Offline tests for per-domain section vocabulary in the skill loader (D1 of
the OptimizationAgent foundationalization, issue #196).

Verifies that:
  - existing domains parse byte-identically to the legacy single-vocabulary
    behavior (no regression for curve_fitting / image_analysis / etc.);
  - the `optimization` domain recognizes its own pipeline vocabulary
    (setup / surrogate / acquisition / diagnostics) under canonical sections
    rather than demoting them to `extras`.

No network, no torch — pure parsing.
"""
import tempfile
from pathlib import Path

from scilink.skills import loader


def _parse(body, **kw):
    return loader._parse_sections(body, **kw)


def test_default_vocab_unchanged():
    """Default (no known_sections arg) still uses _KNOWN_SECTIONS exactly."""
    body = "## Overview\nfoo\n\n## Analysis\nbar\n\n## Planning\nbaz\n"
    sections, extras = _parse(body)
    assert set(sections) == loader._KNOWN_SECTIONS
    assert sections["overview"] == "foo"
    assert sections["analysis"] == "bar"
    assert sections["planning"] == "baz"
    assert extras == {}


def test_vocab_for_lookup():
    assert loader._vocab_for("curve_fitting") is loader._KNOWN_SECTIONS
    assert loader._vocab_for("image_analysis") is loader._KNOWN_SECTIONS
    assert loader._vocab_for("nonexistent") is loader._KNOWN_SECTIONS
    assert loader._vocab_for("optimization") == {
        "overview", "setup", "surrogate", "acquisition",
        "diagnostics", "interpretation", "implementation",
    }


def test_optimization_sections_recognized():
    """Optimization-vocab headings land in canonical sections, not extras."""
    body = (
        "## Overview\nmo\n\n## Setup\nframe the schema\n\n"
        "## Surrogate\nmixed kernel after categorical\n\n"
        "## Acquisition\nqNEHVI for Pareto\n\n## Diagnostics\nLOO-CV\n"
    )
    sections, extras = _parse(body, known_sections=loader._vocab_for("optimization"))
    assert sections["setup"] == "frame the schema"
    assert sections["surrogate"] == "mixed kernel after categorical"
    assert sections["acquisition"] == "qNEHVI for Pareto"
    assert sections["diagnostics"] == "LOO-CV"
    assert extras == {}


def test_optimization_section_is_extra_under_default_vocab():
    """The same headings would be demoted to extras under the default vocab --
    proving the per-domain vocabulary is what rescues them."""
    body = "## Surrogate\nmixed kernel\n\n## Acquisition\nqNEHVI\n"
    sections, extras = _parse(body)  # default vocab
    assert "surrogate" not in sections
    assert "surrogate" in extras and "acquisition" in extras


def test_load_skill_optimization_end_to_end():
    """load_skill(domain='optimization') wires the vocabulary through."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "demo.md"
        p.write_text(
            "---\ndescription: demo\n---\n"
            "## Setup\nbounds and schema\n\n## Acquisition\ncost-aware\n"
        )
        skill = loader.load_skill(str(p), domain="optimization")
        assert skill["setup"] == "bounds and schema"
        assert skill["acquisition"] == "cost-aware"
        assert skill["extras"] == {}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all passed")
