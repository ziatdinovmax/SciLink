"""Discovery test for the CP2K periodic_dft engine skill.

CP2K is a knowledge-only bundle (cp2k.md), like the QE skill: no tools module,
so the check is that the bundle is discovered and the agent exposes it. Input
correctness is exercised by the DFT input-generation benchmark, not here.
"""


def test_cp2k_is_a_discovered_periodic_dft_skill():
    from scilink.skills.loader import list_skills
    assert "cp2k" in list_skills(domain="periodic_dft")


def test_cp2k_is_a_supported_engine():
    from scilink.agents.sim_agents.periodic_dft_agent import PeriodicDFTAgent
    assert "cp2k" in PeriodicDFTAgent.supported_software()


def test_cp2k_skill_frontmatter_is_valid_and_names_binaries():
    """The frontmatter must parse (a mid-value colon once broke it) and its
    detect block must name the cp2k binaries for availability detection."""
    import yaml
    from pathlib import Path
    import scilink
    md = (Path(scilink.__file__).parent / "skills" / "periodic_dft" / "cp2k"
          / "cp2k.md").read_text()
    assert md.startswith("---")
    fm = yaml.safe_load(md.split("---", 2)[1])          # raises if malformed
    assert "cp2k" in fm["detect"]["binaries"]
    assert "RUN_TYPE" in md and "&KIND" in md            # core CP2K input concepts


def test_cp2k_skill_loads_into_sections():
    from scilink.skills.loader import load_skill
    parsed = load_skill("cp2k", domain="periodic_dft")
    assert "implementation" in parsed and "RUN_TYPE" in parsed["implementation"]


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
