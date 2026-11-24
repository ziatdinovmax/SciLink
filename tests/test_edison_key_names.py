"""FutureHouse became Edison Scientific. The key is read under the name
Edison's own client uses, the earlier name still works everywhere it did,
and no reader of the key names a variable of its own."""
import argparse
import re
import subprocess
from pathlib import Path

import pytest

from scilink import auth

REPO = Path(__file__).resolve().parent.parent
NEW, OLD = "EDISON_PLATFORM_API_KEY", "FUTUREHOUSE_API_KEY"


@pytest.fixture
def env(monkeypatch):
    for name in (NEW, OLD):
        monkeypatch.delenv(name, raising=False)
    for service in ("edison", "futurehouse"):
        auth.clear_api_key(service)
    yield monkeypatch
    for service in ("edison", "futurehouse"):
        auth.clear_api_key(service)


def test_the_new_variable_is_read_first_and_the_earlier_one_still_works(env):
    assert auth.edison_api_key() is None
    env.setenv(OLD, "old-key")
    assert auth.edison_api_key() == "old-key"
    assert auth.find_env_var("edison") == (OLD, "old-key")
    env.setenv(NEW, "new-key")
    assert auth.edison_api_key() == "new-key"
    assert auth.find_env_var("edison") == (NEW, "new-key")
    assert auth.find_env_var("futurehouse") == (NEW, "new-key")
    env.setenv(NEW, "")                      # an empty variable (the Dockerfile's default) is no key
    assert auth.edison_api_key() == "old-key"


def test_the_service_answers_to_both_names(env):
    auth.set_api_key("futurehouse", "k1")
    assert auth.get_api_key("edison") == "k1" and auth.get_api_key("FutureHouse") == "k1"
    auth.clear_api_key("edison")
    assert auth.get_api_key("futurehouse") is None
    env.setenv(NEW, "from-env")
    assert auth.get_api_key("futurehouse") == "from-env" == auth.get_api_key("edison")
    message = str(auth.APIKeyNotFoundError("futurehouse"))
    assert NEW in message and "edisonscientific.com" in message


def test_a_key_set_in_code_reaches_every_reader(env):
    """The reader looked at the environment only, so
    ``scilink.set_api_key('edison', ...)``, which the missing-key message
    recommends, changed nothing for any agent."""
    from scilink.agents.lit_agents import literature_agent as la, molecules_agent as ma
    seen = []

    class Client:
        def __init__(self, api_key):
            seen.append(api_key)

    env.setattr(la, "EdisonClient", Client)
    env.setattr(ma, "EdisonClient", Client)
    with pytest.raises(ValueError):
        la.OwlLiteratureAgent()
    auth.set_api_key("edison", "set-in-code")
    assert auth.edison_api_key() == "set-in-code"
    la.OwlLiteratureAgent()
    ma.MoleculesAgent()
    env.setenv(NEW, "from-env")                       # what was set in code wins over the environment
    la.OwlLiteratureAgent()
    auth.clear_api_key("edison")
    auth.set_api_key("futurehouse", "by-the-old-name")
    la.OwlLiteratureAgent()
    auth.clear_api_key("futurehouse")
    la.OwlLiteratureAgent()
    assert seen == ["set-in-code", "set-in-code", "set-in-code", "by-the-old-name", "from-env"]
    assert "futurehouse" not in auth.ENV_VARS         # one list of variables, under one name


def test_the_literature_agents_take_the_key_under_either_name(env):
    from scilink.agents.lit_agents import literature_agent as la, molecules_agent as ma
    seen = []

    class Client:
        def __init__(self, api_key):
            seen.append(api_key)

    env.setattr(la, "EdisonClient", Client)
    env.setattr(ma, "EdisonClient", Client)
    with pytest.raises(ValueError, match=NEW):
        la.OwlLiteratureAgent()
    env.setenv(OLD, "old-key")
    la.OwlLiteratureAgent()
    la.FittingModelLiteratureAgent()
    env.setenv(NEW, "new-key")
    la.OwlLiteratureAgent()
    ma.MoleculesAgent()
    assert seen == ["old-key", "old-key", "new-key", "new-key"]


@pytest.mark.parametrize("mode", ["meta", "analyze", "plan", "simulate"])
@pytest.mark.parametrize("flag", ["--edison-api-key", "--futurehouse-api-key"])
def test_every_mode_takes_the_key_under_the_new_flag_and_the_earlier_one(mode, flag):
    from scilink.cli.shell.modes import ADAPTERS
    adapter = ADAPTERS[mode]
    adapter = adapter() if isinstance(adapter, type) else adapter
    parser = argparse.ArgumentParser()
    adapter.add_arguments(parser)
    args, _ = parser.parse_known_args([flag, "k"])
    assert args.futurehouse_api_key == "k"


def test_no_reader_names_the_variable_itself():
    """A reader that names one variable misses the other: every read of the
    key from the environment goes through ``auth.edison_api_key``."""
    files = subprocess.run(["git", "ls-files", "scilink/*.py"], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout.split()
    read = re.compile(r"(environ(\.get)?|getenv)\s*[\(\[]\s*[\"'](%s|%s)[\"']" % (NEW, OLD))
    offenders = [f for f in files if f != "scilink/auth.py"
                 and read.search((REPO / f).read_text(encoding="utf-8"))]
    assert offenders == []


def test_what_a_person_reads_says_edison_scientific():
    """Labels, help, messages and prompts carry the organisation's name as it
    is now; the earlier one stays only where it is explained (the key module,
    the client's note on the renamed job) and in identifiers."""
    files = subprocess.run(["git", "ls-files", "scilink", "webui/src", "README.md", "Dockerfile"],
                           cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    explained = {"scilink/auth.py"}
    offenders = [f for f in files if f not in explained and (REPO / f).is_file()
                 and "FutureHouse" in (REPO / f).read_text(encoding="utf-8", errors="replace")]
    assert offenders == []
