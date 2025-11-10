import os, sys, tempfile, pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

@pytest.fixture(autouse=True, scope="session")
def _tmp_registry():
    tmpdir = tempfile.mkdtemp(prefix="synaptiq-reg-")
    os.environ["SYNAPTIQ_REGISTRY_PATH"] = os.path.join(tmpdir, "registry.json")
    yield
