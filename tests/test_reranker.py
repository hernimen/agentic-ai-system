import subprocess
import sys


def test_reranker_module_imports_without_sentence_transformers():
    script = """
import importlib.abc
import sys


class BlockSentenceTransformers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "sentence_transformers" or fullname.startswith("sentence_transformers."):
            raise ModuleNotFoundError("sentence-transformers unavailable")
        return None


sys.meta_path.insert(0, BlockSentenceTransformers())
sys.modules.pop("sentence_transformers", None)
import lims_agent.rag.reranker
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
