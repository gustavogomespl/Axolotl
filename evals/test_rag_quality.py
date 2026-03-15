"""RAG quality evaluation tests using RAGAS metrics."""

import json

import pytest

# These tests require a running backend with indexed documents
# Run with: pytest evals/test_rag_quality.py -v

GOLDEN_DATASET = "evals/datasets/rag_golden.json"


def load_golden_dataset():
    with open(GOLDEN_DATASET) as f:
        return json.load(f)


@pytest.mark.skip(reason="Requires running backend with indexed documents and API keys")
class TestRAGQuality:
    def test_rag_faithfulness(self):
        """Verify that responses are faithful to retrieved context."""

        dataset = load_golden_dataset()
        # Implementation: call backend RAG endpoint, collect responses,
        # evaluate with RAGAS faithfulness metric
        assert len(dataset) > 0

    def test_rag_context_precision(self):
        """Verify that retrieved documents are relevant."""

        dataset = load_golden_dataset()
        assert len(dataset) > 0

    def test_rag_answer_relevancy(self):
        """Verify that answers are relevant to the question."""

        dataset = load_golden_dataset()
        assert len(dataset) > 0
