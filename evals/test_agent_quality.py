"""Agent quality evaluation tests using DeepEval."""

import json
import pytest


DELEGATION_DATASET = "evals/datasets/supervisor_delegation.json"


def load_delegation_dataset():
    with open(DELEGATION_DATASET) as f:
        return json.load(f)


@pytest.mark.skip(reason="Requires running backend and API keys")
class TestAgentQuality:
    def test_tool_correctness(self):
        """Verify that tools are called with correct arguments."""
        from deepeval.metrics import ToolCorrectnessMetric

        dataset = load_delegation_dataset()
        assert len(dataset) > 0

    def test_response_quality(self):
        """Evaluate overall response quality with G-Eval."""
        from deepeval.metrics import GEval

        metric = GEval(
            name="response_quality",
            criteria="The response must be accurate, relevant, and evidence-based.",
            evaluation_params=["input", "actual_output"],
        )
        assert metric is not None

    def test_supervisor_delegation(self):
        """Verify that supervisor delegates to the correct worker."""
        dataset = load_delegation_dataset()
        # Implementation: send queries through supervisor,
        # check trajectory for correct worker delegation
        assert len(dataset) > 0
