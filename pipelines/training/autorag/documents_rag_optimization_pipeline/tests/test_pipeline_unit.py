"""Unit tests for the documents_rag_optimization_pipeline pipeline."""

from pathlib import Path

from kfp_components.utils.compiled_pipeline_alignment import (
    assert_checked_in_pipeline_yaml_matches_compiled_ir,
)

from ..pipeline import documents_rag_optimization_pipeline


class TestDocumentsRagOptimizationPipelineUnit:
    """Unit tests for pipeline structure and interface."""

    def test_pipeline_is_callable(self):
        """Pipeline is a GraphComponent (callable with _component_inputs)."""
        assert callable(documents_rag_optimization_pipeline)
        assert hasattr(documents_rag_optimization_pipeline, "_component_inputs")

    def test_pipeline_required_parameters(self):
        """Pipeline declares expected required parameters."""
        inputs = getattr(documents_rag_optimization_pipeline, "_component_inputs", set())
        assert "test_data_secret_name" in inputs
        assert "test_data_bucket_name" in inputs
        assert "test_data_key" in inputs
        assert "input_data_secret_name" in inputs
        assert "input_data_bucket_name" in inputs
        assert "input_data_key" in inputs
        assert "llama_stack_secret_name" in inputs
        assert "responses_request_default_question" not in inputs

    def test_checked_in_pipeline_yaml_matches_source_ir(self):
        """Committed pipeline.yaml matches a fresh compile (IR + deployment; b64 embed + images redacted)."""
        yaml_path = Path(__file__).resolve().parent.parent / "pipeline.yaml"
        assert_checked_in_pipeline_yaml_matches_compiled_ir(
            pipeline_func=documents_rag_optimization_pipeline,
            checked_in_yaml_path=yaml_path,
        )
