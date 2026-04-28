import pathlib

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]

_NOTEBOOKS_DIR = str(pathlib.Path(__file__).parent / "notebook_templates")


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
    embedded_artifact_path=_NOTEBOOKS_DIR,
)
def autogluon_timeseries_models_full_refit(
    model_name: str,
    test_dataset: dsl.Input[dsl.Dataset],
    predictor_path: str,
    sampling_config: dict,
    split_config: dict,
    model_config: dict,
    pipeline_name: str,
    run_id: str,
    models_selection_train_data_path: str,
    extra_train_data_path: str,
    sample_rows: str,
    notebooks: dsl.EmbeddedInput[dsl.Dataset],
    model_artifact: dsl.Output[dsl.Model],
    backtest_num_windows: int = 3,
):
    """Refit a single AutoGluon timeseries model on full training data.

    This component takes a model selected during the selection phase and
    refits it on the full training dataset (selection + extra train data)
    for improved performance. The refitted model is optimized and saved
    for deployment. Each model directory contains a ``model.json`` file
    with model metadata (name, base model, location, metrics).

    After refit, the pipeline hold-out ``test_dataset`` is scored once with
    :meth:`~autogluon.timeseries.TimeSeriesPredictor.evaluate` (final window).
    Additional **rolling backtests** reuse the same test frame with negative
    ``cutoff`` values spaced by ``prediction_length``, matching AutoGluon’s
    documented ``evaluate`` cutoff semantics. Results are written as
    ``metrics/back_testing.json`` (ADR-aligned summary structure).

    Args:
        model_name: Name of the model to refit.
        test_dataset: Test dataset artifact for evaluation.
        predictor_path: Path to the predictor from selection phase.
        sampling_config: Configuration used for data sampling.
        split_config: Configuration used for data splitting.
        model_config: Model configuration from selection phase.
        pipeline_name: Pipeline name for metadata.
        run_id: Pipeline run ID for metadata.
        models_selection_train_data_path: Path to the model-selection train split CSV
            (earlier segment of the train portion).
        extra_train_data_path: Path to the extra train split CSV (later segment of the train portion).
        sample_rows: Sample rows from test dataset as JSON string.
        model_artifact: Output artifact for the refitted model.
        notebooks: Embedded notebook templates (injected by the runtime from the component's embedded_artifact_path).
        backtest_num_windows: Number of rolling ``evaluate`` windows on ``test_dataset``
            using ``cutoff=-k * prediction_length`` for ``k`` from ``backtest_num_windows``
            down to ``1``. Must be >= 1. Backtest windows that cannot be evaluated
            (for example too-short series) are skipped with a warning.
    """
    import json
    import logging
    import math
    import os
    from pathlib import Path

    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    from autogluon.timeseries.metrics import AVAILABLE_METRICS
    from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel

    logger = logging.getLogger(__name__)

    logger.info("Timeseries refit: model=%s", model_name)

    if backtest_num_windows < 1:
        raise ValueError("backtest_num_windows must be >= 1")

    # Load the predictor from selection phase
    try:
        predictor = TimeSeriesPredictor.load(predictor_path)
    except Exception as e:
        logger.error("Failed to load predictor: %s", e)
        raise ValueError(f"Could not load predictor from {predictor_path}: {e}") from e
    logger.debug(
        "Loaded selection predictor from %s; selection_train=%s extra_train=%s",
        predictor_path,
        models_selection_train_data_path,
        extra_train_data_path,
    )

    id_column = model_config.get("id_column")
    timestamp_column = model_config.get("timestamp_column")

    selection_ts_df = TimeSeriesDataFrame.from_path(
        path=models_selection_train_data_path,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    extra_train_ts_df = TimeSeriesDataFrame.from_path(
        path=extra_train_data_path,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    # Full train portion (excluding held-out test) = selection split then extra split, in time order.
    full_train_ts_df = TimeSeriesDataFrame(
        pd.concat([selection_ts_df, extra_train_ts_df], axis=0),
    )
    test_ts_df = TimeSeriesDataFrame.from_path(
        path=test_dataset.path,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    logger.debug(
        "Train rows=%s (selection=%s extra=%s) test rows=%s (id=%s ts=%s)",
        len(full_train_ts_df),
        len(selection_ts_df),
        len(extra_train_ts_df),
        len(test_ts_df),
        id_column,
        timestamp_column,
    )

    # Create model output directory
    model_name_full = f"{model_name}_FULL"
    output_path = Path(model_artifact.path) / model_name_full
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the predictor with the selected model
    predictor_output = output_path / "predictor"

    def is_ensemble_model(predictor, model_name: str) -> bool:
        """Return True if the named model is an ensemble."""
        model_type = predictor._trainer.get_model_attribute(model_name, "type")
        return issubclass(model_type, AbstractTimeSeriesEnsembleModel)

    is_ensemble = is_ensemble_model(predictor, model_name)

    predictor_refit = TimeSeriesPredictor(
        prediction_length=model_config.get("prediction_length"),  # 7 days
        path=predictor_output,
        target=model_config.get("target"),
        eval_metric=model_config.get("eval_metric", "MASE"),
    )

    hyperparams_option = "ensemble_hyperparameters" if is_ensemble else "hyperparameters"
    # TODO: Save model_hyperparams in the output of the previous step & remove the predictor usage here
    additional_fit_params = {hyperparams_option: {model_name: predictor.fit_summary()["model_hyperparams"][model_name]}}
    logger.debug("Refit hyperparameters: %s", additional_fit_params)

    predictor_refit.fit(
        train_data=full_train_ts_df,
        **additional_fit_params,
        # exclude deep learning models pretrained on large time series datasets
        excluded_model_types=[
            "Chronos",
            "Chronos2",
            "Toto",
        ],
    )

    try:
        predictor_refit.save()
    except Exception as e:
        logger.error("Failed to save predictor: %s", e)
        raise ValueError(f"Could not save predictor to {predictor_output}: {e}") from e

    eval_metric_names = list(AVAILABLE_METRICS.keys())

    def _metrics_to_json_dict(raw: dict) -> dict:
        return {
            k: float(v) if hasattr(v, "item") else v
            for k, v in raw.items()
            if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
        }

    def _compute_overall_backtest_metrics(per_window_metrics: list[dict]) -> dict:
        """Compute metric-wise means across successful backtest windows."""
        metric_to_values: dict[str, list[float]] = {}
        for entry in per_window_metrics:
            for metric_name, metric_value in entry["metrics"].items():
                if isinstance(metric_value, (int, float)):
                    metric_to_values.setdefault(metric_name, []).append(float(metric_value))
        return {name: sum(values) / len(values) for name, values in metric_to_values.items() if values}

    try:
        metrics = predictor_refit.evaluate(test_ts_df, metrics=eval_metric_names)
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise ValueError(f"Failed to evaluate model: {e}") from e
    logger.debug("Evaluation metrics (hold-out): %s", metrics)

    prediction_length = int(model_config.get("prediction_length", 1))
    per_window_metrics: list[dict] = []
    skipped_backtest_windows: list[dict] = []
    for k in range(backtest_num_windows, 0, -1):
        cutoff = -k * prediction_length
        try:
            window_metrics = predictor_refit.evaluate(
                test_ts_df,
                metrics=eval_metric_names,
                cutoff=cutoff,
            )
        except Exception as e:
            logger.warning("Backtest evaluation skipped (cutoff=%s): %s", cutoff, e)
            skipped_backtest_windows.append({"cutoff": cutoff, "reason": str(e)})
            continue
        sanitized_window_metrics = _metrics_to_json_dict(window_metrics)
        per_window_metrics.append(
            {
                "window_id": backtest_num_windows - k,
                "cutoff": cutoff,
                "metrics": sanitized_window_metrics,
            },
        )
        logger.debug("Backtest cutoff=%s metrics: %s", cutoff, window_metrics)

    back_testing_summary = {
        "model_name": model_name_full,
        "prediction_length": prediction_length,
        "num_val_windows": backtest_num_windows,
        "eval_metric": model_config.get("eval_metric", "MASE"),
        "target": model_config.get("target"),
        "id_column": model_config.get("id_column"),
        "timestamp_column": model_config.get("timestamp_column"),
        "overall_metrics": _compute_overall_backtest_metrics(per_window_metrics),
        "per_window_metrics": per_window_metrics,
        "metadata": {
            "backtest_strategy": "expanding_window",
            "num_windows_requested": backtest_num_windows,
            "num_windows_succeeded": len(per_window_metrics),
            "num_windows_skipped": len(skipped_backtest_windows),
            "skipped_windows": skipped_backtest_windows,
        },
    }

    # Save additional metadata about the selected model
    predictor_metadata = {
        "model_name": model_name_full,
        "base_model": model_name,
        "selected_model": model_name,
        "prediction_length": model_config.get("prediction_length", 1),
        "eval_metric": model_config.get("eval_metric", "MASE"),
        "target": model_config.get("target"),
        "id_column": model_config.get("id_column"),
        "timestamp_column": model_config.get("timestamp_column"),
        "backtest_num_windows": backtest_num_windows,
    }

    with open(predictor_output / "predictor_metadata.json", "w") as f:
        json.dump(predictor_metadata, f, indent=2)

    metrics_path = output_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    metrics_dict = _metrics_to_json_dict(metrics)

    with open(metrics_path / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(metrics_path / "back_testing.json", "w") as f:
        json.dump(back_testing_summary, f, indent=2)

    # Notebook generation

    notebook_file = "timeseries_notebook.ipynb"

    with open(os.path.join(notebooks.path, notebook_file), "r", encoding="utf-8") as f:
        notebook = json.load(f)

        # Improved retrieve_pipeline_name: trims only the run id or suffix
        def retrieve_pipeline_name(pipeline_name: str) -> str:
            """Attempts to infer the original pipeline name from a name that may have a run id or suffix at the end.

            Removes only the last dash-separated element (the run id or variant),
            handling trailing dashes gracefully to avoid dropping real name segments.
            If only a single element exists, returns as is.
            """
            if not pipeline_name:
                return pipeline_name
            # Strip trailing dashes for robust splitting
            name = pipeline_name.rstrip("-")
            if "-" not in name:
                return name
            tokens = name.split("-")
            if len(tokens) <= 1:
                return tokens[0] if tokens else ""
            return "-".join(tokens[:-1])

        pipeline_name = retrieve_pipeline_name(pipeline_name)

        # Replace <REPLACE_* placeholders (run id, pipeline name, model, sample row, extra pip index, …) in code cells. # noqa: E501
        def replace_placeholder_in_notebook(notebook, replacements):
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                # Replace in every string of the source list
                new_source = []
                for line in cell.get("source", []):
                    for placeholder, value in replacements.items():
                        line = line.replace(placeholder, value)
                    new_source.append(line)
                cell["source"] = new_source
            return notebook

        sample_row_list = json.loads(sample_rows)

        replacements = {
            "<REPLACE_RUN_ID>": run_id,
            "<REPLACE_PIPELINE_NAME>": pipeline_name,
            "<REPLACE_MODEL_NAME>": model_name_full,
            "<REPLACE_SAMPLE_ROW>": str(sample_row_list),
            "<REPLACE_ID_COLUMN>": model_config.get("id_column"),
            "<REPLACE_TIMESTAMP_COLUMN>": model_config.get("timestamp_column"),
            "<REPLACE_KNOWN_COVARIATES_NAMES>": str(model_config.get("known_covariates_names") or []),
        }
        notebook = replace_placeholder_in_notebook(notebook, replacements)

    notebook_path = output_path / "notebooks"
    notebook_path.mkdir(parents=True, exist_ok=True)
    with (notebook_path / "automl_predictor_notebook.ipynb").open("w", encoding="utf-8") as f:
        json.dump(notebook, f)

    # Write model.json alongside predictor/, metrics/, notebooks/
    model_metadata = {
        "name": model_name_full,
        "base_model": model_name,
        "location": {
            "model_directory": model_name_full,
            "predictor": f"{model_name_full}/predictor",
            "metrics": f"{model_name_full}/metrics",
            "notebooks": f"{model_name_full}/notebooks",
        },
        "metrics": {
            "test_data": metrics_dict,
            "back_testing": back_testing_summary,
        },
    }
    with (output_path / "model.json").open("w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # Set artifact metadata
    model_artifact.metadata["display_name"] = model_name_full
    model_artifact.metadata["context"] = {
        "model_config": model_config,
        "sampling_config": sampling_config,
        "split_config": split_config,
        "metrics": {
            "test_data": metrics_dict,
            "back_testing": back_testing_summary,
        },
        "location": {
            "model_directory": model_name_full,
            "predictor": f"{model_name_full}/predictor",
            "metrics": f"{model_name_full}/metrics",
            "notebooks": f"{model_name_full}/notebooks",
        },
        "pipeline_info": {
            "pipeline_name": pipeline_name,
            "run_id": run_id,
        },
    }

    logger.info("Timeseries refit done: %s (artifact under %s)", model_name_full, output_path)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_timeseries_models_full_refit,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
