from pathlib import Path
from typing import List, NamedTuple

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]

_SHARED_DIR = Path(__file__).parent.parent / "shared"


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_SHARED_DIR),
)
def timeseries_leaderboard_evaluation(
    models: List[dsl.Model],
    eval_metric: str,
    html_artifact: dsl.Output[dsl.HTML],
    embedded_artifact: dsl.EmbeddedInput[dsl.Artifact],
) -> NamedTuple("outputs", best_model=str):
    """Evaluate refitted AutoGluon timeseries models and generate a leaderboard.

    This component aggregates metrics from a list of model artifacts produced by
    ``autogluon_timeseries_models_full_refit`` (collected via ``dsl.Collected``) and
    generates an HTML-formatted leaderboard ranking the models by their performance metric.

    Each artifact in ``models`` must have been produced by
    ``autogluon_timeseries_models_full_refit``, which writes the following layout under
    the artifact path::

        {artifact.path}/{model_name_full}/metrics/metrics.json
        {artifact.path}/{model_name_full}/metrics/back_testing.json  (optional)
        {artifact.path}/{model_name_full}/predictor/
        {artifact.path}/{model_name_full}/notebooks/

    When ``back_testing.json`` is missing, empty, or invalid (e.g. series too short for
    rolling cutoffs), backtest columns show ``n/a`` and ``backtest_n`` is ``0``.

    Note: KFP does not propagate artifact metadata through executor inputs for collected
    artifact lists, so metrics are read from ``metrics/metrics.json`` on the filesystem
    rather than from ``artifact.metadata``.

    Args:
        models: List of model artifacts from ``autogluon_timeseries_models_full_refit``
            collected via ``dsl.Collected``. Each artifact provides metrics and location
            metadata for one refitted model.
        eval_metric: Metric name for ranking (e.g. ``"MASE"``, ``"WAPE"``);
            leaderboard is sorted descending (AutoGluon uses higher-is-better convention
            so metrics like MASE are negated - higher value means better model).
        html_artifact: Output artifact for the HTML-formatted leaderboard.
        embedded_artifact: Embedded shared files injected by the KFP runtime;
            provides ``leaderboard_html_template.html``.

    Returns:
        NamedTuple with ``best_model`` (str): display name of the top-ranked model.

    Raises:
        TypeError: If ``eval_metric`` is empty or not a string.
        ValueError: If ``models`` list is empty, or if no artifact produced a ``metrics.json``.

    Note:
        Artifacts without a ``metrics/metrics.json`` file are skipped with a warning
        (e.g. when the corresponding refit task failed).

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_timeseries_leaderboard_evaluation import (
            timeseries_leaderboard_evaluation
        )

        @dsl.pipeline(name="ts-evaluation-pipeline")
        def evaluation_pipeline():
            with dsl.ParallelFor(items=selection_task.outputs["top_models"]) as model_name:
                refit_task = autogluon_timeseries_models_full_refit(model_name=model_name, ...)
            leaderboard = timeseries_leaderboard_evaluation(
                models=dsl.Collected(refit_task.outputs["model_artifact"]),
                eval_metric="MASE",
            )
    """  # noqa: E501
    import json
    import logging
    import math
    from pathlib import Path

    import pandas as pd
    from leaderboard_utils import _build_leaderboard_html, _build_leaderboard_table, _round_metrics

    logger = logging.getLogger(__name__)

    def _load_backtest_rows(metrics_json_path: Path) -> list:
        """Load ``back_testing.json`` next to ``metrics.json``; return [] if unusable."""
        backtest_path = metrics_json_path.parent / "back_testing.json"
        if not backtest_path.is_file():
            return []
        try:
            data = json.loads(backtest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning("Ignoring invalid back_testing.json at %s: %s", backtest_path, e)
            return []
        if not isinstance(data, dict):
            return []
        per_window_metrics = data.get("per_window_metrics")
        if not isinstance(per_window_metrics, list):
            return []
        return [x for x in per_window_metrics if isinstance(x, dict) and isinstance(x.get("metrics"), dict)]

    def _numeric_metric_keys(metrics_dict: dict) -> set[str]:
        return {k for k, v in metrics_dict.items() if isinstance(v, (int, float))}

    def _backtest_mean_for_metric(backtest_rows: list, metric_key: str) -> float | str:
        vals = []
        for entry in backtest_rows:
            v = entry["metrics"].get(metric_key)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                vals.append(float(v))
        if not vals:
            return "n/a"
        return round(sum(vals) / len(vals), 4)

    # Input validation
    if not isinstance(eval_metric, str) or not eval_metric.strip():
        raise TypeError("eval_metric must be a non-empty string.")
    if not models:
        raise ValueError("models list must not be empty.")

    results = []
    for artifact in models:
        artifact_path = Path(artifact.path)
        base_uri = artifact.uri.rstrip("/")

        # KFP does not propagate artifact metadata through executor inputs for collected
        # artifact lists — artifact.metadata is always empty here. Read from the files
        # written by autogluon_timeseries_models_full_refit instead.
        #
        # Layout written by the refit component:
        #   {artifact.path}/{model_name_full}/metrics/metrics.json
        #   {artifact.path}/{model_name_full}/metrics/back_testing.json  (optional)
        #   {artifact.path}/{model_name_full}/predictor/
        #   {artifact.path}/{model_name_full}/notebooks/
        metrics_file = next(artifact_path.glob("*/metrics/metrics.json"), None)
        if metrics_file is None:
            logger.warning(
                "Skipping artifact at '%s': no metrics/metrics.json found. "
                "The refit task may have failed for this model.",
                artifact.uri,
            )
            continue
        model_name = metrics_file.relative_to(artifact_path).parts[0]

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_dict = json.load(f)

        backtest_rows = _load_backtest_rows(metrics_file)

        predictor_uri = f"{base_uri}/{model_name}/predictor"
        notebook_uri = f"{base_uri}/{model_name}/notebooks/automl_predictor_notebook.ipynb"

        results.append(
            {
                "model": model_name,
                "metrics_dict": metrics_dict,
                "backtest_rows": backtest_rows,
                "notebook": notebook_uri,
                "predictor": predictor_uri,
            }
        )

    if not results:
        raise ValueError(
            "No valid model artifacts found. All refit tasks may have failed "
            "or produced no metrics/metrics.json output."
        )

    holdout_metric_keys: set[str] = set()
    for entry in results:
        holdout_metric_keys.update(_numeric_metric_keys(entry["metrics_dict"]))

    table_rows: list[dict] = []
    sorted_metric_keys = sorted(holdout_metric_keys)
    backtest_value_cols = [f"backtest_{k}_mean" for k in sorted_metric_keys]
    for entry in results:
        metrics_dict = entry["metrics_dict"]
        backtest_rows = entry["backtest_rows"]
        n_bt = len(backtest_rows)
        row: dict = {
            "model": entry["model"],
            **_round_metrics({k: metrics_dict[k] for k in sorted_metric_keys if k in metrics_dict}),
            "backtest_n": n_bt,
            "notebook": entry["notebook"],
            "predictor": entry["predictor"],
        }
        for mk in sorted_metric_keys:
            col_name = f"backtest_{mk}_mean"
            if n_bt == 0:
                row[col_name] = "n/a"
            else:
                row[col_name] = _backtest_mean_for_metric(backtest_rows, mk)
        table_rows.append(row)

    ordered_columns = ["model"] + sorted_metric_keys + ["backtest_n"] + backtest_value_cols + ["notebook", "predictor"]

    # Notice: AutoGluon follows the "higher is better" strategy for all metrics.
    # Metrics like MASE and WAPE have their signs flipped so that higher value = better model.
    leaderboard_df = pd.DataFrame(table_rows, columns=ordered_columns).sort_values(
        by=eval_metric,
        ascending=False,
    )
    n = len(leaderboard_df)
    leaderboard_df.index = pd.RangeIndex(start=1, stop=n + 1, name="rank")

    html_table = _build_leaderboard_table(leaderboard_df)

    best_model_name = leaderboard_df.iloc[0]["model"]
    template_path = Path(embedded_artifact.path) / "leaderboard_html_template.html"
    html_content = _build_leaderboard_html(
        template_path=template_path,
        table_html=html_table,
        eval_metric=eval_metric,
        best_model_name=best_model_name,
        num_models=len(leaderboard_df),
    )
    with open(html_artifact.path, "w", encoding="utf-8") as f:
        f.write(html_content)

    html_artifact.metadata["data"] = leaderboard_df.to_dict()
    html_artifact.metadata["display_name"] = "automl_timeseries_leaderboard"
    return NamedTuple("outputs", best_model=str)(best_model=best_model_name)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        timeseries_leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
