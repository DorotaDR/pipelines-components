"""Functional / benchmark tests for the tabular_data_loader component.

TEMPORARY: these tests exercise the *real* component (real boto3, pandas, sklearn)
against real datasets stored in an S3 bucket, and measure how long the component
takes to download + sample. They are intended for profiling the slowness reported
in RHOAIENG-54351, not for CI.

Unlike ``test_component_unit.py`` (which mocks everything), this module runs the
component end-to-end, so it requires:

* ``uv sync --extra dev`` (boto3, pandas, sklearn available), and
* S3 credentials + dataset locations provided via environment variables.

Configuration is read from a ``.env`` file (see ``.env.example`` in this directory);
the repo-root ``.env`` is loaded first, then this directory's ``.env`` overrides it.
Real environment variables set in the shell always win over both. Copy the template
and run:

    cp components/data_processing/automl/tabular_data_loader/tests/.env.example \
       components/data_processing/automl/tabular_data_loader/tests/.env
    # edit .env, then:
    uv run pytest components/data_processing/automl/tabular_data_loader/tests/test_component_benchmark.py \
        -m integration -v -s --log-cli-level=DEBUG

``--log-cli-level=DEBUG`` streams the component's DEBUG/INFO log messages live so you
can see download/sampling progress. Each test is skipped automatically when its
dataset / credential config is absent.
"""

import os
import time

import pytest

from ..component import automl_data_loader

# S3 credentials the component reads from the environment (see get_s3_client).
_REQUIRED_S3_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_S3_ENDPOINT")

_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    """Load .env from repo root (cwd) and from this directory (import guard: not at module scope).

    Values already present in the real environment are preserved: ``load_dotenv``
    does not override them unless ``override=True`` is passed, so a shell export
    still beats the file. The directory-local ``.env`` overrides the repo-root one.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    from pathlib import Path

    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


def _env(name: str, *fallbacks: str):
    """Return the first non-empty value for ``name``/``fallbacks`` after loading .env."""
    _ensure_dotenv_loaded()
    for key in (name, *fallbacks):
        value = os.environ.get(key)
        if value:
            return value
    return None


class _Artifact:
    """Minimal stand-in for a KFP Output[Artifact]/Output[Dataset]."""

    def __init__(self, path: str, uri: str = ""):
        self.path = path
        self.uri = uri
        self.metadata: dict = {}


def _missing_s3_env() -> list[str]:
    _ensure_dotenv_loaded()
    return [name for name in _REQUIRED_S3_ENV if not os.environ.get(name)]


def _bench_bucket():
    """Bucket holding both benchmark datasets."""
    return _env("BENCH_BUCKET", "AWS_S3_BUCKET")


def _s3_object_size_bytes(file_key: str):
    """Return the S3 object ContentLength in bytes, mirroring the component's client config."""
    import boto3

    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_S3_ENDPOINT"),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    head = s3_client.head_object(Bucket=_bench_bucket(), Key=file_key)
    return head["ContentLength"]


def _run_and_time(
    *,
    tmp_path,
    file_key: str,
    label_column: str,
    task_type: str,
    sampling_method: str,
    preset: str = "balanced",
):
    """Invoke the real component against S3 and return (result, elapsed_seconds).

    Defaults to the ``"balanced"`` preset so the benchmark exercises the 1 GB
    sampling budget; pass ``preset="speed"`` to measure the 100 MB path instead.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    sampled_test_dataset = _Artifact(path=str(tmp_path / "sampled_test_dataset.csv"))
    component_status = _Artifact(path=str(tmp_path / "component_status"))

    try:
        size_bytes = _s3_object_size_bytes(file_key)
    except Exception as exc:  # noqa: BLE001 - size is informational only
        size_bytes = None
        print(f"\n[benchmark] could not read object size for {file_key}: {exc}")

    start = time.perf_counter()
    result = automl_data_loader.python_func(
        file_key=file_key,
        bucket_name=_bench_bucket(),
        workspace_path=str(workspace),
        label_column=label_column,
        sampled_test_dataset=sampled_test_dataset,
        component_status=component_status,
        sampling_method=sampling_method,
        task_type=task_type,
        preset=preset,
    )
    elapsed = time.perf_counter() - start

    n_samples = result.sample_config.get("n_samples")
    if size_bytes is not None:
        size_mb = size_bytes / (1024 * 1024)
        throughput = f"{size_mb / elapsed:.1f} MB/s" if elapsed > 0 else "n/a"
        size_str = f"{size_mb:.1f} MB ({size_bytes} bytes)"
    else:
        throughput = "n/a"
        size_str = "unknown"
    print(
        f"\n[benchmark] task_type={task_type} sampling_method={sampling_method} preset={preset} "
        f"key=s3://{_bench_bucket()}/{file_key}\n"
        f"[benchmark]   object_size={size_str}\n"
        f"[benchmark]   elapsed={elapsed:.1f}s  throughput={throughput}  sampled_rows={n_samples}"
    )
    return result, elapsed


@pytest.mark.integration
class TestTabularDataLoaderBenchmark:
    """Time the component against large real datasets (RHOAIENG-54351)."""

    def test_binary_stratified(self, tmp_path):
        """Large classification dataset: binary task with stratified sampling."""
        missing = _missing_s3_env()
        if missing:
            pytest.skip(f"Missing S3 env vars: {', '.join(missing)}")
        bucket, key, label = _bench_bucket(), _env("BENCH_BINARY_KEY"), _env("BENCH_BINARY_LABEL")
        if not (bucket and key and label):
            pytest.skip("Set BENCH_BUCKET, BENCH_BINARY_KEY, BENCH_BINARY_LABEL to run.")

        result, elapsed = _run_and_time(
            tmp_path=tmp_path,
            file_key=key,
            label_column=label,
            task_type="binary",
            sampling_method="stratified",
        )
        assert result.sample_config.get("n_samples", 0) > 0

    def test_regression_random(self, tmp_path):
        """Large regression dataset: regression task with random sampling."""
        missing = _missing_s3_env()
        if missing:
            pytest.skip(f"Missing S3 env vars: {', '.join(missing)}")
        bucket, key, label = _bench_bucket(), _env("BENCH_REGRESSION_KEY"), _env("BENCH_REGRESSION_LABEL")
        if not (bucket and key and label):
            pytest.skip("Set BENCH_BUCKET, BENCH_REGRESSION_KEY, BENCH_REGRESSION_LABEL to run.")

        result, elapsed = _run_and_time(
            tmp_path=tmp_path,
            file_key=key,
            label_column=label,
            task_type="regression",
            sampling_method="random",
        )
        assert result.sample_config.get("n_samples", 0) > 0
