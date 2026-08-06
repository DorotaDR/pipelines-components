from kfp import dsl


@dsl.component(base_image="python:3.11")
def determine_data_source_and_config() -> str:
    """Determine data source (S3 or PVC) from secret environment variables.

    Inspects the injected secret environment to determine whether to use S3 or PVC
    for data loading. The secret must contain either a complete S3 credential set
    or a complete PVC configuration set, but not both.

    **S3 credentials (all required):**
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_S3_ENDPOINT
    - AWS_S3_BUCKET

    **PVC credentials (all required):**
    - PVC_NAME

    The component is designed to work with `use_secret_as_env(..., optional=True)` so
    missing keys don't fail at injection time. Validation happens inside this task.

    Returns:
        str: Either "s3" or "pvc" indicating the detected data source.

    Raises:
        ValueError: If both S3 and PVC credentials are present, or if neither set is complete.
            Error messages explicitly name missing or conflicting keys for actionable debugging.

    Example:
        # In pipeline with S3 secret
        detect_task = determine_data_source_and_config()
        use_secret_as_env(
            detect_task,
            secret_name="my-s3-connection",
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
                "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
                "AWS_S3_BUCKET": "AWS_S3_BUCKET",
                "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
            },
            optional=True,
        )
        # detect_task.output will be "s3"

        # In pipeline with PVC secret
        detect_task = determine_data_source_and_config()
        use_secret_as_env(
            detect_task,
            secret_name="my-pvc-connection",
            secret_key_to_env={
                "PVC_NAME": "PVC_NAME",
            },
            optional=True,
        )
        # detect_task.output will be "pvc"
    """
    import os

    # S3 required keys
    s3_keys = {
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_S3_ENDPOINT": os.environ.get("AWS_S3_ENDPOINT"),
        "AWS_S3_BUCKET": os.environ.get("AWS_S3_BUCKET"),
    }

    # PVC required keys
    pvc_keys = {
        "PVC_NAME": os.environ.get("PVC_NAME"),
    }

    # Check which keys are present
    s3_present = [k for k, v in s3_keys.items() if v]
    pvc_present = [k for k, v in pvc_keys.items() if v]

    s3_complete = len(s3_present) == len(s3_keys)
    pvc_complete = len(pvc_present) == len(pvc_keys)

    # Validate mutual exclusivity and completeness
    if s3_complete and pvc_complete:
        raise ValueError(
            "Ambiguous data source configuration: secret contains both S3 and PVC credentials. "
            f"S3 keys present: {list(s3_keys.keys())}. "
            f"PVC keys present: {list(pvc_keys.keys())}. "
            "Please provide only one credential set (either S3 or PVC, not both)."
        )

    if s3_complete:
        return "s3"

    if pvc_complete:
        return "pvc"

    # Neither complete - build helpful error message
    s3_missing = [k for k, v in s3_keys.items() if not v]
    pvc_missing = [k for k, v in pvc_keys.items() if not v]

    error_parts = ["Incomplete data source configuration in secret."]

    if s3_present:
        error_parts.append(f"Partial S3 credentials detected. Missing keys: {s3_missing}")
    if pvc_present:
        error_parts.append(f"Partial PVC credentials detected. Missing keys: {pvc_missing}")
    if not s3_present and not pvc_present:
        error_parts.append(
            "No S3 or PVC credentials found. "
            f"Provide either S3 keys {list(s3_keys.keys())} or PVC keys {list(pvc_keys.keys())}."
        )

    raise ValueError(" ".join(error_parts))


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        determine_data_source_and_config,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
