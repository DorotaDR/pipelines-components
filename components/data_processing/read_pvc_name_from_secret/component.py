from kfp import dsl


@dsl.component(base_image="python:3.11")
def read_pvc_name_from_secret() -> str:
    """Extract PVC name from secret environment variables.

    Reads the PVC_NAME from the injected secret environment and returns it
    as a pipeline value. This allows the PVC name to stay in the Kubernetes
    Secret (Connection) rather than being hardcoded as a pipeline parameter.

    This component is typically used after `determine_data_source_and_config`
    confirms that PVC credentials are present, and before a data loader task
    that needs to mount the PVC.

    **Expected environment variable:**
    - PVC_NAME: Name of the existing PVC to mount

    Returns:
        str: The PVC name from the secret.

    Raises:
        ValueError: If PVC_NAME is not set or is empty in the environment.

    Example:
        # In pipeline
        data_source = determine_data_source_and_config()
        use_secret_as_env(...)  # inject PVC_NAME

        with dsl.If(data_source.output == "pvc"):
            pvc_name_task = read_pvc_name_from_secret()
            use_secret_as_env(
                pvc_name_task,
                secret_name="my-pvc-connection",
                secret_key_to_env={"PVC_NAME": "PVC_NAME"},
            )
            data_loader_task = tabular_data_loader(...)
            data_loader_task.mount_pvc(
                pvc_name=pvc_name_task.output,
                mount_path="/mnt/data",
            )
    """
    import os

    pvc_name = os.environ.get("PVC_NAME")

    if not pvc_name:
        raise ValueError(
            "PVC_NAME environment variable is not set or is empty. "
            "Ensure the secret contains a valid PVC_NAME key when using PVC data source."
        )

    return pvc_name


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        read_pvc_name_from_secret,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
