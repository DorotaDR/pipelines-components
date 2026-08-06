# Determine Data Source And Config ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Determine data source (S3 or PVC) from secret environment variables.

Inspects the injected secret environment to determine whether to use S3 or PVC for data loading. The secret must contain either a complete S3 credential set or a complete PVC configuration set, but not both.

**S3 credentials (all required):** - AWS_ACCESS_KEY_ID - AWS_SECRET_ACCESS_KEY - AWS_S3_ENDPOINT - AWS_S3_BUCKET

**PVC credentials (all required):** - PVC_NAME

The component is designed to work with `use_secret_as_env(..., optional=True)` so missing keys don't fail at injection time. Validation happens inside this task.

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | Either "s3" or "pvc" indicating the detected data source. |

## Metadata 🗂️

- **Name**: determine_data_source_and_config
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Last Verified**: 2026-07-02 00:00:00+00:00
- **Tags**:
  - data-source
  - configuration
  - helper
- **Owners**:
  - Approvers:
    - dlaczak
  - Reviewers:
    - dlaczak

## Additional Resources 📚

- **Adr**: [https://github.com/red-hat-data-services/pipelines-components/blob/main/.adr/automl/features/pvc_data_source.md](https://github.com/red-hat-data-services/pipelines-components/blob/main/.adr/automl/features/pvc_data_source.md)
