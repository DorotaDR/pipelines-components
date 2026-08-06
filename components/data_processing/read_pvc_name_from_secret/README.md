# Read Pvc Name From Secret ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Extract PVC name from secret environment variables.

Reads the PVC_NAME from the injected secret environment and returns it as a pipeline value. This allows the PVC name to stay in the Kubernetes Secret (Connection) rather than being hardcoded as a pipeline parameter.

This component is typically used after `determine_data_source_and_config` confirms that PVC credentials are present, and before a data loader task that needs to mount the PVC.

**Expected environment variable:** - PVC_NAME: Name of the existing PVC to mount

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | The PVC name from the secret. |

## Metadata 🗂️

- **Name**: read_pvc_name_from_secret
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Last Verified**: 2026-07-02 00:00:00+00:00
- **Tags**:
  - pvc
  - configuration
  - helper
- **Owners**:
  - Approvers:
    - dlaczak
  - Reviewers:
    - dlaczak

## Additional Resources 📚

- **Adr**: [https://github.com/red-hat-data-services/pipelines-components/blob/main/.adr/automl/features/pvc_data_source.md](https://github.com/red-hat-data-services/pipelines-components/blob/main/.adr/automl/features/pvc_data_source.md)
