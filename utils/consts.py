import os

REDHAT_INDEX_URL = "https://console.redhat.com/api/pypi/public-rhai/rhoai/3.4/cpu-ubi9/simple/"

DEFAULT_AUTOML_IMAGE = "quay.io/opendatahub/odh-automl:odh-stable"
DEFAULT_AUTORAG_IMAGE = "quay.io/opendatahub/odh-autorag:odh-stable"

AUTOML_IMAGE = os.getenv("RELATED_IMAGE_MPI_AUTOML_RUNTIME", DEFAULT_AUTOML_IMAGE)
AUTORAG_IMAGE = os.getenv("RELATED_IMAGE_MPI_AUTORAG_RUNTIME", DEFAULT_AUTORAG_IMAGE)
