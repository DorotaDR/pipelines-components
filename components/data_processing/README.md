# Data Processing Components

This directory contains components in the **Data Processing** category:

- [Dataset Download](./dataset_download/README.md): Download and prepare datasets from multiple sources.
- [Determine Data Source And Config](./determine_data_source_and_config/README.md): Determine data source (S3 or PVC) from secret environment variables.
- [Download Model](./download_model/README.md): Download a HuggingFace model to a PVC for caching.
- [Ingest To Milvus](./ingest_to_milvus/README.md): Read chunks from S3, embed, and insert into Milvus.
- [Parse And Chunk](./parse_and_chunk/README.md): Parse PDFs and write chunked JSONL files to S3.
- [Read Pvc Name From Secret](./read_pvc_name_from_secret/README.md): Extract PVC name from secret environment variables.
- [Sdg Hub](./sdg/README.md): Run an SDG Hub flow to generate synthetic data.
- [Yoda Data Processor](./yoda_data_processor/README.md): Prepare the training and evaluation datasets by downloading and preprocessing.

## Subcategories

- [Automl](./automl/README.md)
- [Autorag](./autorag/README.md)
