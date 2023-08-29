# Pickled Batch label correction (Needed only when necessary)

This script is designed to process audio and video embeddings, extract target data, and leverage Amazon Web Services (AWS) S3 for data storage and retrieval. By matching video embeddings, updating batch files, and utilizing S3 for storage, the script efficiently manages substantial data sets.

## Prerequisites

Before utilizing this script, ensure the following prerequisites are met:

- An active AWS account with appropriate access keys (`aws_access_key_id` and `aws_secret_access_key`) and S3 bucket permissions.
- A Python environment equipped with required libraries: `boto3`, `pandas`, `pickle`, `os`, and `numpy`.

## Functionality

1. **Library Imports**:

    Begin by importing the necessary libraries:

    ```python
    import boto3
    import pandas as pd
    import pickle
    import os
    import numpy as np
    ```

2. **Define Audio Paths**:

    Define paths for batched audio embeddings:

    ```python
    audio_paths = [f'batch_{i}_0.pickle' for i in range(1, 16890)]
    ```

3. **AWS S3 Connection**:

    Establish a connection to AWS S3 using provided access keys:

    ```python
    s3_client = boto3.client('s3', aws_access_key_id='xxx', aws_secret_access_key='xxx')
    ```

4. **Load Main Video Embeddings**:

    Load main video embeddings from a local file named `rf_video_embeddings.pkl`:

    ```python
    with open('rf_video_embeddings.pkl', 'rb') as file:
        main_video_embeddings = pickle.load(file)
    ```

5. **Load Target Data**:

    Load target data from a CSV file into a Pandas DataFrame:

    ```python
    targets_df = pd.read_csv('targets.csv')  # Specify the actual CSV file name/path
    ```

6. **S3 Bucket Specification**:

    Specify the S3 bucket name:

    ```python
    bucket_name = 'adsresearch1'
    ```

7. **Batch Processing Loop**:

    Iterate through each batch file in the S3 bucket:

    ```python
    for batch_index, batch_file_s3 in enumerate(audio_paths):
        # Batch processing steps...
    ```

    Inside the loop, the following actions are performed:

    - Download batch embeddings from S3.
    - Load batch embeddings from the local file system.
    - Match video embeddings with main embeddings.
    - Extract target data.
    - Update batch data.
    - Save and upload the updated batch file.
    - Display progress and clean up temporary files.

## Functionality

- AWS S3 connection for data storage.
- Loading and processing video embeddings.
- Matching video embeddings to extract target data.
- Updating batch files with computed targets.
- Leveraging S3 for uploading and downloading files.
- Progress tracking and temporary file management.
