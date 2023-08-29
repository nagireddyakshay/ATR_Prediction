import boto3
import pandas as pd
import pickle
import os
import numpy as np

# Audio path
# Audio paths
audio_paths = []
for i in range(1, 16890):
    audio_paths.append(f'batch_{i}_0.pickle')

# Connect to the S3 service
self.aws_access_key_id = 'YOUR_AWS_ACCESS_KEY_ID'
self.aws_secret_access_key = 'YOUR_AWS_SECRET_ACCESS_KEY'
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key)

# Load the main video embeddings file from your local file system
with open('rf_video_embeddings.pkl', 'rb') as file:
    main_video_embeddings = pickle.load(file)

# Load the targets CSV file from your local file system
targets_df = pd.read_csv()

# Specify the S3 bucket name
bucket_name = 'adsresearch1'

# List of batch file names in the S3 bucket
batch_files_s3 = audio_paths

# len(batch_files_s3)
# Iterate over each batch file in the S3 bucket
for batch_index, batch_file_s3 in enumerate(batch_files_s3):
    # Download the batch embeddings file from the S3 bucket
    batch_file_local = 'temp_batch.pkl'  # Temporary local file name
    s3_client.download_file(bucket_name, f'complete_embeddings/{batch_file_s3}', batch_file_local)

    # Load the batch embeddings file from the local file system
    with open(batch_file_local, 'rb') as file:
        batch_data = pickle.load(file)

    # Retrieve the video embeddings from the batch
    video_embeddings = batch_data['video']

    targets = []

    # Iterate over each video embedding in the batch
    for embedding in video_embeddings:

        # Loop through the main embeddings file
        for ind, video_emb in enumerate(main_video_embeddings['embeddings']):
            if np.array_equal(video_emb, embedding):
                match_index = int(main_video_embeddings['creative_id'][ind])
                # Look up the target for the corresponding ID in the targets CSV file
                try:
                    targets.append(
                        targets_df[targets_df['creative_data_id'] == match_index]['attention_data_completed'].values[0])
                except:
                    print('Except invoked')
                    targets.append(0)

    # Replace the target in the original batch file
    batch_data['targets'] = targets

    # Save the updated batch file with the same file name in the S3 bucket
    with open(batch_file_local, 'wb') as file:
        pickle.dump(batch_data, file)

    # Upload the updated batch file to the S3 bucket with the same file name
    s3_client.upload_file(batch_file_local, bucket_name, f'complete_embeddings/{batch_file_s3}')

    print('Batches Processed', batch_index + 1, 'Out of 16k change')

    # Remove the temporary local file
    os.remove(batch_file_local)