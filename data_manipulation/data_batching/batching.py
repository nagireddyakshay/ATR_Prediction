# Import necessary libraries
import csv
import numpy as np
import pickle
import tempfile
import traceback
import boto3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Clear CUDA memory
torch.cuda.empty_cache()

# Determine device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# List of audio paths for embeddings
audio_paths = []
for m in range(1, 245):
    audio_paths.append(f'embedding_batch_{m}.pkl')

# Read labels from CSV
labels = pd.read_csv(r'labels.csv')

# Define a custom module to flatten input
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# Define a module to reshape to 256 dimensions
class ReshapeTo256(nn.Module):
    def __init__(self):
        super(ReshapeTo256, self).__init__()
        self.flatten = Flatten()
        self.linear = None
        self.input_size = None

    def forward(self, input):
        if self.linear is None:
            self.input_size = np.prod(input.shape[1:])
            self.linear = nn.Linear(self.input_size, 256).to(input.device)

        x = self.flatten(input)
        x = self.linear(x)
        return x

# Custom dataset class
class TuneInDataset(Dataset):
    def __init__(self, num_samples, input_length, vocab_size, img_input_shape, sound_embedding_dim,
                 sound2_embedding_dim, s3_bucket, audio_folder, audio_pickle_paths, segment_size=1):
        # Initialize attributes
        self.batch_size = None
        self.num_samples = num_samples
        self.input_length = input_length
        self.vocab_size = vocab_size
        self.img_input_shape = img_input_shape
        self.sound_embedding_dim = sound_embedding_dim
        self.sound2_embedding_dim = sound2_embedding_dim
        self.s3_bucket = s3_bucket
        self.audio_folder = audio_folder
        self.audio_pickle_paths = audio_pickle_paths
        self.text_embeddings = np.load('rf_text_embeddings.pkl', allow_pickle=True)
        self.video_embeddings = np.load('rf_video_embeddings.pkl', allow_pickle=True)
        self.targets = torch.tensor(
            labels['attention_data_completed'].to_list())  # targets are still randomized, replace with actual targets
        self.segment_size = segment_size
        self.current_segment = 0
        self.audio = []
        self.back_audio = []
        self.video = []
        self.text = []
        self.aws_access_key_id = 'YOUR_AWS_ACCESS_KEY_ID'
        self.aws_secret_access_key = 'YOUR_AWS_SECRET_ACCESS_KEY'
        self.s3_client = boto3.client('s3', aws_access_key_id=self.aws_access_key_id,
                                      aws_secret_access_key=self.aws_secret_access_key)

        # Load a temporary embedding for random assignment
        self.temp_emb = pd.read_pickle(r's3://adsresearch1/temp_embeddings/batch_1_0.pickle')
        self.random_assign = self.temp_emb['audio'][0]
        self.random_assign_bg = self.temp_emb['back_audio'][0]
        self.audio_features = None
        self.background_audio_features = None

        # Set the segment size and prepare batches
        self.segment_size = segment_size
        self.current_segment = 6241
        self.prepare_batches()

    def process_segment(self):
        # Process audio and video embeddings for a segment
        start_idx = self.current_segment * self.segment_size
        end_idx = min((self.current_segment + 1) * self.segment_size, len(self.video_embeddings['creative_id']))

        # Load sound batch indexes
        df = pd.read_csv('sound_indexes.csv')
        audio_batches = df['index'].tolist()

        model_audio = ReshapeTo256().to(device)
        model_backaudio = ReshapeTo256().to(device)
        for video in list(self.video_embeddings['creative_id'])[start_idx:end_idx]:
            if video in self.text_embeddings['creative_id']:
                text_embeddings_idx = self.text_embeddings['creative_id'].index(video)

                # Get video and text embeddings
                video_features = self.video_embeddings['embeddings'][text_embeddings_idx]
                text_features = self.text_embeddings['embeddings'][text_embeddings_idx]
                text_features = torch.tensor(text_features.reshape(1, 384))

            try:
                # Check if the video id is present in which audio batch
                for batch_index, sound_batch in enumerate(audio_batches):

                    if video in sound_batch:
                        with tempfile.NamedTemporaryFile() as tmp_file:
                            self.s3_client.download_file(self.s3_bucket, f'audio_embeddings/embedding_batch_{str(batch_index+1)}.pkl',
                                                        tmp_file.name)
                            with open(tmp_file.name, 'rb') as f:
                                sound_embeddings = pickle.load(f)
                                sound_embeddings_idx = sound_embeddings['id'].index(video)
                                self.audio_features = sound_embeddings['audio'][sound_embeddings_idx]

                                # If no audio features are found, assign random assignment
                                if len(self.audio_features) == 0:
                                    self.audio_features = self.random_assign
                                    # Log missing audio embeddings
                                    with open("missed_embeddings.csv", mode='a', newline='') as file:
                                        writer = csv.writer(file)
                                        writer.writerow(['audio' + video])

                                # If background audio features are missing, assign random assignment
                                if len(sound_embeddings['backaudio'][sound_embeddings_idx]) != 0:
                                    self.background_audio_features = sound_embeddings['backaudio'][sound_embeddings_idx]
                                else:
                                    self.background_audio_features = self.random_assign_bg
                                    # Log missing background audio embeddings
                                    with open("missed_embeddings.csv", mode='a', newline='') as file:
                                        writer = csv.writer(file)
                                        writer.writerow(['back' + video])

            except Exception as e:
                print("Error Occurred: ", str(e))  # Print error message

            # Process audio and background audio features
            if self.audio_features is not None:
                audio_features_arr = torch.tensor(self.audio_features, dtype=torch.float32).to(device)
                audio_features_arr = model_audio(audio_features_arr)
            else:
                audio_features_arr = None
                print(f"No Audio features for Video: {video}")

            if self.background_audio_features is not None:
                background_audio_features_arr = torch.tensor(self.background_audio_features, dtype=torch.float32).to(device)
                background_audio_features_arr = model_backaudio(background_audio_features_arr)
            else:
                background_audio_features_arr = None
                print(f"No background audio features for video: {video}")

            # Append processed features to respective lists
            self.video.append(video_features)
            self.text.append(text_features)
            self.audio.append(audio_features_arr)
            self.back_audio.append(background_audio_features_arr)

        self.current_segment += 1

    def has_next_segment(self):
        # Check if there's another segment to process
        return self.current_segment * self.segment_size < len(self.video_embeddings['creative_id'])

    def prepare_batches(self):
        # Prepare batches of data
        with torch.no_grad():
            self.batch_size = 16
            while self.has_next_segment():
                self.process_segment()
                print('Segments Processed', self.current_segment, 'Out of 8445')
                num_batches = len(self.audio) // self.batch_size + (1 if len(self.audio) % self.batch_size > 0 else 0)

                for i in range(num_batches):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, len(self.audio))
                    batch = {
                        'video': self.video[start_idx:end_idx],
                        'text': self.text[start_idx:end_idx],
                        'audio': self.audio[start_idx:end_idx],
                        'back_audio': self.back_audio[start_idx:end_idx],
                        'targets': self.targets[start_idx:end_idx]
                    }
                    self.save_batch(batch, i)
                # Clear processed data for the current segment
                self.video.clear()
                self.text.clear()
                self.audio.clear()
                self.back_audio.clear()

    def save_batch(self, batch, batch_idx):
        # Save a batch of data to S3
        data_pickle = pickle.dumps(batch)
        file_key = f'latest_embeddings/batch_{self.current_segment}_{batch_idx}.pickle'
        # Create an S3 resource
        s3_resource = boto3.resource('s3', aws_access_key_id=self.aws_access_key_id,
                                     aws_secret_access_key=self.aws_secret_access_key)
        # Access the bucket and put the object
        bucket = s3_resource.Bucket(self.s3_bucket)
        bucket.put_object(Key=file_key, Body=data_pickle)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get a data sample
        video = self.video[index]
        text = self.text[index]
        audio = self.audio[index]
        back_audio = self.back_audio[index]
        target = self.targets[index]

        return {
            'video': video,
            'text': text,
            'audio': audio,
            'back_audio': back_audio,
            'target': target
        }


# Main function
def main():
    try:
        # Define hyperparameters
        num_samples = 16889
        batch_size = 1
        input_length = 384
        vocab_size = 10000
        embedding_dim = 256
        lstm_units = 256
        img_input_shape = (1, 512, 512)
        sound_embedding_dim = 256
        sound2_embedding_dim = 256
        dense_units = 256
        num_classes = 1

        # Define your S3 bucket details and audio pickle file paths
        s3_bucket = 'adsresearch1'
        audio_folder = 'audio_embeddings'
        audio_pickle_paths = audio_paths

        # Create the dataset and dataloader
        dataset = TuneInDataset(num_samples, input_length, vocab_size, img_input_shape, sound_embedding_dim,
                                sound2_embedding_dim, s3_bucket, audio_folder, audio_pickle_paths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    except Exception as e:
        print(e)
        traceback.print_exc()


# Entry point of the script
if __name__ == '__main__':
    main()