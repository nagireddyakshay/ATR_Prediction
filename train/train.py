#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import boto3
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
from scipy.stats import spearmanr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn.utils.rnn import pad_sequence
torch.cuda.empty_cache()


def sequences_to_tensor(sequences):
    if sequences is None:
        return None

    sequence_tensors = []
    for sequence in sequences:
        if sequence is not None:
            tensor = torch.tensor(sequence)
            if len(tensor.shape) < 1:
                tensor = tensor.unsqueeze(0)
            sequence_tensors.append(tensor)

    if len(sequence_tensors) == 0:  # if all sequences were None
        return None

    # Pad the sequences
    padded_sequence = pad_sequence(sequence_tensors, batch_first=True)

    return padded_sequence

class S3Dataset(Dataset):
    # Initialize missing data counter
    missing_counter = 0

    def __init__(self, bucket, prefix, aws_access_key_id, aws_secret_access_key):
        self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        self.bucket = bucket
        self.prefix = prefix

        # Get a list of all objects in the bucket with the given prefix
        self.keys = []
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            self.keys.extend([item['Key'] for item in page.get('Contents', []) if
                              not item['Key'].endswith('/')])  # exclude directories

        # Cache for data
        self.cache = {}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx_tuple):
        # Unpack the tuple
        if isinstance(idx_tuple, tuple):
            idx, compare_with = idx_tuple
        else:
            idx = idx_tuple
            compare_with = None

        key = self.keys[idx]

        # If the data is not in the cache, download it
        if key not in self.cache:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            self.cache[key] = pickle.loads(response['Body'].read())

        # Get the data from the cache
        batch = self.cache[key]
        video_tensors = sequences_to_tensor(batch['video'])
        text_tensors = sequences_to_tensor(batch['text'])
        audio_tensors = sequences_to_tensor(batch['audio'])
        back_audio_tensors = sequences_to_tensor(batch['back_audio'])
        targets_tensor = torch.tensor(batch['targets'])

        # Check if any tensors are None
        if (
                video_tensors is None
                or text_tensors is None
                or audio_tensors is None
                or back_audio_tensors is None
                or targets_tensor is None
        ):
            print(f"Skipping sample {idx} due to missing tensors.")
            # Increase the missing counter by 1
            #             self.missing_counter += 1
            #             print(f"Total missing data count: {self.missing_counter}")
            # Ensure we don't exceed the length of dataset
            if idx + 1 >= len(self.keys):
                return None
            return self.__getitem__((idx + 1, compare_with))  # Skip to the next sample

        # Compare data with another batch if specified
        if compare_with is not None:
            compare_key = self.keys[compare_with]
            if compare_key not in self.cache:
                response = self.s3.get_object(Bucket=self.bucket, Key=compare_key)
                self.cache[compare_key] = pickle.loads(response['Body'].read())

            compare_batch = self.cache[compare_key]
            compare_targets_tensor = torch.tensor(compare_batch['targets'])

            print(f"Comparing 'targets' in batch {idx} and batch {compare_with}:")
            equality_mask = targets_tensor == compare_targets_tensor
            print(f" - Equality mask for 'targets': {equality_mask}")

        return {
            'video': video_tensors,
            'text': text_tensors,
            'audio': audio_tensors,
            'back_audio': back_audio_tensors,
            'targets': targets_tensor,
        }


class MultiModalAttention(nn.Module):
    def __init__(self, text_features_dim, context_features_dim):
        super(MultiModalAttention, self).__init__()
        self.W1 = nn.Linear(text_features_dim, text_features_dim)
        self.W2 = nn.Linear(context_features_dim, text_features_dim)
        self.V = nn.Linear(text_features_dim, 1)

    def forward(self, text_features, context_features):
        text_proj = self.W1(text_features)
        context_proj = self.W2(context_features)

        combined_proj = torch.tanh(text_proj + context_proj)
        attention_weights = F.softmax(self.V(combined_proj), dim=1)
        attended_features = attention_weights * text_features

        return attended_features  # No sum operation, maintain dimensionality


class MultiModalAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, img_input_dim, sound_embedding_dim, dense_units,
                 num_classes, brand_classes, creative_classes):
        super(MultiModalAttentionModel, self).__init__()
        self.text_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.text_lstm_layer = nn.LSTM(embedding_dim, lstm_units, batch_first=True)

        self.img_dense_layer = nn.Sequential(
            nn.Linear(img_input_dim, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, dense_units),
        )
        self.sound_input_layer = nn.LSTM(sound_embedding_dim, dense_units, batch_first=True)
        self.sound2_input_layer = nn.LSTM(sound_embedding_dim, dense_units, batch_first=True)

        self.brand_layer = nn.Linear(brand_classes, dense_units)
        self.creative_layer = nn.Linear(creative_classes, dense_units)

        self.attention_layer_text_img = MultiModalAttention(lstm_units, dense_units)
        self.attention_layer_text_sound = MultiModalAttention(lstm_units, dense_units)
        self.attention_layer_text_sound2 = MultiModalAttention(lstm_units, dense_units)
        self.attention_layer_text_brand = MultiModalAttention(lstm_units, dense_units)
        self.attention_layer_text_creative = MultiModalAttention(lstm_units, dense_units)

        self.dropout_layer = nn.Dropout(0.5)
        self.dense_layer = nn.Sequential(
            nn.Linear(lstm_units + 5 * dense_units, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, dense_units),
        )
        self.output_layer = nn.Linear(dense_units, num_classes)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)

    def forward(self, text_input, img_input, sound_input, sound2_input, brand_input, creative_input):
        text_input = text_input.view(text_input.size(0), -1)
        text_input = text_input.to(torch.int64)
        text_embed = self.text_embedding_layer(text_input)
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)
        text_lstm_out, _ = self.text_lstm_layer(text_embed)
        text_lstm_out = text_lstm_out[:, -1, :]

        img_input = img_input.float()
        img_input = img_input.view(img_input.size(0), -1)
        img_dense = self.img_dense_layer(img_input)

        sound_input = sound_input.float()
        sound_input = sound_input.view(sound_input.size(0), sound_input.size(1), -1)
        sound_lstm_out, _ = self.sound_input_layer(sound_input)
        sound_lstm_out = sound_lstm_out[:, -1, :]

        sound2_input = sound2_input.float()
        sound2_input = sound2_input.view(sound2_input.size(0), sound2_input.size(1), -1)
        sound2_lstm_out, _ = self.sound2_input_layer(sound2_input)
        sound2_lstm_out = sound2_lstm_out[:, -1, :]

        brand_input = brand_input.float()
        brand_dense = self.brand_layer(brand_input)

        creative_input = creative_input.float()
        creative_dense = self.creative_layer(creative_input)

        attended_features_img = self.attention_layer_text_img(text_lstm_out, img_dense)
        attended_features_sound = self.attention_layer_text_sound(text_lstm_out, sound_lstm_out)
        attended_features_sound2 = self.attention_layer_text_sound2(text_lstm_out, sound2_lstm_out)
        attended_features_brand = self.attention_layer_text_brand(text_lstm_out, brand_dense)
        attended_features_creative = self.attention_layer_text_creative(text_lstm_out, creative_dense)

        context_features = torch.cat(
            [text_lstm_out, attended_features_img, attended_features_sound, attended_features_sound2,
             attended_features_brand, attended_features_creative], dim=1)

        dropout = self.dropout_layer(context_features)
        dense = self.dense_layer(dropout)
        output = self.output_layer(dense)

        return output


def normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std


# Custom collate function to handle None values
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    num_samples = 1
    batch_size = 64
    input_length = 384
    vocab_size = 3254
    embedding_dim = 256
    lstm_units = 256
    img_input_shape = (1, 512, 512)
    sound_embedding_dim = 256
    sound2_embedding_dim = 256
    dense_units = 256
    num_classes = 1

    # Your AWS credentials and bucket information
    bucket = 'adsresearch1'
    prefix = 'complete_embeddings'
    aws_access_key_id = 'YOUR_AWS_ACCESS_KEY_ID'
    aws_secret_access_key = 'YOUR_AWS_SECRET_ACCESS_KEY'

    # Create the S3 dataset
    dataset = S3Dataset(bucket, prefix, aws_access_key_id, aws_secret_access_key)

    #     # Split the dataset into training and validation
    #     train_size = int(0.8 * len(dataset))  # 80% for training
    #     valid_size = len(dataset) - train_size
    #     train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    # Create the S3 dataset
    dataset = S3Dataset(bucket, prefix, aws_access_key_id, aws_secret_access_key)

    # Use only a subset of the dataset for quick testing
    total_samples = 1000  # Use 1000 samples for quick testing.
    train_size = int(0.8 * total_samples)  # Use 80% of total_samples for training.

    # Get a list of indices from 0 to len(dataset) - 1
    indices = list(range(len(dataset)))

    # Shuffle the indices
    random.shuffle(indices)

    # Use the first total_samples indices for your subsets
    indices = indices[:total_samples]

    # Create subsets based on the new indices
    subset_sampler_train = SubsetRandomSampler(indices[:train_size])
    subset_sampler_valid = SubsetRandomSampler(indices[train_size:total_samples])

    # Create the data loaders
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=subset_sampler_train,
                                  collate_fn=custom_collate_fn)
    valid_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=subset_sampler_valid,
                                  collate_fn=custom_collate_fn)

    model = MultiModalAttentionModel(vocab_size, embedding_dim, lstm_units, img_input_shape[0] * img_input_shape[1],
                                     sound_embedding_dim, dense_units, num_classes).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Learning Rate Scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 10
    best_valid_loss = float('inf')
    epochs_losses = []
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for i, data in enumerate(train_dataloader):
            if any(elem is None for elem in data.values()):
                print("Skipping batch due to missing data.")
                continue

            # Unpack the data from the batch
            text_inputs = data['text'].to(device)
            img_inputs = normalize(data['video']).to(device)
            sound_inputs = normalize(data['audio']).to(device)
            sound2_inputs = normalize(data['back_audio']).to(device)
            targets = data['targets'].to(device)

            #             print(f"text_inputs shape: {text_inputs.shape}")
            #             print(f"img_inputs shape: {img_inputs.shape}")
            #             print(f"sound_inputs shape: {sound_inputs.shape}")
            #             print(f"sound2_inputs shape: {sound2_inputs.shape}")
            #             print(f"targets shape: {targets.shape}")

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(text_inputs, img_inputs, sound_inputs, sound2_inputs).squeeze(-1)
            loss = criterion(outputs, targets.squeeze(-1).float())
            #             print('loss', loss)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches
        epochs_losses.append(epoch_loss)
        #         torch.onnx.export(model, (text_inputs, img_inputs, sound_inputs, sound2_inputs), "model.onnx")

        # After training, switch to eval mode for validation
        model.eval()
        valid_loss = 0.0
        num_valid_batches = 0
        # Initialize empty lists to store the targets and the outputs
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                if any(elem is None for elem in data.values()):
                    print("Skipping batch due to missing data.")
                    continue

                # Unpack the data from the batch
                text_inputs = data['text'].to(device)
                img_inputs = normalize(data['video']).to(device)
                sound_inputs = normalize(data['audio']).to(device)
                sound2_inputs = normalize(data['back_audio']).to(device)
                targets = data['targets'].to(device)

                outputs = model(text_inputs, img_inputs, sound_inputs, sound2_inputs).squeeze(-1)
                loss = criterion(outputs, targets.squeeze(-1).float())

                # Save the targets and the outputs for the correlation calculation
                all_targets.extend(targets.squeeze(-1).float().tolist())
                all_outputs.extend(outputs.tolist())

                valid_loss += loss.item()
                num_valid_batches += 1

        valid_loss = valid_loss / num_valid_batches
        all_targets_np = np.array(all_targets)
        all_outputs_np = np.array(all_outputs)

        print(f'Number of NaN values in targets: {np.isnan(all_targets_np).sum()}')
        print(f'Number of NaN values in outputs: {np.isnan(all_outputs_np).sum()}')

        print(f'Variability in targets: {np.ptp(all_targets_np)}')
        print(f'Variability in outputs: {np.ptp(all_outputs_np)}')

        unique_targets = np.unique(all_targets_np)
        unique_outputs = np.unique(all_outputs_np)

        print(f'Number of unique targets: {unique_targets.shape[0]}')
        print(f'Number of unique outputs: {unique_outputs.shape[0]}')

        # Calculate Spearman's correlation
        spearman_corr, _ = spearmanr(all_targets, all_outputs)
        print(f'Spearman Correlation: {spearman_corr:.4f}')

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {valid_loss:.4f}, Spearman Correlation: {spearman_corr:.4f}')

        train_losses.append(epoch_loss)  # Add the training loss for this epoch
        valid_losses.append(valid_loss)  # Add the validation loss for this epoch

        # Save the model if it has the best validation loss so far
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # Plot the training and validation loss
        epoch_count = range(1, len(train_losses) + 1)
        plt.plot(epoch_count, train_losses, 'r--')
        plt.plot(epoch_count, valid_losses, 'b-')
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('loss_epochs.png')
