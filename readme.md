# ATR Prediction

This repository hosts code and data for predicting Advertisement Attention Time Rate (ATR), utilizing audio video and text embeddings.

## 1. data

Contains the training data stored in an S3 bucket, serving as the source for data manipulation and model training.

## 2. data_manipulation

Houses subfolders for different data manipulation tasks:

- **data_batching**: Involves the organization of data into batches for efficient processing.
- **label_extraction**: Focuses on extracting target labels from the provided dataset.
- **label_fix**: Addresses any issues related to label correctness and alignment.

## 3. train

Reserved for training scripts and related files, designed to utilize the manipulated data and generate models.

## Usage

This repository's folder structure is designed to facilitate efficient data manipulation, label extraction, and model training.
