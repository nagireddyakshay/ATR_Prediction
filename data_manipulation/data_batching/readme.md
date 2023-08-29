# TuneIn Dataset Batching

This code processes audio, video and text embeddings for feeding it into training module. It includes a custom dataset class, data processing methods, and AWS S3 integration for efficient data handling.

**1. Import Libraries**

The script begins by importing essential libraries required for its execution. These include libraries for data manipulation (`Pandas`), deep learning (`torch.nn`), file handling (`os`, `pickle`), AWS interaction (`boto3`), and error handling (`traceback`).

**2. Clear CUDA Memory and Device Selection**

The code clears the CUDA memory to ensure a clean start and then determines whether a GPU (`cuda`) or CPU (`cpu`) is available. The selected device is used for subsequent computations.

**3. Audio Paths and Labels**

The script generates a list of audio paths for embeddings using a loop. It reads label information from a CSV file called `labels.csv`.

**4. Define Custom Modules**

The code defines two custom PyTorch modules: `Flatten` and `ReshapeTo256`. These modules are later utilized within the neural network.

**4.1 `Flatten` Module**

The `Flatten` module is defined to reshape the input tensor from a higher-dimensional shape into a 1D tensor, maintaining the batch size. It is utilized in the `ReshapeTo256` module to prepare the input data for a fully connected layer.

**4.2 `ReshapeTo256` Module**

The `ReshapeTo256` module is designed to reshape the input tensor to match the required input size for the subsequent fully connected layer. It initializes the necessary components within its `forward` method. Specifically:
- Upon its first usage, the module determines the `input_size` based on the shape of the input tensor.
- It initializes a fully connected (`linear`) layer with output size 256 and sends it to the appropriate device (GPU or CPU).
- The `forward` method receives an input tensor and:
  - Uses the `Flatten` module to reshape the input tensor.
  - Passes the flattened tensor through the fully connected layer, resulting in a tensor of size `[batch_size, 256]`.

**5. Custom Dataset Class (TuneInDataset)**

This class represents the core of the code, managing data preparation and processing for machine learning. It consists of several methods:

**5.1 Initialization**

- The `__init__` method initializes various attributes:
  - `num_samples`: The total number of samples in the dataset.
  - `input_length`: The length of input data (e.g., text or audio features).
  - `vocab_size`: The vocabulary size for text data.
  - `img_input_shape`: Shape of input images (video frames).
  - `sound_embedding_dim`: Dimensionality of sound embeddings.
  - `sound2_embedding_dim`: Dimensionality of background sound embeddings.
  - `s3_bucket`: AWS S3 bucket name.
  - `audio_folder`: Folder containing audio embeddings.
  - `audio_pickle_paths`: List of audio embedding paths.
  - `text_embeddings`: Loaded preprocessed text embeddings.
  - `video_embeddings`: Loaded preprocessed video embeddings.
  - `targets`: Tensor of target labels.
  - `segment_size`: Number of samples processed per segment.
  - `current_segment`: Current segment being processed.
  - `audio`, `back_audio`, `video`, `text`: Lists to store processed embeddings.
  - AWS credentials (`aws_access_key_id`, `aws_secret_access_key`).

**5.2 Process Segment**

The `process_segment` method processes audio and video embeddings for a specific segment. It handles downloading audio embedding files from AWS S3, assigning random embeddings for missing data, and processing audio and background audio features.

**5.3 has_next_segment**

The `has_next_segment` method checks whether more segments are available for processing, based on the current segment index.

**5.4 prepare_batches**

The `prepare_batches` method iteratively calls `process_segment` to process data and prepares batches for training. It packages video, text, audio, background audio, and target data into batches and saves them to AWS S3.

**5.5 save_batch**

The `save_batch` method serializes and saves processed batch data to AWS S3, organized under the `latest_embeddings/` directory.

**5.6 \_\_len\_\_ and \_\_getitem\_\_**

The `__len__` method returns the total number of samples in the dataset, while the `__getitem__` method provides access to individual samples by index.

**6. Main Function**

The `main` function encapsulates the core execution of the script. It defines hyperparameters, S3 bucket details, and audio paths, creates the dataset instance, and loads it into a DataLoader for training.

**7. Entry Point**

The script's entry point is checked using `if __name__ == '__main__':`. If the script is executed directly, the `main` function is called.

**Inputs and Outputs**

- **Text and Video Embeddings**: Loaded from `rf_text_embeddings.pkl` and `rf_video_embeddings.pkl`.
- **Audio Embedding Files**: Downloaded from AWS S3 using provided URLs.
- **Label Information**: Loaded from the CSV file `labels.csv`.
- **Audio Batch Indexes**: Utilized from `sound_indexes.csv` to map videos to audio batches.
- **AWS S3 Access Keys and Bucket Details**: Configured in the code.
- **Processed Batches of Data**: Stored as Pickle files in the specified S3 location (`latest_embeddings/`).
- **Log of Missing Embeddings**: A CSV file named `missed_embeddings.csv` records videos without audio or background audio embeddings.

**Locations**

- `rf_text_embeddings.pkl` and `rf_video_embeddings.pkl`: These files should be in the current working directory.
- `labels.csv`: Should be in the current working directory.
- Audio embedding files: Downloaded from AWS S3 using provided URLs.
- `missed_embeddings.csv`: Created in the current working directory to log missing embeddings.
- Processed batch data: Saved to AWS S3 under the `latest_embeddings/` folder.

