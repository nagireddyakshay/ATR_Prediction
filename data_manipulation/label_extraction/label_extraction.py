#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries

import boto3  # Import the Boto3 library for AWS interactions
import pandas as pd  # Import the Pandas library for data manipulation and analysis

# Connect to AWS S3 and retrieve data files
client = boto3.client('s3')  # Create an AWS S3 client
# Read CSV files from S3 into DataFrames
airing = pd.read_csv('s3://adsresearch1/attentions.csv')  # Read 'attentions.csv' from S3
airing_1 =  pd.read_csv('s3://adsresearch1/attentions-1.csv')  # Read 'attentions-1.csv' from S3

# Combine the two 'airing' DataFrames into one
final_airing = pd.concat([airing, airing_1], axis=0)

# Read 'attention_with_id.csv' and 'attention_with_id-1.csv' from S3 and combine into one DataFrame
attention = pd.read_csv('s3://adsresearch1/attention_with_id.csv')
attention_1 = pd.read_csv('s3://adsresearch1/attention_with_id-1.csv')
final_attention = pd.concat([attention, attention_1], axis=0)

# Delete the individual 'airing' and 'attention' DataFrames to save memory
del airing, airing_1, attention, attention_1

# Merge 'final_airing' and 'final_attention' DataFrames based on 'airing_data_id'
total_attentions = pd.merge(final_airing, final_attention[['airing_data_id', 'attention_data_completed']], on='airing_data_id')

# Filter out rows where 'attention_data_completed' is not NaN
total_attentions = total_attentions[total_attentions['attention_data_completed'].notna()].copy(deep=True)

# Delete 'final_airing' and 'final_attention' DataFrames to save memory
del final_airing, final_attention

# Merge 'sample' DataFrame with 'creatives' DataFrame based on 'creative_data_id'
sample_date = pd.merge(sample, creatives[['creative_data_id', 'creative_data_airing_date_first_et']], on='creative_data_id')

# Filter out rows where 'creative_data_airing_date_first_et' is not NaN
sample_date = sample_date[sample_date['creative_data_airing_date_first_et'].notna()]

# Merge 'sample_date' DataFrame with 'total_attentions' DataFrame based on 'creative_data_id' and 'creative_data_airing_date_first_et'
final_labels = pd.merge(sample_date, total_attentions[["creative_data_id", "airing_data_aired_at_et", "audience_data_impressions_raw", "audience_data_impressions_per_airing", "attention_data_completed"]],
                        left_on=['creative_data_id', 'creative_data_airing_date_first_et'],
                        right_on=['creative_data_id', 'airing_data_aired_at_et'], how='left')

# Filter out rows where 'attention_data_completed' is not NaN
final_labels = final_labels[final_labels['attention_data_completed'].notna()].copy(deep=True)

# Save the resulting DataFrame to 'creatives_with_label.csv' on S3
final_labels.to_csv(r's3://adsresearch1/creatives_with_label.csv')