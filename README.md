# Sample LLM Coding

## Overview

This project demonstrates a sample project that utilizes a Large Language Model (LLM). The application uses Hugging Face models.

This project consists of two main components:

1. **Sentiment Analysis**: A custom sentiment analysis model using the `DistilBERT` transformer model.
2. **Text Generation**: A text generation pipeline using the `GPT-2` model.

## Requirements

- Python 3.10.x

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/rcw3bb/sample-llm-transformers.git
    cd sample-llm-transformers
    ```

2. Install the required packages:
    ```sh
    poetry install
    ```

## Sentiment Analysis

### Description

The sentiment analysis component uses the `DistilBERT` model to classify the sentiment of poem verses. The model is fine-tuned on a custom dataset.

### Files

- `hf-custom-sentiments.py`: Contains the code for loading the dataset, tokenizing, training the model, and making predictions.

### Usage

1. Run the sentiment analysis script:
    ```sh
    python hf-custom-sentiments.py
    ```

2. The script will:
    - Load the `poem_sentiment` dataset.
    - Tokenize the dataset.
    - Train the `DistilBERT` model.
    - Make predictions on sample verses.

## Text Generation

### Description

The text generation component uses the `GPT-2` model to generate synthetic text based on a given prompt.

### Files

- `hf-text-generation.py`: Contains the code for setting up the text generation pipeline and generating text.

### Usage

1. Run the text generation script:
    ```sh
    python hf-text-generation.py
    ```

2. The script will:
    - Initialize the `GPT-2` text generation pipeline.
    - Generate synthetic text based on the provided prompt.
