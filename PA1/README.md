# Sentiment Classification using Deep Averaging Networks and Byte Pair Encoding

This project implements sentiment classification on a dataset of movie review snippets. It provides two models for the task:
1. **Deep Averaging Network (DAN)**: A model that averages word embeddings and passes them through fully connected layers for classification.
2. **Subword DAN (SUBWORDDAN)**: A variant of DAN using Byte Pair Encoding (BPE) for subword tokenization.

## Prerequisites

This project requires Python 3.6 or later and the following libraries:
- `torch`
- `argparse`
- `matplotlib`

## Dataset

The dataset consists of movie reviews labeled as positive (1) or negative (0). It is split into a training set and a development set. Sentences are pre-tokenized but not lowercased.

## Models

### 1. Deep Averaging Network (DAN)
The DAN model averages word embeddings (GloVe or randomly initialized) and passes the averaged vector through dense layers. GloVe embeddings (either 50d or 300d) are used if available. To change the embeddings used, specify `50` or `300` in the configuration variable `EMBEDDING_DIM` in `main.py`.

### 2. Subword DAN (SUBWORDDAN)
In this model, Byte Pair Encoding (BPE) is used for tokenizing sentences into subword units, followed by the same structure as the DAN model. This enables the model to work with subwords. To specify the vocabulary size, change the variable `VOCAB_SIZE` in `main.py`. Note that this must be greater than 65, because this is the number of distinct ASCII characters in the vocabulary. 

## Usage

To run the sentiment classification task, use the following command format:

```bash
python main.py --model <MODEL_TYPE> --epochs <NUMBER_OF_EPOCHS>
```

- `MODEL_TYPE`: Choose between `BOW`, `DAN`, or `SUBWORDDAN`.
- `NUMBER_OF_EPOCHS`: Number of epochs for training (default: 10).

### Examples:

1. Train and evaluate the Deep Averaging Network (DAN) with GloVe embeddings:

```bash
python main.py --model DAN
```

2. Train and evaluate the Subword-based DAN (SUBWORDDAN) with BPE tokenization:

```bash
python main.py --model SUBWORDDAN
```

### Visualization:
After training, the script will generate and save plots for training and development accuracy to monitor performance over time.

## Byte Pair Encoding (BPE)
The BPE tokenizer is implemented to iteratively merge frequent character pairs into subwords, producing a compact vocabulary. This approach helps deal with out-of-vocabulary words and improves the efficiency of the model.

## Experimentation
You can experiment with:
- **Embedding dimensions**: Choose between 50d or 300d GloVe embeddings for DAN.
- **Vocabulary size**: Adjust the BPE vocabulary size for the `SUBWORDDAN` model by modifying `VOCAB_SIZE` in `main.py`.

## Output
- Training and development accuracy is printed during training.
- Accuracy plots are saved as PNG files after training.

## Conda Environment Setup
To replicate the environment used in this project, follow the steps below to export the current conda environment and install it from `venv.yaml`:

1. **Install the environment**:

Run the following command to create the environment from the `venv.yaml` file:

```bash
conda env create -f venv.yaml
```

2. **Activate the environment**:

```bash
conda activate CSE256
```