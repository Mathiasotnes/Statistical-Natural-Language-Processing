##############################################################
## CSE 256                                                  ##
## Deep Averaging Network (DAN) model                       ##
## -------------------------------------------------------- ##
## Author: Mathias Otnes                                    ##
## -------------------------------------------------------- ##
## Description:                                             ##                   
##   Implementation of the Deep Averaging Network (DAN)     ##
##   model for sentiment classification. The model          ##
##   averages word embeddings and passes them through       ##
##   dense layers to classify sentiment.                    ##
##############################################################

import torch
from torch import nn
from torch.utils.data import Dataset
from utils import Indexer
import torch.nn.functional as F
from sentiment_data import read_word_embeddings


##############################################################
##                      Configuration                       ##
##############################################################


##############################################################
##                      Implementation                      ##
##############################################################

# Dataset class for sentiment examples (with word indices)
class SentimentDatasetEmbedding(Dataset):
    """
    Dataset class that converts sentences into word indices using pre-trained word embeddings
    and pads each sequence to the length of the longest sentence.

    Args:
        examples (list): List of SentimentExample objects.
        word_embeddings (WordEmbeddings): Object that contains a word indexer and pre-trained embeddings.
    """
    def __init__(self, examples, word_embeddings):
        self.examples = examples
        self.word_embeddings = word_embeddings

        # Find the maximum sentence length in the dataset
        self.max_len = max(len(ex.words) for ex in self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns the padded word embeddings and the label for the given index.

        Args:
            idx (int): Index of the example.

        Returns:
            torch.LongTensor: Padded tensor of word embeddings.
            torch.Tensor: Tensor of label (0 or 1).
        """
        sentiment_example = self.examples[idx]
        sentence = sentiment_example.words
        label = sentiment_example.label

        # Ensure UNK token is used when the word is not found in the indexer
        unk_idx = self.word_embeddings.word_indexer.index_of("UNK")
        indices = [self.word_embeddings.word_indexer.add_and_get_index(word, add=False) for word in sentence]
        
        # Replace -1 (unknown word) with the UNK index
        indices = [i if i != -1 else unk_idx for i in indices]

        # Pad or truncate the indices to the fixed length (self.max_len)
        if len(indices) < self.max_len:
            # Pad the sequence with 0s (assumed to be the PAD token)
            padded_indices = indices + [0] * (self.max_len - len(indices))
        else:
            # Truncate the sequence to max_len
            padded_indices = indices[:self.max_len]

        return torch.LongTensor(padded_indices), torch.tensor(label)


class DAN(nn.Module):
    """
    Deep Averaging Network (DAN) for sentiment classification. The model converts word indices
    into embeddings, averages them, and passes through fully connected layers to classify sentiment.

    Args:
        embedding_dim (int): Dimension of the word embeddings (50 or 300).
        hidden_size (int): Size of the hidden layer in the fully connected network.
    """
    def __init__(self, embedding_dim, hidden_size):
        super(DAN, self).__init__()

        # Convert embedding dimension to a number if passed as "50d" or "300d"
        if embedding_dim == 50:
            self.embedding_dim = 50
            self.glove_file = "./data/glove.6B.50d-relativized.txt"
        elif embedding_dim == 300:
            self.embedding_dim = 300
            self.glove_file = "./data/glove.6B.300d-relativized.txt"
        else:
            raise ValueError(f"Unsupported embedding dimension: {embedding_dim}. Choose 50 or 300.")

        # Load the word embeddings using the provided function from sentiment_data
        word_embeddings = read_word_embeddings(self.glove_file)
        
        # Initialize the embedding layer with GloVe vectors (frozen by default)
        self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=True)
        
        # Define the fully connected layers
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, 2)  # Output layer (binary classification: positive/negative)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the model. Receives word indices, converts them to embeddings,
        averages them (ignoring padding), and passes through fully connected layers.

        Args:
            x (torch.Tensor): Input tensor of word indices.

        Returns:
            torch.Tensor: Log probabilities for each class (0 or 1).
        """
        # Look up word embeddings
        embeddings = self.embedding(x)

        # Create a mask where padding (0) is 0, and actual tokens are 1
        mask = (x != 0).float()

        # Calculate the sum of embeddings along the sentence and divide by the number of valid tokens
        # This excludes the padding from contributing to the average
        sum_embeddings = torch.sum(embeddings * mask.unsqueeze(-1), dim=1)
        valid_tokens = torch.sum(mask, dim=1).unsqueeze(-1)
        avg_embeddings = sum_embeddings / valid_tokens

        # Pass through the fully connected layers
        x = self.dropout1(avg_embeddings)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = self.fc3(x)
        return self.log_softmax(x)