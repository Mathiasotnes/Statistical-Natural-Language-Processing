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
import re
import collections
import json


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

class SentimentDatasetBPE(Dataset):
    """
    Dataset class that takes in the training data and performs BPE tokenization using a shared Tokenizer.

    Args:
        examples (list): List of SentimentExample objects.
        tokenizer (Tokenizer): Tokenizer for BPE tokenization and indexing.
    """
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max(len(ex.words) for ex in self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns the tokenized sentence and the label for the given example.

        Args:
            idx (int): Index of the example.

        Returns:
            torch.LongTensor: Tensor of tokenized sentence.
            torch.Tensor: Tensor of label (0 or 1).
        """
        example = self.examples[idx]
        tokenized_sentence = self.tokenizer.tokenize(example.words)
        label = example.label

        # Pad or truncate the indices to the fixed length (self.max_len)
        if len(tokenized_sentence) < self.max_len:
            tokenized_sentence += [0] * (self.max_len - len(tokenized_sentence))
        else:
            tokenized_sentence = tokenized_sentence[:self.max_len]

        # Convert tokenized sentence to indices
        tokenized_sentence = self.tokenizer.get_indices(tokenized_sentence)

        return torch.LongTensor(tokenized_sentence), label
    
class BPETokenizer:
    """
    Handles BPE tokenization and vocabulary management.
    """
    def __init__(self, vocab_size, examples, load_path=None, save_path=None):
        self.vocab_size = vocab_size
        self.indexer = Indexer()
        self.merged_cache = {}
        
        # Store all the training text data as a list of tokenized sentences (split by characters)
        self.text_data = self._initialize_text_data(examples)
        
        # Build initial vocabulary based on characters
        self.vocab = self._build_initial_char_vocab(self.text_data)
        initial_vocab_size = len(self.vocab)
        
        if load_path:
            load_vocab = self._load_vocab(load_path)
            if not load_vocab:
                print(f"Could not load vocabulary from {load_path}. Building new vocabulary...")
                num_merges = self.vocab_size - initial_vocab_size
                self._perform_bpe(num_merges)
                self._save_vocab(load_path)
            else:
                self.vocab = load_vocab
        else:
            # Calculate the number of merges required to reach the desired vocabulary size
            num_merges = self.vocab_size - initial_vocab_size
            self._perform_bpe(num_merges)
            if save_path:
                self._save_vocab(save_path)
        
        print("Vocabulary size:", len(self.vocab))
        
        # Add the UNK token to the indexer
        self.indexer.add_and_get_index("UNK", add=True)

        # Add the tokens from the final vocabulary to the indexer
        for token in self.vocab.keys():
            self.indexer.add_and_get_index(token, add=True)
        
    def _load_vocab(self, load_path):
        try:
            with open(load_path, 'r') as f:
                vocab = json.load(f)
            print(f"Vocabulary loaded from {load_path}")
            return vocab
        except FileNotFoundError:
            return None
    
    def _save_vocab(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.vocab, f)
        print(f"Vocabulary saved to {save_path}")

    def _initialize_text_data(self, examples):
        """
        Store all the text data as tokenized words.
        Each word is split into characters with a space between them and an end-of-word token.
        """
        text_data = []
        for example in examples:
            sentence = example.words
            tokenized_sentence = []
            for word in sentence:
                tokenized_word = ' '.join(list(word)) + ' </w>'  # Split into chars and add end-of-word token
                tokenized_sentence.append(tokenized_word)
            text_data.append(tokenized_sentence)
        return text_data

    def _build_initial_char_vocab(self, text_data):
        """Build the initial vocabulary based on unique characters in the text data."""
        vocab = collections.defaultdict(int)
        for sentence in text_data:
            for word in sentence:
                chars = word.split()
                for char in chars:
                    vocab[char] += 1
        return vocab

    def _get_stats(self):
        """Count frequency of pairs of consecutive symbols."""
        pairs = collections.defaultdict(int)
        for sentence in self.text_data:
            for word in sentence:
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i + 1]] += 1
        return pairs

    def _merge_vocab(self, pair):
        """Merge the most frequent pair into a new symbol in both the vocab and the text data."""
        bigram = re.escape(' '.join(pair))  # Escape any special characters in the bigram
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')  # Look for the pair as a whole word
        merged_token = ''.join(pair)

        # Update the vocabulary
        self.vocab[merged_token] = 1

        # Update the stored text data
        for i, sentence in enumerate(self.text_data):
            updated_sentence = []
            for word in sentence:
                word_out = p.sub(''.join(pair), word)
                updated_sentence.append(word_out)
            self.text_data[i] = updated_sentence
    
    def _perform_bpe(self, num_merges):
        """Perform BPE merging until the desired vocabulary size is reached."""
        for i in range(num_merges):
            pairs = self._get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self._merge_vocab(best_pair)
            if (i+1) % 100 == 0:
                print(f"Merged {i+1} tokens")

    def tokenize(self, sentence):
        """Tokenizes a sentence using the learned BPE vocabulary."""
        tokenized = []
        for word in sentence:
            chars = list(word) + ['</w>']  # Add end-of-word token
            subword = chars[0]  # Start with the first character
            subword_tokens = []

            for char in chars[1:]:
                # Try to find the longest match of subwords in the vocab
                candidate = subword + char
                if candidate in self.vocab:
                    subword = candidate  # Continue building the subword
                else:
                    subword_tokens.append(subword)  # Add the existing subword to tokens
                    subword = char  # Start a new subword

            subword_tokens.append(subword)  # Add the final subword

            # Add the tokenized subwords to the final list, check against the vocabulary
            for token in subword_tokens:
                if token in self.vocab:
                    tokenized.append(token)
                else:
                    tokenized.append('UNK')  # Use UNK for unknown subwords

        return tokenized

    def get_indices(self, tokenized_sentence):
        """Converts a tokenized sentence into indices using the indexer."""
        return [self.indexer.index_of(token) if self.indexer.contains(token) else self.indexer.index_of("UNK")
                for token in tokenized_sentence]


class DAN(nn.Module):
    """
    Deep Averaging Network (DAN) for sentiment classification. The model converts word indices
    into embeddings, averages them, and passes through fully connected layers to classify sentiment.

    Args:
        embedding_dim (int): Dimension of the word embeddings (50 or 300).
        use_glove (bool): Whether to use pre-trained GloVe embeddings.
        vocab_size (int): Size of the vocabulary (default: 512).
    """
    def __init__(self, embedding_dim, use_glove=True, vocab_size=512):
        super(DAN, self).__init__()

        # Convert embedding dimension to a number if passed as "50d" or "300d"
        if use_glove:
            if embedding_dim == 50:
                self.embedding_dim = 50
                self.glove_file = "./data/glove.6B.50d-relativized.txt"
            elif embedding_dim == 300:
                self.embedding_dim = 300
                self.glove_file = "./data/glove.6B.300d-relativized.txt"
            else:
                raise ValueError(f"Unsupported embedding dimension: {embedding_dim}. Choose 50 or 300.")
            word_embeddings = read_word_embeddings(self.glove_file)
            self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=True)
        # Use a random embedding layer
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(vocab_size+1, embedding_dim)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(self.embedding_dim, 100)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 500)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(500, 2)
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
        # print("x max", x.max())
        # print("x min", x.min())
        embeddings = self.embedding(x)

        # Create a mask where padding (0) is 0, and actual tokens are 1
        mask = (x != 0).float()

        # Calculate the sum of embeddings along the sentence and divide by the number of valid tokens
        # This excludes the padding from contributing to the average
        sum_embeddings = torch.sum(embeddings * mask.unsqueeze(-1), dim=1)
        valid_tokens = torch.sum(mask, dim=1).unsqueeze(-1)
        avg_embeddings = sum_embeddings / valid_tokens

        # Pass through the fully connected layers
        x = F.relu(self.fc1(avg_embeddings))
        x = self.do1(x)
        x = F.relu(self.fc2(x))
        x = self.do2(x)
        x = F.relu(self.fc3(x))
        x = self.log_softmax(x)
        return x

# Byte Pair Encoding (BPE) functions
def get_stats(vocab: dict) -> dict:
    """Count frequency of pairs of consecutive symbols."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in) -> dict:
    """Merge the most frequent pair into a new symbol."""
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def bpe(vocab: dict, num_merges: int) -> dict:
    """Perform num_merges iterations of BPE."""
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab