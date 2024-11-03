#########################################################
## CSE 256 - Statistical Natural Language Processing   ##
## Transformer assignment (PA2)                        ##
## --------------------------------------------------- ##
## Author:   Mathias Otnes                             ##
## Date:     2024-11-06                                ##
#########################################################

#######################
## Libraries

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import time
from itertools import cycle
import matplotlib.pyplot as plt
import argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from utilities import Utilities
from transformer import CLSModel, Encoder, Decoder
from disentangled_attention import DisentangledDecoder

#######################
## Constants

seed = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() 
    # else "mps" if torch.backends.mps.is_available() # Metal Performance Shaders (MPS) on macOS
    else "cpu"
)

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 50  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training


#######################
## Implementation

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ 
    Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        pred, loss = decoderLMmodel(X, Y)
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def train_classifier(classifier, train_loader, test_loader, epochs, plot=False):
    """ Train the classifier on the train_loader and evaluate on the test_loader. """
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = classifier(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        train_accuracy = compute_classifier_accuracy(classifier, train_loader)
        test_accuracy = compute_classifier_accuracy(classifier, test_loader)
        print(f"Epoch {epoch + 1}: Train accuracy: {train_accuracy:.2f}%, Test accuracy: {test_accuracy:.2f}%")
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    if plot:
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Test accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Classifier Training")
        plt.grid(which='both')
        plt.show()
    return classifier

def train_language_model(decoderLMmodel, train_loader, plot=False):
    """Trains the language model on the train_loader and evaluates on the test_loader."""
    optimizer = torch.optim.Adam(decoderLMmodel.parameters(), lr=learning_rate)
    decoderLMmodel.train()
    total_loss = 0
    train_loader_iter = cycle(train_loader)  # Infinite iterator to keep training until max_iters
    perplexities = []
    for i in range(max_iters+1):
        xb, yb = next(train_loader_iter)
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, loss = decoderLMmodel(xb, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % eval_interval == 0 and i > 0:
            perplexity = compute_perplexity(decoderLMmodel, train_loader, eval_iters)
            perplexities.append(perplexity)
            print(f"Iteration {i}: Perplexity: {perplexity:.2f}")
            total_loss = 0
    if plot:
        plt.plot(perplexities)
        plt.xlabel("Iteration")
        plt.ylabel("Perplexity")
        plt.title("Language Model Training (Obama)")
        plt.grid(which='both')
        plt.show()
    return decoderLMmodel

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run different parts of the NLP models: classifier, language model, and disentangled language model."
    )
    
    parser.add_argument(
        '--part1',
        action='store_true',
        help='Run the classifier model.'
    )
    
    parser.add_argument(
        '--part2',
        action='store_true',
        help='Run the language model.'
    )
    
    parser.add_argument(
        '--part3',
        action='store_true',
        help='Run the disentangled language model.'
    )
    
    # Optional arguments for LM sub-configurations
    parser.add_argument(
        '--train-lm',
        action='store_true',
        help='Train the language model.'
    )
    
    parser.add_argument(
        '--obama',
        action='store_true',
        help='Run LM on Obama test set.'
    )
    
    parser.add_argument(
        '--wbush',
        action='store_true',
        help='Run LM on W. Bush test set.'
    )
    
    parser.add_argument(
        '--hbush',
        action='store_true',
        help='Run LM on H. Bush test set.'
    )
    
    # Optional arguments for verbosity and plotting
    parser.add_argument(
        '--show_plot',
        action='store_true',
        help='Plot training results.'
    )
    
    parser.add_argument(
        '--save_plot',
        action='store_true',
        help='Plot training results.'
    )
    
    return parser.parse_args()

#######################
## Main program entry

def main(args):
    
    #######################
    ## Configuration 
    
    # Models to run
    classifier  = args.part1
    LMmodel     = args.part2
    DELMmodel   = args.part3
    
    # LMmodel sub-configurations
    LMtrain     = args.train_lm
    obama       = args.obama
    wbush       = args.wbush
    hbush       = args.hbush
    
    # Train all models if no specific sub-configurations are set
    if not (obama or wbush or hbush):
        LMtrain = True
        obama   = True
        wbush   = True
        hbush   = True
    
    # Plotting
    show_plot   = args.show_plot
    save_plot   = args.save_plot
    
    
    #######################
    ## Main program
    
    print(f"\r\n{'='*40}")
    print("| Transformer Blocks (PA2) - CSE 256   |")
    print(f"{'='*40}\r\n")
    
    print("Using device:", device, end="\r\n\n")
    torch.manual_seed(seed)
    
    t0 = time.time()
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size for CLS is: ", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)
  
    trainLMTextFile = "speechesdataset/train_LM.txt"
    testLMObamaTextFile = "speechesdataset/test_LM_obama.txt"
    testLMWBushTextFile = "speechesdataset/test_LM_wbush.txt"
    testLMHBushTextFile = "speechesdataset/test_LM_hbush.txt"
    
    with open(trainLMTextFile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    with open(testLMObamaTextFile, 'r', encoding='utf-8') as f:
        lmtestObamaText = f.read()
    with open(testLMWBushTextFile, 'r', encoding='utf-8') as f:
        lmtestWBushText = f.read()
    with open(testLMHBushTextFile, 'r', encoding='utf-8') as f:
        lmtestHBushText = f.read()
        
    train_LM_dataset        = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    test_LM_obama_dataset   = LanguageModelingDataset(tokenizer, lmtestObamaText, block_size)
    test_LM_wbush_dataset   = LanguageModelingDataset(tokenizer, lmtestWBushText, block_size)
    test_LM_hbush_dataset   = LanguageModelingDataset(tokenizer, lmtestHBushText, block_size)
    train_LM_loader         = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    test_LM_obama_loader    = DataLoader(test_LM_obama_dataset, batch_size=batch_size, shuffle=False)
    test_LM_wbush_loader    = DataLoader(test_LM_wbush_dataset, batch_size=batch_size, shuffle=False)
    test_LM_hbush_loader    = DataLoader(test_LM_hbush_dataset, batch_size=batch_size, shuffle=False)
    
    t_data = time.time() - t0
    print(f"Successfully loaded data in {t_data:.3f} seconds")
    
    
    #######################
    ## CLS Model
    
    if classifier:
        print(f"\r\n{'='*40}")
        print("Classifier Model (CLS)")
        print(f"{'='*40}\r\n")
        
        encoder = Encoder(
            vocab_size=tokenizer.vocab_size,
            d_model=n_embd,
            num_heads=n_head,
            num_blocks=n_layer,
            hidden_dim=n_hidden,
            dropout=0.0,
            echo_specs=True
        ).to(device)
        
        cls_model = CLSModel(
            encoder=encoder,
            n_hidden=n_hidden,
            num_classes=n_output,
            echo_specs=True
        ).to(device)
        
        train_classifier(cls_model, train_CLS_loader, test_CLS_loader, epochs_CLS, plot=show_plot)
        
        # Sanity check
        utils = Utilities(tokenizer, cls_model)
        utils.sanity_check("And that is why I intend to personally pursue this outcome with all the patience and dedication that the task requires.", block_size, show_plots=show_plot, save_plots=save_plot)

    
    #######################
    ## Language Model
    
    if LMmodel:
        
        if LMtrain:
            print(f"\r\n{'='*40}")
            print("Training Language Model (LM)")
            print(f"{'='*40}\r\n")
            decoder = Decoder(
                vocab_size=tokenizer.vocab_size,
                d_model=n_embd,
                num_heads=n_head,
                num_blocks=n_layer,
                hidden_dim=100,
                dropout=0.0,
                echo_specs=True
            ).to(device)
            train_language_model(decoder, train_LM_loader)
            
            # Sanity check
            utils = Utilities(tokenizer, decoder)
            utils.sanity_check("The quick brown fox eats the lazy dog", block_size, show_plots=show_plot, save_plots=save_plot)
        
        if obama:
            print(f"\r\n{'='*40}")
            print("Obama model")
            print(f"{'='*40}\r\n")
            decoder = Decoder(
                vocab_size=tokenizer.vocab_size,
                d_model=n_embd,
                num_heads=n_head,
                num_blocks=n_layer,
                hidden_dim=100,
                dropout=0.0,
                echo_specs=True
            ).to(device)
            print(f"\r\n{'='*40}")
            print("Training on Obama test set...")
            print(f"{'='*40}\r\n")
            train_language_model(decoder, test_LM_obama_loader)
            
            # Sanity check
            utils = Utilities(tokenizer, decoder)
            utils.sanity_check("America, we weaken those ties when we allow our political dialogue to become so corrosive that people of good character aren't even willing to enter into public service; so coarse with rancor that Americans with whom we disagree are seen not just as misguided, but as malevolent.", block_size, show_plots=show_plot, save_plots=save_plot)
        
        if wbush:
            print(f"\r\n{'='*40}")
            print("W. Bush model")
            print(f"{'='*40}\r\n")
            decoder = Decoder(
                vocab_size=tokenizer.vocab_size,
                d_model=n_embd,
                num_heads=n_head,
                num_blocks=n_layer,
                hidden_dim=100,
                dropout=0.0,
                echo_specs=True
            ).to(device)
            print(f"\r\n{'='*40}")
            print("Training on W. Bush test set...")
            print(f"{'='*40}\r\n")
            train_language_model(decoder, test_LM_wbush_loader)
            
            # Sanity check
            utils = Utilities(tokenizer, decoder)
            utils.sanity_check("America, we weaken those ties when we allow our political dialogue to become so corrosive that people of good character aren't even willing to enter into public service; so coarse with rancor that Americans with whom we disagree are seen not just as misguided, but as malevolent.", block_size, show_plots=show_plot, save_plots=save_plot)
        
        if hbush:
            print(f"\r\n{'='*40}")
            print("H. Bush model")
            print(f"{'='*40}\r\n")
            decoder = Decoder(
                vocab_size=tokenizer.vocab_size,
                d_model=n_embd,
                num_heads=n_head,
                num_blocks=n_layer,
                hidden_dim=100,
                dropout=0.0,
                echo_specs=True
            ).to(device)
            print(f"\r\n{'='*40}")
            print("Training on H. Bush test set...")
            print(f"{'='*40}\r\n")
            train_language_model(decoder, test_LM_hbush_loader)
            
            # Sanity check
            utils = Utilities(tokenizer, decoder)
            utils.sanity_check("America, we weaken those ties when we allow our political dialogue to become so corrosive that people of good character aren't even willing to enter into public service; so coarse with rancor that Americans with whom we disagree are seen not just as misguided, but as malevolent.", block_size, show_plots=show_plot, save_plots=save_plot)
    
    if DELMmodel:
        print(f"\r\n{'='*40}")
        print("Disentangled Language Model (DELM)")
        print(f"{'='*40}\r\n")
        
        decoder = DisentangledDecoder(
            vocab_size=tokenizer.vocab_size,
            d_model=n_embd,
            num_heads=n_head,
            num_blocks=n_layer,
            hidden_dim=100,
            dropout=0.0,
            echo_specs=True
        ).to(device)
        
        train_language_model(decoder, train_LM_loader)
        
        # Sanity check
        utils = Utilities(tokenizer, decoder)
        utils.sanity_check("The quick brown fox eats the lazy dog", block_size, show_plots=show_plot, save_plots=save_plot)

    t_end = time.time() - t0
    print(f"\r\nProgram executed in {t_end:.3f} seconds\r\n")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    exit(0)
