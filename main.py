# main.py

import torch
from torch import nn
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import DAN, SentimentDatasetEmbedding, SentimentDatasetBPE, BPETokenizer


##################################################
##                Configuration                 ##
##################################################

EMBEDDING_DIM           = 50
EMBEDDING               = f"glove.6B.{EMBEDDING_DIM}d-relativized.txt"
USE_GLOVE               = True
VOCAB_SIZE              = 4096
TOKENIZER_LOAD_PATH     = f'./tokenizers/bpe_{VOCAB_SIZE}.json'
TOKENIZER_SAVE_PATH     = f'./tokenizers/bpe_{VOCAB_SIZE}.json'


##################################################
##                Implementation                ##
##################################################

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float() # We want a LongTensor for the indices

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader, epochs=100):
    # loss_fn = nn.NLLLoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(epochs):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Assert invalid arguments
    assert args.model in ["BOW", "DAN", "SUBWORDDAN"], "Model type must be 'BOW', 'DAN' or 'SUBWORDDAN'"

    # Check if the model type is "BOW"
    if args.model == "BOW":
        
        # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")
        
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        plt.show()

    elif args.model == "DAN":
        
        # Read the GloVe embeddings and sentiment data
        start_time = time.time()
        word_embeddings = read_word_embeddings(f"data/{EMBEDDING}")
        train_examples = read_sentiment_examples("data/train.txt")
        dev_examples = read_sentiment_examples("data/dev.txt")

        # Use the SentimentDatasetEmbedding class to map words to indices
        train_data = SentimentDatasetEmbedding(train_examples, word_embeddings)
        dev_data = SentimentDatasetEmbedding(dev_examples, word_embeddings)

        # Create DataLoader objects
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")

        # Initialize the DAN model
        model = DAN(embedding_dim=EMBEDDING_DIM, use_glove=USE_GLOVE, vocab_size=len(word_embeddings.word_indexer))
        
        # Run experiment with DAN
        train_acc, test_acc = experiment(model, train_loader, test_loader, epochs=args.epochs)
        
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(test_acc, label='Dev Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy for DAN')
        plt.legend()
        plt.grid()
        
        # Save the training accuracy figure
        accuracy_file = 'accuracy_glove_embeddings.png'
        plt.savefig(accuracy_file)
        print(f"\n\nAccuracy plot saved as {accuracy_file}")
        
        plt.show()
    
    elif args.model == "SUBWORDDAN":
        
        assert VOCAB_SIZE > 66, "Vocabulary size must be greater than 66 (number of ASCII characters)"
        
        # Load data
        start_time = time.time()
        
        train_examples = read_sentiment_examples("data/train.txt")
        dev_examples = read_sentiment_examples("data/dev.txt")
        
        # Use the SentimentDatasetBPE class to tokenize the sentences
        # Create the tokenizer with the training data
        tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE, examples=train_examples, load_path=TOKENIZER_LOAD_PATH, save_path=TOKENIZER_SAVE_PATH)

        # Pass the tokenizer into both train and dev datasets
        train_data = SentimentDatasetBPE(train_examples, tokenizer)
        dev_data = SentimentDatasetBPE(dev_examples, tokenizer)
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data loaded in : {elapsed_time} seconds")
        
        # Initialize the DAN model
        model = DAN(embedding_dim=EMBEDDING_DIM, use_glove=False, vocab_size=VOCAB_SIZE)
        
        # Run experiment with DAN
        train_acc, test_acc = experiment(model, train_loader, test_loader, epochs=args.epochs)
        
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(test_acc, label='Dev Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy for DAN (Subword Tokenization)')
        plt.legend()
        plt.grid()
        
        # Save the training accuracy figure
        accuracy_file = 'accuracy_subword.png'
        plt.savefig(accuracy_file)
        print(f"\n\nAccuracy plot saved as {accuracy_file}")
        
        plt.show()


if __name__ == "__main__":
    main()
