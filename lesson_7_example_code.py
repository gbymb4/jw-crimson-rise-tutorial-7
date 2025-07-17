# -*- coding: utf-8 -*-
"""
LSTM Sentiment Analysis with IMDB Movie Reviews Dataset
Created on Thu Jul 17 14:25:02 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
import re
import numpy as np

# For IMDB dataset
try:
    import torchtext
    from torchtext.datasets import IMDB
    from torchtext.data.utils import get_tokenizer
    HAS_TORCHTEXT = True
except ImportError:
    HAS_TORCHTEXT = False

# Alternative: Manual IMDB dataset loading
import os
import tarfile
import urllib.request
from pathlib import Path

def download_imdb_dataset(data_dir="./imdb_data"):
    """Download and extract IMDB dataset manually"""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = data_dir / "aclImdb_v1.tar.gz"
    extract_path = data_dir / "aclImdb"
    
    if not extract_path.exists():
        print("Downloading IMDB dataset...")
        urllib.request.urlretrieve(url, tar_path)
        
        print("Extracting dataset...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        # Clean up
        tar_path.unlink()
    
    return extract_path

def load_imdb_data(data_dir, split='train', max_samples=None):
    """Load IMDB data from directory structure"""
    reviews = []
    labels = []
    
    # Load positive reviews
    pos_dir = data_dir / split / 'pos'
    if pos_dir.exists():
        for i, file_path in enumerate(pos_dir.glob('*.txt')):
            if max_samples and i >= max_samples // 2:
                break
            with open(file_path, 'r', encoding='utf-8') as f:
                review = f.read()
                reviews.append(review)
                labels.append(1)
    
    # Load negative reviews
    neg_dir = data_dir / split / 'neg'
    if neg_dir.exists():
        for i, file_path in enumerate(neg_dir.glob('*.txt')):
            if max_samples and i >= max_samples // 2:
                break
            with open(file_path, 'r', encoding='utf-8') as f:
                review = f.read()
                reviews.append(review)
                labels.append(0)
    
    # Shuffle the data
    combined = list(zip(reviews, labels))
    np.random.shuffle(combined)
    reviews, labels = zip(*combined)
    
    return list(reviews), list(labels)

def get_imdb_dataset(max_samples=5000):
    """Get IMDB dataset with fallback options"""
    
    # Option 1: Try torchtext (if available)
    if HAS_TORCHTEXT:
        try:
            print("Loading IMDB dataset using torchtext...")
            train_iter = IMDB(split='train')
            test_iter = IMDB(split='test')
            
            train_data = []
            test_data = []
            
            # Convert iterators to lists (limited samples for demo)
            print("Processing training data...")
            for i, (label, text) in enumerate(train_iter):
                if i >= max_samples:
                    break
                # Convert label: 1 = negative, 2 = positive in torchtext
                sentiment = 1 if label == 2 else 0
                train_data.append((text, sentiment))
            
            print("Processing test data...")
            for i, (label, text) in enumerate(test_iter):
                if i >= max_samples // 5:  # Smaller test set
                    break
                sentiment = 1 if label == 2 else 0
                test_data.append((text, sentiment))
            
            return train_data, test_data
            
        except Exception as e:
            print(f"Torchtext failed: {e}")
            print("Falling back to manual download...")
    
    # Option 2: Manual download
    try:
        data_dir = download_imdb_dataset()
        
        train_reviews, train_labels = load_imdb_data(data_dir, 'train', max_samples)
        test_reviews, test_labels = load_imdb_data(data_dir, 'test', max_samples // 5)
        
        train_data = list(zip(train_reviews, train_labels))
        test_data = list(zip(test_reviews, test_labels))
        
        return train_data, test_data
        
    except Exception as e:
        print(f"Manual download failed: {e}")
        print("Using fallback sample data...")
        
        # Option 3: Fallback to sample data
        return get_sample_movie_reviews()

def get_sample_movie_reviews():
    """Fallback sample data if IMDB download fails"""
    reviews = [
        # Positive reviews
        ("This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.", 1),
        ("I loved every minute of this film. The cinematography was beautiful and the story was compelling.", 1),
        ("Brilliant performance by the lead actor. The movie had great pacing and an excellent soundtrack.", 1),
        ("One of the best movies I've seen this year. Highly recommend to anyone who loves good cinema.", 1),
        ("The film exceeded all my expectations. Great direction and wonderful character development.", 1),
        ("Amazing movie with incredible special effects. The story was both touching and thrilling.", 1),
        ("This is a masterpiece of modern cinema. The plot was complex but beautifully executed.", 1),
        ("I was blown away by this film. Every scene was crafted with care and attention to detail.", 1),
        ("The movie was entertaining from start to finish. Great acting and a well-written script.", 1),
        ("Absolutely loved this movie. The ending was perfect and tied everything together nicely.", 1),
        
        # Negative reviews
        ("This movie was terrible. The acting was poor and the plot made no sense whatsoever.", 0),
        ("I hated this film. It was boring and the characters were completely unbelievable.", 0),
        ("Worst movie I've ever seen. The dialogue was awful and the story was confusing.", 0),
        ("This film was a complete waste of time. Poor direction and terrible special effects.", 0),
        ("The movie was disappointing. I expected much better from such a renowned director.", 0),
        ("I couldn't even finish watching this movie. It was that bad and uninteresting.", 0),
        ("The plot was nonsensical and the acting was wooden. Would not recommend to anyone.", 0),
        ("This movie failed on every level. Poor writing, bad acting, and weak cinematography.", 0),
        ("I was bored throughout the entire film. The story dragged and nothing interesting happened.", 0),
        ("The movie was painfully slow and the ending was unsatisfying. Complete disappointment.", 0),
    ]
    
    # Split into train and test
    train_size = int(0.8 * len(reviews))
    train_data = reviews[:train_size]
    test_data = reviews[train_size:]
    
    return train_data, test_data

class MovieReviewDataset(Dataset):
    def __init__(self, reviews, word_to_idx, max_length=200):
        self.reviews = reviews
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review, label = self.reviews[idx]
        
        # Simple tokenization (remove HTML tags and special characters)
        review = re.sub(r'<[^>]+>', '', review)  # Remove HTML tags
        words = re.findall(r'\b\w+\b', review.lower())
        
        # Convert to indices
        word_indices = []
        for word in words:
            word_idx = self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
            word_indices.append(word_idx)
        
        # Pad or truncate
        if len(word_indices) < self.max_length:
            word_indices.extend([self.word_to_idx['<PAD>']] * (self.max_length - len(word_indices)))
        else:
            word_indices = word_indices[:self.max_length]
        
        return torch.tensor(word_indices), torch.tensor(label, dtype=torch.float)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM processes the sequence
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state (from both directions)
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate last layer both directions
        
        # Apply dropout and classification
        output = self.dropout(final_hidden)
        output = self.classifier(output)
        output = self.sigmoid(output)
        
        return output.squeeze()

def build_vocabulary(reviews, min_freq=2, max_vocab_size=10000):
    """Build vocabulary from reviews"""
    word_counter = Counter()
    
    for review, _ in reviews:
        # Clean and tokenize
        review = re.sub(r'<[^>]+>', '', review)  # Remove HTML tags
        words = re.findall(r'\b\w+\b', review.lower())
        for word in words:
            word_counter[word] += 1
    
    # Create word-to-index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    
    # Add words that appear at least min_freq times
    for word, count in word_counter.most_common(max_vocab_size - 2):
        if count >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx

def train_sentiment_model():
    print("Loading IMDB movie review dataset...")
    
    # Load dataset
    train_reviews, test_reviews = get_imdb_dataset(max_samples=5000)
    
    print(f"Training samples: {len(train_reviews)}")
    print(f"Test samples: {len(test_reviews)}")
    
    # Build vocabulary
    print("Building vocabulary...")
    word_to_idx = build_vocabulary(train_reviews, min_freq=2, max_vocab_size=10000)
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Split training data for validation
    train_size = int(0.8 * len(train_reviews))
    train_data = train_reviews[:train_size]
    val_data = train_reviews[train_size:]
    
    # Create datasets
    train_dataset = MovieReviewDataset(train_data, word_to_idx, max_length=200)
    val_dataset = MovieReviewDataset(val_data, word_to_idx, max_length=200)
    test_dataset = MovieReviewDataset(test_reviews, word_to_idx, max_length=200)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = SentimentLSTM(
        vocab_size=len(word_to_idx),
        embedding_dim=128,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    )
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 20
    train_losses = []
    val_accuracies = []
    
    print(f"\nTraining LSTM model for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_reviews, batch_labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_reviews)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch_reviews, batch_labels in val_loader:
                outputs = model(batch_reviews)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        accuracy = correct / total
        avg_train_loss = total_loss / num_batches
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1:2d}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {accuracy:.4f}')
    
    # Test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_reviews, batch_labels in test_loader:
            outputs = model(batch_reviews)
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)
    
    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    return model, word_to_idx, train_losses, val_accuracies, test_accuracy

def predict_sentiment(model, review, word_to_idx, max_length=200):
    """Predict sentiment for a single review"""
    model.eval()
    
    # Clean and tokenize
    review = re.sub(r'<[^>]+>', '', review)  # Remove HTML tags
    words = re.findall(r'\b\w+\b', review.lower())
    
    # Convert to indices
    word_indices = []
    for word in words:
        word_idx = word_to_idx.get(word, word_to_idx['<UNK>'])
        word_indices.append(word_idx)
    
    # Pad or truncate
    if len(word_indices) < max_length:
        word_indices.extend([word_to_idx['<PAD>']] * (max_length - len(word_indices)))
    else:
        word_indices = word_indices[:max_length]
    
    # Convert to tensor
    input_tensor = torch.tensor(word_indices).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()
        prediction = "Positive" if probability > 0.5 else "Negative"
    
    return prediction, probability

def plot_training_progress(train_losses, val_accuracies):
    """Plot training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train the model
    model, word_to_idx, losses, accuracies, test_acc = train_sentiment_model()
    
    # Plot training progress
    plot_training_progress(losses, accuracies)
    
    # Test predictions
    print("\n" + "="*60)
    print("TESTING SENTIMENT PREDICTIONS")
    print("="*60)
    
    test_reviews = [
        "This movie was absolutely amazing! The cinematography was breathtaking and the story was compelling from start to finish.",
        "I found this film to be quite boring. The plot was predictable and the acting felt wooden throughout.",
        "The movie started slow but really picked up in the second half. Overall, a satisfying experience.",
        "Despite having a great cast, the movie was disappointing. The script was weak and the direction uninspired.",
        "A masterpiece of modern cinema! Every scene was crafted with care and the performances were outstanding.",
        "I couldn't even finish this movie. It was painfully slow and nothing interesting happened.",
        "The film had its moments but overall felt like a missed opportunity. Not terrible, but not great either."
    ]
    
    for review in test_reviews:
        prediction, confidence = predict_sentiment(model, review, word_to_idx)
        print(f"\nReview: {review}")
        print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Vocabulary Size: {len(word_to_idx)}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "="*60)
    print("LSTM ARCHITECTURE INSIGHTS")
    print("="*60)
    print("The LSTM model processes movie reviews using the IMDB dataset:")
    print("1. Each review is tokenized and converted to word embeddings")
    print("2. Bidirectional LSTM processes the sequence in both directions")
    print("3. Hidden states capture contextual information and dependencies")
    print("4. Final hidden states are concatenated and fed to a classifier")
    print("5. Sigmoid activation outputs probability of positive sentiment")
    print("\nKey advantages of this approach:")
    print("- Handles variable-length sequences naturally")
    print("- Captures long-range dependencies in text")
    print("- Bidirectional processing provides richer context")
    print("- Learns meaningful word representations through embeddings")
    print("- Robust to different writing styles and vocabulary")