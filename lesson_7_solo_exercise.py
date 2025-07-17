# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:40:18 2025

@author: taske
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import time

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

def get_product_reviews(max_samples=5000):
    """
    Load IMDB dataset as product reviews
    Include both positive and negative reviews
    """
    
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
        return get_sample_product_reviews()

def get_sample_product_reviews():
    """Fallback sample data if IMDB download fails"""
    reviews = [
        # Positive reviews
        ("This product exceeded my expectations! Great quality and fast shipping.", 1),
        ("Absolutely love this item. Works perfectly and looks amazing.", 1),
        ("Best purchase I've made in a long time. Highly recommended!", 1),
        ("Excellent product with outstanding customer service. Five stars!", 1),
        ("The quality is superb and the price is reasonable. Very satisfied.", 1),
        ("This product is exactly what I was looking for. Perfect!", 1),
        ("Amazing value for money. The product works flawlessly.", 1),
        ("I'm very impressed with this purchase. Great build quality.", 1),
        ("Fantastic product that delivers on all its promises.", 1),
        ("Couldn't be happier with this purchase. Excellent quality!", 1),
        
        # Negative reviews
        ("This product was terrible. Poor quality and doesn't work as advertised.", 0),
        ("Waste of money. The product broke after just one use.", 0),
        ("Very disappointed with this purchase. The quality is awful.", 0),
        ("Don't buy this product. It's cheaply made and overpriced.", 0),
        ("The product arrived damaged and customer service was unhelpful.", 0),
        ("This item is nothing like the description. Complete disappointment.", 0),
        ("Poor build quality and it stopped working after a week.", 0),
        ("I regret buying this product. It's not worth the money.", 0),
        ("The product doesn't meet basic quality standards. Avoid!", 0),
        ("Terrible experience with this product. Would not recommend.", 0),
    ]
    
    # Split into train and test
    train_size = int(0.8 * len(reviews))
    train_data = reviews[:train_size]
    test_data = reviews[train_size:]
    
    return train_data, test_data

# TODO: Implement ProductReviewDataset class
class ProductReviewDataset(Dataset):
    def __init__(self, reviews, word_to_idx, max_length=100):
        self.reviews = reviews
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review, label = self.reviews[idx]
        
        # Clean and tokenize
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

# TODO: Implement GRU model
class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(SentimentGRU, self).__init__()
        
        # TODO: Define layers
        # - Embedding layer
        # - GRU layer (bidirectional)
        # - Dropout
        # - Classifier
        # - Sigmoid activation
        
        pass
        
    def forward(self, x):
        # TODO: Implement forward pass
        # 1. Embed inputs
        # 2. Pass through GRU
        # 3. Extract final hidden state
        # 4. Apply classifier
        pass

# LSTM implementation for comparison (completed)
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
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
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

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001):
    """Generic training function for both GRU and LSTM"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    
    start_time = time.time()
    
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
        
        with torch.no_grad():
            for batch_reviews, batch_labels in val_loader:
                outputs = model(batch_reviews)
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
        
        accuracy = correct / total
        avg_train_loss = total_loss / num_batches
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(accuracy)
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1:2d}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Acc: {accuracy:.4f}')
    
    training_time = time.time() - start_time
    return train_losses, val_accuracies, training_time

# TODO: Implement training function
def train_gru_model():
    """Train the GRU model and return results"""
    print("Loading product review dataset...")
    
    # Load dataset
    train_reviews, test_reviews = get_product_reviews(max_samples=5000)
    
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
    train_dataset = ProductReviewDataset(train_data, word_to_idx, max_length=200)
    val_dataset = ProductReviewDataset(val_data, word_to_idx, max_length=200)
    test_dataset = ProductReviewDataset(test_reviews, word_to_idx, max_length=200)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # TODO: Initialize GRU model
    # Uncomment and complete this when GRU is implemented
    # model = SentimentGRU(
    #     vocab_size=len(word_to_idx),
    #     embedding_dim=128,
    #     hidden_dim=64,
    #     num_layers=2,
    #     dropout=0.3
    # )
    
    # For now, return placeholder data
    print("⚠️  GRU model not implemented yet - returning placeholder data")
    return None, word_to_idx, [], [], 0.0, test_loader

# TODO: Implement prediction function
def predict_product_sentiment(model, review, word_to_idx, max_length=200):
    """Predict sentiment for a product review"""
    if model is None:
        return "Not implemented", 0.5
    
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

# TODO: Compare GRU vs LSTM
def compare_architectures():
    """Compare GRU and LSTM performance"""
    print("="*60)
    print("COMPARING GRU vs LSTM ARCHITECTURES")
    print("="*60)
    
    # Load dataset
    train_reviews, test_reviews = get_product_reviews(max_samples=3000)  # Smaller for comparison
    word_to_idx = build_vocabulary(train_reviews, min_freq=2, max_vocab_size=8000)
    
    # Prepare data
    train_size = int(0.8 * len(train_reviews))
    train_data = train_reviews[:train_size]
    val_data = train_reviews[train_size:]
    
    train_dataset = ProductReviewDataset(train_data, word_to_idx, max_length=150)
    val_dataset = ProductReviewDataset(val_data, word_to_idx, max_length=150)
    test_dataset = ProductReviewDataset(test_reviews, word_to_idx, max_length=150)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model parameters
    model_params = {
        'vocab_size': len(word_to_idx),
        'embedding_dim': 128,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3
    }
    
    # Train LSTM
    print("Training LSTM model...")
    lstm_model = SentimentLSTM(**model_params)
    lstm_losses, lstm_accuracies, lstm_time = train_model(lstm_model, train_loader, val_loader, num_epochs=15)
    
    # TODO: Train GRU (uncomment when implemented)
    # print("Training GRU model...")
    # gru_model = SentimentGRU(**model_params)
    # gru_losses, gru_accuracies, gru_time = train_model(gru_model, train_loader, val_loader, num_epochs=15)
    
    # Test both models
    def test_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_reviews, batch_labels in test_loader:
                outputs = model(batch_reviews)
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)
        return correct / total
    
    lstm_test_acc = test_model(lstm_model, test_loader)
    
    # TODO: Test GRU (uncomment when implemented)
    # gru_test_acc = test_model(gru_model, test_loader)
    
    # Model size comparison
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    # gru_params = sum(p.numel() for p in gru_model.parameters())
    
    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"LSTM - Test Accuracy: {lstm_test_acc:.4f}, Training Time: {lstm_time:.2f}s, Parameters: {lstm_params:,}")
    print(f"GRU  - Test Accuracy: [TODO], Training Time: [TODO]s, Parameters: [TODO]")
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(lstm_losses, label='LSTM Loss', color='blue')
    # plt.plot(gru_losses, label='GRU Loss', color='red')  # TODO: Uncomment when GRU is implemented
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(lstm_accuracies, label='LSTM Accuracy', color='blue')
    # plt.plot(gru_accuracies, label='GRU Accuracy', color='red')  # TODO: Uncomment when GRU is implemented
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    models = ['LSTM']  # TODO: Add 'GRU' when implemented
    test_accs = [lstm_test_acc]  # TODO: Add gru_test_acc when implemented
    colors = ['blue']  # TODO: Add 'red' when implemented
    
    plt.bar(models, test_accs, color=colors, alpha=0.7)
    plt.title('Final Test Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("ARCHITECTURE ANALYSIS")
    print("="*60)
    print("Key Differences:")
    print("• LSTM: Uses input, forget, and output gates with cell state")
    print("• GRU: Uses reset and update gates, simpler architecture")
    print("• GRU typically has fewer parameters and faster training")
    print("• LSTM may perform better on tasks requiring long-term memory")
    print("• Both handle vanishing gradient problem better than vanilla RNNs")

def plot_training_progress(train_losses, val_accuracies, title="Training Progress"):
    """Plot training progress"""
    if not train_losses or not val_accuracies:
        print("No training data to plot")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title(f'{title} - Accuracy')
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
    
    print("="*60)
    print("GRU SENTIMENT ANALYSIS EXERCISE")
    print("="*60)
    print("This exercise will help you implement a GRU model for sentiment analysis")
    print("Dataset: IMDB Movie Reviews (adapted as product reviews)")
    print("Task: Binary sentiment classification (positive/negative)")
    print()
    
    # Step 1: Train GRU model
    print("Step 1: Training GRU model...")
    model, word_to_idx, losses, accuracies, test_acc, test_loader = train_gru_model()
    
    # Step 2: Plot training progress
    if losses and accuracies:
        print("Step 2: Plotting training progress...")
        plot_training_progress(losses, accuracies, "GRU Model")
    
    # Step 3: Test predictions
    print("\nStep 3: Testing sentiment predictions...")
    test_reviews = [
        "This product is absolutely amazing! The quality exceeded all my expectations.",
        "Terrible product, waste of money. Poor quality and doesn't work as advertised.",
        "The item arrived quickly and works well. Good value for the price.",
        "I'm disappointed with this purchase. The product feels cheap and flimsy.",
        "Outstanding quality and excellent customer service. Highly recommend!",
        "Don't buy this product. It broke after one day of use.",
        "The product is okay but nothing special. Average quality for the price."
    ]
    
    for review in test_reviews:
        prediction, confidence = predict_product_sentiment(model, review, word_to_idx)
        print(f"\nReview: {review}")
        print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
    
    # Step 4: Compare architectures
    print("\nStep 4: Comparing GRU vs LSTM architectures...")
    compare_architectures()
    
    print("\n" + "="*60)
    print("EXERCISE INSTRUCTIONS")
    print("="*60)
    print("TODO: Complete the following implementations:")
    print("1. SentimentGRU class - implement the GRU-based model")
    print("2. Compare the GRU model with the provided LSTM implementation")
    print("3. Analyze the differences in performance, speed, and memory usage")
    print("\nHints:")
    print("• GRU is similar to LSTM but uses fewer gates")
    print("• Use nn.GRU instead of nn.LSTM")
    print("• GRU output shape is different from LSTM (no cell state)")
    print("• For bidirectional GRU, concatenate final hidden states from both directions")
    print("• Test your implementation by running the comparison function")
    
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Training samples: {len(word_to_idx) if word_to_idx else 'N/A'}")
    print(f"Vocabulary size: {len(word_to_idx) if word_to_idx else 'N/A'}")
    print("Dataset source: IMDB Movie Reviews")
    print("Task: Binary sentiment classification")
    print("Sequence length: 200 tokens")
    print("Batch size: 32")