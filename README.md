# Deep Learning with PyTorch - Session 7: LSTM/GRU for Sentiment Analysis

## Session Timeline

| Time      | Activity                                    |
| --------- | ------------------------------------------- |
| 0:00 - 0:10 | 1. Check-in + Session 6 Recap              |
| 0:10 - 0:30 | 2. Why CNNs Fail at Long-Distance Text     |
| 0:30 - 0:55 | 3. LSTM Example: Movie Review Sentiment    |
| 0:55 - 1:20 | 4. Your Turn: GRU for Product Reviews      |
| 1:20 - 1:30 | 5. Results Comparison & Wrap-up            |

---

## 1. Check-in + Session 6 Recap

### Quick Recap Questions
* How did word embeddings improve your NER model compared to one-hot encoding?
* What limitations did you notice with the feedforward approach for NER?
* Can you explain why each word was classified independently in our model?
* What patterns did you observe in the embedding visualizations?

### Key Takeaways from Session 6
* **Dense Representations**: Embeddings capture semantic relationships between words
* **Lookup Table**: Embedding layer converts indices to learned dense vectors
* **Independent Processing**: Each word classified without considering context
* **Fundamental Limitation**: No sequence modeling or long-distance dependencies

---

## 2. Why CNNs Fail at Long-Distance Text

### The Problem with CNNs for Text

**Limited Receptive Field:**
- CNNs use fixed-size kernels (e.g., 3x3, 5x5)
- Can only "see" local neighborhoods
- Long-distance relationships require very deep networks

**Text vs. Images:**
- Images: Local patterns (edges, shapes) build up to global understanding
- Text: Meaning often depends on words far apart in the sequence

### Examples Where Distance Matters

**Sentiment Analysis:**
```
"The movie started slowly and I thought it would be boring, but the amazing plot twist in the final act made it absolutely incredible!"
```
- Early words: negative sentiment
- Later words: positive sentiment
- Overall: positive (requires understanding full context)

**Negation Handling:**
```
"I don't think this product is bad" vs "I think this product is bad"
```
- Single word "don't" completely flips meaning
- May be separated from "bad" by many words

### Enter RNNs: Sequential Processing

**Key Advantages:**
1. **Sequential Processing**: Process one word at a time, building up context
2. **Memory**: Hidden state carries information from previous words
3. **Variable Length**: Can handle sentences of any length
4. **Long-Distance Dependencies**: Information can flow across entire sequence

**LSTM vs. GRU:**
- **LSTM**: More complex, three gates (forget, input, output)
- **GRU**: Simpler, two gates (update, reset)
- **Both**: Solve vanishing gradient problem of vanilla RNNs

---

## 3. LSTM Example: Movie Review Sentiment

### Task: Binary Sentiment Classification

**Dataset**: Movie reviews labeled as positive (1) or negative (0)
**Goal**: Given a review, predict if it's positive or negative sentiment
**Why This Task**: Sentiment often depends on understanding the full review context

### Complete LSTM Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

# Sample movie review data
def get_movie_reviews():
    """Get sample movie review data"""
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
        
        # More complex examples (positive with negative words)
        ("The movie started slowly and I thought it would be boring, but the amazing plot twist made it incredible!", 1),
        ("I wasn't expecting much from this film, but it surprised me with its depth and emotion.", 1),
        ("Despite some flaws in the beginning, the movie became absolutely brilliant in the second half.", 1),
        
        # More complex examples (negative with positive words)
        ("The movie had good actors but the story was so bad that even great performances couldn't save it.", 0),
        ("I wanted to love this film but it was just too confusing and poorly executed.", 0),
        ("The cinematography was beautiful but unfortunately the plot was completely nonsensical.", 0),
    ]
    
    return reviews

class MovieReviewDataset(Dataset):
    def __init__(self, reviews, word_to_idx, max_length=100):
        self.reviews = reviews
        self.word_to_idx = word_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review, label = self.reviews[idx]
        
        # Simple tokenization
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

def train_sentiment_model():
    print("Loading movie review data...")
    reviews = get_movie_reviews()
    
    # Build vocabulary
    word_counter = Counter()
    for review, _ in reviews:
        words = re.findall(r'\b\w+\b', review.lower())
        for word in words:
            word_counter[word] += 1
    
    # Create word-to-index mapping
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counter.most_common():
        word_to_idx[word] = len(word_to_idx)
    
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Split data
    train_size = int(0.8 * len(reviews))
    train_reviews = reviews[:train_size]
    val_reviews = reviews[train_size:]
    
    # Create datasets
    train_dataset = MovieReviewDataset(train_reviews, word_to_idx)
    val_dataset = MovieReviewDataset(val_reviews, word_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = SentimentLSTM(
        vocab_size=len(word_to_idx),
        embedding_dim=100,
        hidden_dim=64,
        num_layers=2
    )
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    train_losses = []
    val_accuracies = []
    
    print("\nTraining LSTM model...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch_reviews, batch_labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_reviews)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
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
        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(accuracy)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d}, Loss: {total_loss/len(train_loader):.4f}, Val Acc: {accuracy:.4f}')
    
    return model, word_to_idx, train_losses, val_accuracies

def predict_sentiment(model, review, word_to_idx, max_length=100):
    """Predict sentiment for a single review"""
    model.eval()
    
    # Tokenize
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
    
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Train the model
    model, word_to_idx, losses, accuracies = train_sentiment_model()
    
    # Plot training progress
    plot_training_progress(losses, accuracies)
    
    # Test predictions
    print("\n" + "="*50)
    print("TESTING SENTIMENT PREDICTIONS")
    print("="*50)
    
    test_reviews = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "The film was boring and poorly made. Complete waste of time.",
        "I didn't expect much but it turned out to be surprisingly good.",
        "The movie started well but the ending was disappointing.",
        "Great actors, terrible story. Even good performances couldn't save it."
    ]
    
    for review in test_reviews:
        prediction, confidence = predict_sentiment(model, review, word_to_idx)
        print(f"\nReview: {review}")
        print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
    
    print("\n" + "="*50)
    print("LSTM ARCHITECTURE INSIGHTS")
    print("="*50)
    print("The LSTM model processes reviews sequentially:")
    print("1. Each word is converted to an embedding vector")
    print("2. LSTM processes words one by one, building up context")
    print("3. Hidden state carries information from earlier words")
    print("4. Final hidden state represents the entire review")
    print("5. Classifier uses this representation to predict sentiment")
    print("\nThis allows the model to understand:")
    print("- Long-distance dependencies (early negation affecting later words)")
    print("- Context shifts (positive words after 'but' overriding negative start)")
    print("- Sequential flow of sentiment throughout the review")
```

---

## 4. Your Turn: GRU for Product Reviews

### Task Description

Now it's your turn to implement a GRU-based model for product review sentiment analysis. GRU is simpler than LSTM but often performs similarly well.

### Your Assignment

**Dataset**: Product reviews (electronics, books, etc.)
**Architecture**: GRU instead of LSTM
**Goal**: Compare GRU performance to LSTM

### Starter Code Structure

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

# TODO: Create product review dataset
def get_product_reviews():
    """
    Create a dataset of product reviews
    Include both positive and negative reviews
    Make some reviews complex (positive with negative words, etc.)
    """
    reviews = [
        # TODO: Add at least 20 product reviews with labels
        # Format: ("review text", label)  # label: 1 for positive, 0 for negative
    ]
    return reviews

# TODO: Implement ProductReviewDataset class
class ProductReviewDataset(Dataset):
    def __init__(self, reviews, word_to_idx, max_length=100):
        # TODO: Initialize dataset
        pass
    
    def __len__(self):
        # TODO: Return dataset length
        pass
    
    def __getitem__(self, idx):
        # TODO: Return tokenized review and label
        pass

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
        
    def forward(self, x):
        # TODO: Implement forward pass
        # 1. Embed inputs
        # 2. Pass through GRU
        # 3. Extract final hidden state
        # 4. Apply classifier
        pass

# TODO: Implement training function
def train_gru_model():
    """Train the GRU model and return results"""
    # TODO: 
    # 1. Load data and build vocabulary
    # 2. Create datasets and dataloaders
    # 3. Initialize model, loss, optimizer
    # 4. Training loop with validation
    # 5. Return model and training history
    pass

# TODO: Implement prediction function
def predict_product_sentiment(model, review, word_to_idx, max_length=100):
    """Predict sentiment for a product review"""
    # TODO: Implement prediction logic
    pass

# TODO: Compare GRU vs LSTM
def compare_architectures():
    """Compare GRU and LSTM performance"""
    # TODO: 
    # 1. Train both models on same data
    # 2. Compare training time, accuracy, memory usage
    # 3. Test on same examples
    # 4. Analyze differences
    pass

# Main execution
if __name__ == "__main__":
    # TODO: Complete the implementation and run experiments
    pass
```

### Key Differences to Implement

**GRU vs LSTM:**
- GRU: `nn.GRU()` instead of `nn.LSTM()`
- GRU returns only hidden state (no cell state)
- Simpler architecture, potentially faster training

**Your Tasks:**
1. **Create Dataset**: Build product review dataset with varied examples
2. **Implement GRU**: Use GRU layer instead of LSTM
3. **Training Loop**: Train and validate model
4. **Comparison**: Compare GRU vs LSTM performance
5. **Analysis**: Discuss when to use each architecture

---

## 5. Results Comparison & Wrap-up

### Expected Results

**LSTM Model:**
- Should achieve ~85-95% accuracy on movie reviews
- Good at handling sentiment shifts and negations
- Bidirectional processing helps with context

**GRU Model:**
- Similar accuracy to LSTM
- Faster training due to simpler architecture
- Less memory usage

### Key Insights

**Why RNNs Work for Sentiment:**
1. **Sequential Processing**: Natural fit for text data
2. **Memory**: Can remember earlier context
3. **Flexible Length**: Handle variable-length reviews
4. **Long Dependencies**: Connect distant words effectively

**LSTM vs GRU Trade-offs:**
- **LSTM**: More parameters, potentially better for complex tasks
- **GRU**: Simpler, faster, often similar performance
- **Choice**: Depends on data size, complexity, computational constraints

### Architecture Comparison

| Model | Receptive Field | Memory | Best For |
|-------|----------------|---------|----------|
| CNN | Local (kernel size) | None | Local patterns |
| LSTM | Full sequence | Long-term | Complex sequences |
| GRU | Full sequence | Medium-term | Simpler sequences |

### Next Session Preview

**Attention Mechanisms:**
- Why even LSTMs have limitations
- How attention solves the information bottleneck
- Transformer architectures
- BERT and modern NLP

### Homework

**Complete the GRU Implementation:**
- Finish the product review sentiment classifier
- Compare training time and accuracy with LSTM
- Test on longer reviews to see performance differences

**Experiment with Architecture:**
- Try different hidden dimensions
- Compare single vs bidirectional GRU
- Analyze what each model learns about sentiment

---

**Session Summary**: This session demonstrated how LSTM/GRU architectures naturally handle sequential text data and long-distance dependencies that CNNs struggle with, using sentiment analysis as a practical example where context and sequence order are crucial for understanding meaning.