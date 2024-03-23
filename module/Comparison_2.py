import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
# Use LabelEncoder to encode the labels
label_encoder = LabelEncoder()

# Load the data
df = pd.read_excel("BGL_processed.xlsx") # File address and file name

# Define the dataset scale
percentage_of_data = 0.5  # Set the dataset size
number_of_rows = int(len(df) * percentage_of_data)
df = df.iloc[:number_of_rows]

df = df[['Processed_Content', 'Label']]
df['Label'] = label_encoder.fit_transform(df['Label'])
df['Label'] = df['Label'].astype(int)

# Define the dataset
# Vocabulary building
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4  # Start indexing from 4 because 0,1,2,3 are already reserved

        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
                          for token in text.split()]

        return [self.stoi["<SOS>"]] + tokenized_text + [self.stoi["<EOS>"]]

# Define the dataset
class Dataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['Processed_Content']
        label = self.df.iloc[index]['Label']

        numericalized_text = self.vocab.numericalize(text)
        pad_len = self.max_len - len(numericalized_text)
        numericalized_text += [self.vocab.stoi["<PAD>"]] * pad_len  # Padding

        return torch.tensor(numericalized_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Parameters
MAX_LEN = 100  # Or another value depending on your data
BATCH_SIZE = 32
FREQ_THRESHOLD = 2

# Building the vocab
vocab = Vocabulary(FREQ_THRESHOLD)
vocab.build_vocabulary(df['Processed_Content'].tolist())

# Data Loaders
dataset = Dataset(df, vocab, MAX_LEN)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Definition
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.attention = Attention(hidden_dim * 2)  # Because it's bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_output, (hidden, cell) = self.lstm(embedded)
        attention_output, _ = self.attention(lstm_output)
        return self.fc(attention_output)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = attention_weights * lstm_output
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

# Initialize model
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
OUTPUT_DIM = df['Label'].nunique()
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
VOCAB_SIZE = len(vocab)

model = BiLSTMAttention(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

# Move model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training function
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for texts, labels in iterator:
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(texts)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for texts, labels in iterator:
            texts = texts.to(device)
            labels = labels.to(device)

            predictions = model(texts)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# Training loop
N_EPOCHS = 10

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    valid_loss = evaluate(model, test_loader, criterion)

    print(f'Epoch: {epoch + 1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

print("Training complete.")
