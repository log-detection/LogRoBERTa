import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = 128

# Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    df['Processed_Content'], df['Label'], test_size=0.2, random_state=17
)

# Data loaders
train_dataset = SentimentDataset(
    texts=X_train.to_numpy(),
    labels=y_train.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=64,  # Adjust based on your hardware
    shuffle=True
)

# Create the validation dataset
val_dataset = SentimentDataset(
    texts=X_val.to_numpy(),
    labels=y_val.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
)

# Create the validation DataLoader
val_data_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights


# Model Definition
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.attention = Attention(256)
        self.classifier = nn.Linear(256 * 2, n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]

        lstm_output, _ = self.lstm(bert_output)
        context_vector, _ = self.attention(lstm_output)
        return self.classifier(context_vector)


# Model Initialization
n_classes = df['Label'].nunique()
model = SentimentClassifier(n_classes=n_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training Hyperparameters
EPOCHS = 4  # Adjust based on your needs
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss().to(device)

from sklearn.metrics import classification_report

# Training loop with classification report
for epoch in range(EPOCHS):
    # Lists to hold actual and predicted labels
    true_labels = []
    pred_labels = []

    for data in train_data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)  # Labels are already tensors

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        # Move preds to CPU for sklearn metrics
        preds = preds.detach().cpu().numpy()
        # Labels are already tensors, so we just get them to CPU
        labels = labels.detach().cpu().numpy()

        # Append batch prediction results
        pred_labels.extend(preds)
        true_labels.extend(labels)

        # Convert labels back to tensor for loss calculation
        labels = torch.tensor(labels).to(device)  # Convert labels back to tensor

        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Calculate classification report
    class_report = classification_report(true_labels, pred_labels, digits=4)
    print(f'Epoch {epoch + 1}/{EPOCHS} finished')
    print(class_report)

    # Initiate validation process
    model.eval()  # Set the model to evaluation mode
    val_true_labels = []
    val_pred_labels = []

    with torch.no_grad():  # Disable gradient computation
        for data in val_data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            val_pred_labels.extend(preds)
            val_true_labels.extend(labels)
    # Compute and print the classification report for the validation set
    val_class_report = classification_report(val_true_labels, val_pred_labels, digits=4)
    print(f'Epoch {epoch + 1}/{EPOCHS} - Validation')
    print(val_class_report)
