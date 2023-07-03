import torch
import sqlite3
import pandas as pd
from transformers import AlbertTokenizerFast,AlbertForSequenceClassification,AlbertTokenizer
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, data, labels, tokenizer):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        encoded_text = self.tokenizer(self.data[index], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        label = torch.tensor(self.labels[index])
        attention_mask = encoded_text['attention_mask']
        return encoded_text, attention_mask, label

# Connect to the SQLite database and retrieve data and labels
conn = sqlite3.connect('mydatabase.db')
cursor = conn.execute("SELECT data, label FROM mytable")
rows = cursor.fetchall()

# Convert the data and labels to a Pandas DataFrame
df = pd.DataFrame(rows, columns=['data', 'label'])

# Load the Albert tokenizer
tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')

# Create an instance of the custom dataset
dataset = TextDataset(df['data'].tolist(), df['label'].tolist(), tokenizer)

# Create a data loader to load the dataset in batches
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Albert model
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 3
# Train the model with the data loader
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch[0]
        attention_mask = batch[1]
        labels = batch[2]
        outputs = model(inputs['input_ids'], attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # loss = loss_fn(outputs.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()