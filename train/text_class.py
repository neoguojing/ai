import torch
import sqlite3
import pandas as pd
from transformers import AlbertTokenizerFast,AlbertForSequenceClassification,AlbertTokenizer
from torch.utils.data import Dataset, DataLoader
import jieba
from collections import Counter
import utils

batch_size = 32
num_epochs = 3
max_length = 128
label_map = {'f': 0, 'm': 1,'s': 2,'a': 3}
# Load the Albert tokenizer
tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
stopwords = []
with open('baidu_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().split('\n')
 
def clean(text):
    words = jieba.cut(text, cut_all=False)
    words = [word for word in words if not word.encode('utf-8').isalnum()]
    words = [word for word in words if word not in stopwords]
    word_count = Counter(words)
    words = [word for word in words if word_count[word] > 2]
    words = list(set(words))
    return ' '.join(words),word_count

def default_segmentation(text):
    text,_ = clean(text)
    return ' '.join(tokenizer.cut(text))

# Define a custom dataset
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.data.loc[index, 'input_ids'])
        attention_mask = torch.tensor(self.data.loc[index, 'attention_mask'])
        label = torch.tensor(self.data.loc[index, 'label'])
        return input_ids, attention_mask, label

def get_dataloader(table):
    # Connect to the SQLite database and retrieve data and labels
    conn = sqlite3.connect('telegram.db')
    cursor = conn.execute(f"SELECT text, label FROM {table}")
    rows = cursor.fetchall()

    # Convert the data and labels to a Pandas DataFrame
    df = pd.DataFrame(rows, columns=['text', 'label'])

    df['text'] = df['text'].apply(lambda x: default_segmentation(x))
    # Create input sequences by encoding the text
    df['input_ids'] = df['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_length, truncation=True))

    # Pad the input sequences to have a fixed length
    df['input_ids'] = df['input_ids'].apply(lambda x: x + [0]*(max_length-len(x)))

    # Convert the labels to numerical values
    df['label'] = df['label'].map(label_map)

    # Create attention masks to ignore the padding tokens
    df['attention_mask'] = df['input_ids'].apply(lambda x: [int(i>0) for i in x])

    # Create an instance of the custom dataset
    dataset = TextDataset(df)
    # Create a data loader to load the dataset in batches
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return dataloader


def train(dataloader):
    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2',num_labels=4)
    # Define the Albert model
    device = utils.get_device()
    model.to(device)
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = BCEWithLogitsLoss()

    # Train the model with the data loader
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = criterion(outputs.logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print the loss
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
        model.save_pretrained('neo_albert_model')

def evaluate(dataloader):
    model = load_model()
    device = utils.get_device()
    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            predicted_labels = torch.argmax(outputs.logits, dim=1)

            # Compute accuracy
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

        print('Test Accuracy: {:.2f}%'.format(correct / total * 100))

def load_model(model_path="neo_albert_model"):
    model = AlbertForSequenceClassification.from_pretrained(model_path)
    return  model
 




