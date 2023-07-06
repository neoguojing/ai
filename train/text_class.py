import torch
import sqlite3
import pandas as pd
from transformers import AlbertTokenizerFast,AlbertForSequenceClassification,AlbertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import utils
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

batch_size = 8
num_epochs = 3
max_length = 512
label_map = {'f': 0, 'm': 1,'s': 2,'a': 3}
label_rev_map = {0: 'f', 1: 'm',2: 's',3: 'a'}
# Load the Albert tokenizer
tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')

def default_segmentation(text):
    return ' '.join(tokenizer.tokenize(text,padding=True, truncation=True, max_length=max_length))

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
        text = self.data.loc[index, 'text']
        id = self.data.loc[index, 'id']
        return input_ids, attention_mask, label,text,id

def get_dataloader(table,whatfor):
    # Connect to the SQLite database and retrieve data and labels
    conn = sqlite3.connect('../telegram/db/telegram.db')
    cursor = conn.execute(f"SELECT summary, label,chat_id FROM {table} where whatfor = '{whatfor}'")
    rows = cursor.fetchall()

    # Convert the data and labels to a Pandas DataFrame
    df = pd.DataFrame(rows, columns=['text', 'label','id'])
    df['text'] = df['text'].apply(lambda x: default_segmentation(x))
    print(df['text'])
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
    # 2e-5 for 3 epchos
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = BCEWithLogitsLoss()

    # # Load checkpoint
    # checkpoint = torch.load('checkpoint.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
        
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     }, 'checkpoint.pth')

        model.save_pretrained('neo_albert_model')

def evaluate(dataloader,model_path="neo_albert_model"):
    model = load_model(model_path)
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
            ids =  batch[4].numpy()
            origin_label = labels.cpu().numpy()
            print(origin_label)
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)  # compute softmax probabilities
            _, predicted_labels = torch.max(probs, dim=1)  # get predicted labels

            # Output classification labels and confidence
            for i in range(len(predicted_labels)):
                label = predicted_labels[i].item()
                confidence = probs[i][label].item()
                
                o_index = origin_label[i]
                if origin_label[i] != label:
                    print(f"Test example {ids[i]}: Predicted label: {label_rev_map[label]}, Confidence: {confidence};origin lable:{label_rev_map[o_index]}")

            # Compute accuracy
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

def load_model(model_path="neo_albert_model"):
    model = AlbertForSequenceClassification.from_pretrained(model_path)
    return  model
 

def do_train():
    train_loader = get_dataloader("telegram_user_summary","train")
    train(train_loader)

def do_test():
    test_loader = get_dataloader("telegram_user_summary","test")
    evaluate(test_loader)

def do_inference():
    test_loader = get_dataloader("telegram_user_summary","inference")
    evaluate(test_loader,"neo_albert_model92.5")


do_test()
