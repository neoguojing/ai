
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ImageDatasetLoader:
    def __init__(self, root_dir, transform):
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    def get_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

# Define transformations for the train set
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset with ImageFolder
train_dataset_loader = ImageDatasetLoader(root_dir='path_to_train_data', transform=train_transforms)

# Define the dataloader
train_loader = train_dataset_loader.get_dataloader(batch_size=32, shuffle=True)

from torch.utils.data import Dataset
from transformers import AlbertTokenizer
import torch

class TextDataset(Dataset):
    def __init__(self, file_path, max_length=512):
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.max_length = max_length
        self.lines = open(file_path, 'r').readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        encoding = self.tokenizer.encode_plus(
            line,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': line,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# Load the text dataset
train_text_dataset = TextDataset(file_path='path_to_train_text_data')

# Define the dataloader for text data
train_text_loader = DataLoader(train_text_dataset, batch_size=32, shuffle=True)
