import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

# Define transforms for data augmentation and normalization
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define a custom dataset
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

# Define the directories for the dataset
data_dirs = ['./data/class1/', './data/class2/', './data/class3/']

# Create a list of datasets
datasets = [ImageFolderDataset(data_dir, transform=train_transform) for data_dir in data_dirs]

# Merge the datasets into a single dataset
merged_dataset = torch.utils.data.ConcatDataset(datasets)

# Create a data loader to load the dataset in batches
batch_size = 32
dataloader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)


# Load the pretrained model
model = models.resnet50(pretrained=True)

# Freeze all the layers in the network
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer (layer4)
for param in model.layer4.parameters():
    param.requires_grad = True

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# classifier = nn.Sequential(nn.Linear(2048, 512),
#                            nn.ReLU(),
#                            nn.Dropout(0.2),
#                            nn.Linear(512, 10), # 10 classes as an example
#                            nn.LogSoftmax(dim=1))

num_classes = len(data_dirs)
classifier = torch.nn.Linear(model.fc.in_features, num_classes)
# Replace the model's classifier with this new classifier
model.fc = classifier

# Define the criterion and the optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.003)

# Move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the number of epochs
epochs = 5

# Start the training loop
for epoch in range(epochs):
    running_loss = 0
    for batch in dataloader: # assuming trainloader is the data loader
        # Move input and label tensors to the device
        images, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/len(dataloader):.3f}.. ")
