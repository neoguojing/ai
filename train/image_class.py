import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import utils

# Define the number of epochs
epochs = 5
# Create a data loader to load the dataset in batches
batch_size = 32

learn_rate=0.003

# 定义损失函数和优化器（如果需要）
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=learn_rate)

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

def get_dataloader(data_dirs):

    # Create a list of datasets
    datasets = [ImageFolderDataset(data_dir, transform=train_transform) for data_dir in data_dirs]

    # Merge the datasets into a single dataset
    merged_dataset = torch.utils.data.ConcatDataset(datasets)


    dataloader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)
    return dataloader,


def train(dataloader):
    device = utils.get_device()
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

    model.to(device)

    # Start the training loop
    for epoch in range(epochs):
        running_loss = 0
        for batch in dataloader: # assuming trainloader is the data loader
            # Move input and label tensors to the device
            inputs, labels = batch
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

    torch.save(model.state_dict(), 'resnet50.pth')

def load_model(model_path='resnet50.pth'):
    # 创建一个ResNet50模型
    model = models.resnet50(pretrained=False)

    # 加载保存的状态字典
    state_dict = torch.load(model_path)

    # 将状态字典加载到模型中
    model.load_state_dict(state_dict)
    return model


def evaluate(dataloader):
    model = load_model()
    # 设置模型为评估模式
    model.eval()
    
    # 初始化计数器和损失值
    num_correct = 0
    total_samples = 0
    total_loss = 0

    # 在测试集上进行评估
    with torch.no_grad():
        for batch in dataloader: # assuming trainloader is the data loader
            # Move input and label tensors to the device
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            output = model(inputs)

            # 计算损失
            loss = criterion(output, labels)
            total_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            num_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # 计算平均损失和准确率
    avg_loss = total_loss / len(dataloader)
    accuracy = num_correct / total_samples
    print(f"Accuracy: {accuracy:.2f}%")
