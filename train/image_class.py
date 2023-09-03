import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import utils
from PIL import Image

# Define the number of epochs
epochs = 5
# Create a data loader to load the dataset in batches
batch_size = 32

learn_rate=0.001

num_classes = 5

# {'chat': 0, 'class': 1, 'code': 2, 'g': 3, 'other': 4}
class_map = {0:'chat', 1:'class', 2:'code', 3:'g', 4:'other'}


# 定义损失函数和优化器（如果需要）
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
# criterion = nn.NLLLoss()
# criterion = nn.BCELoss()


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
        self.class_to_idx = self.dataset.class_to_idx
        print(self.class_to_idx)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        dataset = self.dataset[index]
        return dataset

def get_dataloader(data_dirs):

    # Create a list of datasets
    datasets = [ImageFolderDataset(data_dir, transform=train_transform) for data_dir in data_dirs]

    # Merge the datasets into a single dataset
    merged_dataset = torch.utils.data.ConcatDataset(datasets)


    dataloader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_net():
    # Load the pretrained model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
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
    #                            nn.Linear(512, num_classes), # 10 classes as an example
    #                            nn.LogSoftmax(dim=1))

    # classifier = torch.nn.Linear(model.fc.in_features, num_classes)
    # model.fc = classifier

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train(dataloader):
    device = utils.get_device()
    model = get_net()

    model.to(device)

    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=learn_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
    
    # Start the training loop
    for epoch in range(epochs):
        running_loss = 0
        for batch in dataloader: # assuming trainloader is the data loader
            # Move input and label tensors to the device
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            print(labels)
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

    torch.save(model.state_dict(), 'fine_tuned_resnet50.pth')

def load_model(model_path='fine_tuned_resnet50.pth'):
    # 创建一个ResNet50模型
    model = get_net()
    # 加载保存的状态字典
    state_dict = torch.load(model_path)
    # 将状态字典加载到模型中
    model.load_state_dict(state_dict)
    return model


def evaluate(dataloader):
    device = utils.get_device()
    model = load_model()
    # 设置模型为评估模式
    model.eval()
    model.to(device)
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
            print(labels)

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
    print(f"Accuracy: {accuracy * 100:.2f}%")


def do_inference(image_path):
    device = utils.get_device()
    model = load_model()
    model.eval()
    model.to(device)


    # Load and preprocess the image
    image = Image.open(image_path)
    image = train_transform(image).unsqueeze(0)
    image = image.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image)

    # Get the predicted class
    _, predicted = torch.max(output.data, 1)

    # Get the confidence of the predicted class
    softmax = nn.Softmax(dim=1)
    probabilities = softmax(output)
    confidence = torch.max(probabilities).item()
    print(f"Confidence: {confidence * 100:.2f}%")

    # Convert predicted to class name with class_map
    predicted_class = class_map[predicted.item()]
    print(predicted_class,predicted)

    utils.move_file(image_path,predicted_class)
    return predicted_class,confidence


def do_train():
    data_loader = get_dataloader(["/data/dataset/train"])
    train(data_loader)

def do_evaluate():
    data_loader = get_dataloader(["/data/dataset/test"])
    evaluate(data_loader)

# do_train()
# do_evaluate()
# do_inference("/data/dataset/train/class/2.jpeg")
utils.recursively_iterate_dir("/data/dataset/file",do_inference)