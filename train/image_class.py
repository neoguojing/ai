import torch
import torch.nn as nn
from torchvision import models

# Load the pretrained model
model = models.resnet50(pretrained=True)

# Freeze all the layers in the network
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last layer (layer4)
for param in model.layer4.parameters():
    param.requires_grad = True

# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
classifier = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(512, 10), # 10 classes as an example
                           nn.LogSoftmax(dim=1))

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
    for inputs, labels in trainloader: # assuming trainloader is the data loader
        # Move input and label tensors to the device
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
          f"Train loss: {running_loss/len(trainloader):.3f}.. ")
