import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import argparse
import os
import torchvision


# In this section defines neural network models.
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = torchvision.models.alexnet(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x

class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.vgg(x)
        return x

# Trains the neural network model on the training dataset.
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

# This function evaluates the performance of the fine-tuned model by calculating the average loss and accuracy over the dataset
def fine_tuning(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(test_loader.dataset), correct / total

# Saving model weights
def save_weights(model, model_name, scenario):
    if not os.path.exists("weight"):
        os.makedirs("weight")
    torch.save(model.state_dict(), f"weight/{model_name}_Scenario{scenario}.pt")

# Get LFW dataset with specific parameters
lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

#split the dataset
X = lfw_dataset.images
y = lfw_dataset.target

#split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

#apply transformation
X_train = torch.stack([transform(x) for x in X_train])
X_test = torch.stack([transform(x) for x in X_test])

# Now converting your data to PyTorch tensor format. This will format your data into a suitable format that the PyTorch library can use
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define main function
def main(model_name="AlexNet", scenario=1, bypass_train=False):
    # Determine number of classes
    num_classes = len(torch.unique(y_train))

    # Select model
    if model_name == "AlexNet":
        model = AlexNet(num_classes)
    elif model_name == "ResNet":
        model = ResNet(num_classes)
    elif model_name == "VGG":
        model = VGG(num_classes)
    else:
        raise ValueError("Invalid model name")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define data loaders
    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # Defines the loss function and optimizer to be used during training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # If you do not pass the training, the model will be trained.
    if not bypass_train:
        for epoch in range(5):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = fine_tuning(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{5}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Save model weights
        save_weights(model, model_name, scenario)

# Parse the command line arguments and get the model name, scenario and whether the user wants to skip the tutorial from the user.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="AlexNet", choices=["AlexNet", "ResNet", "VGG"])
parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3, 4])
parser.add_argument("--bypass_train", action="store_true")
args = parser.parse_args()

# Run main function
main(model_name=args.model, scenario=args.scenario, bypass_train=args.bypass_train)
