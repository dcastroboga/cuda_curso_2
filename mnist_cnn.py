import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)
logging.basicConfig(filename='output/mnist_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    acc = correct / len(test_loader.dataset) * 100
    print(f"Accuracy: {acc:.2f}%")
    logging.info(f"Accuracy: {acc:.2f}%")
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    acc_history = []
    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        acc_history.append(acc)

    plt.plot(range(1, 6), acc_history, marker='o')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig("output/accuracy_plot.png")
    logging.info("Saved accuracy plot to output/accuracy_plot.png")

if __name__ == "__main__":
    main()
