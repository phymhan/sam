import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

use_sam = False

# Define the MLP classifier
class SmallMLP(nn.Module):
    def __init__(self):
        super(SmallMLP, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load and preprocess CIFAR-10 data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Initialize the model, loss function, and optimizer
model = SmallMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

from sam import SAM
sam_optimizer = SAM(model.parameters(), optim.SGD, rho=0.05, lr=0.001, momentum=0.9)

# Training loop
num_epochs = 10
losses = []
gradient_norms = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        if not use_sam:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            # first forward-backward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # use this loss for any training statistics
            loss.backward()
            sam_optimizer.first_step(zero_grad=True)
            
            # second forward-backward pass
            criterion(model(inputs), labels).backward()  # make sure to do a full forward pass
            sam_optimizer.second_step(zero_grad=True)

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            losses.append(running_loss / 200)
            running_loss = 0.0

            sharpness_value = 0.0
            num_batches = 0
            for j, data2 in enumerate(testloader):
                # if j >= 10:  # Stop after 10 batches
                #     break
                inputs2, labels2 = data2
                optimizer.zero_grad()  # Still need to zero out gradients even in evaluation
                outputs2 = model(inputs2)
                loss2_old = criterion(outputs2, labels2)
                loss2_old.backward()
                sam_optimizer.first_step()
                loss2 = criterion(model(inputs2), labels2)
                sharpness = loss2 - loss2_old
                sam_optimizer.undo_first_step()
                sharpness_value += sharpness.item()
                num_batches += 1
            gradient_norms.append(sharpness_value / num_batches)


print('Finished Training')
print(f'max sharpness: {max(gradient_norms)}')

# Plot loss and gradient norm curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(gradient_norms)
plt.xlabel('Iterations')
plt.ylabel('Sharpness')
plt.title('Sharpness during Training')

plt.tight_layout()
plt.savefig(f'loss_and_sharpness_{use_sam}.png')
plt.show()
