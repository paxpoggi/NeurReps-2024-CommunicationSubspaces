import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoBranchCNN_CIFAR(nn.Module):
    def __init__(self):
        super(TwoBranchCNN_CIFAR, self).__init__()
        # Initial common layer
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)  # Adjusted for 3-channel input
        
        # Branch 1
        self.branch1_conv1 = nn.Conv2d(20, 20, kernel_size=5)
        self.branch1_conv2 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.branch1_conv3 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.branch1_drop = nn.Dropout2d()
        self.branch1_fc1 = nn.Linear(2000, 50)  # Adjusted for flattened output size

        # Branch 2
        self.branch2_conv1 = nn.Conv2d(20, 20, kernel_size=5)
        self.branch2_conv2 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.branch2_conv3 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.branch2_drop = nn.Dropout2d()
        self.branch2_fc1 = nn.Linear(2000, 50)  # Adjusted for flattened output size
        
        # Final classifier
        self.final_fc = nn.Linear(100, 10)  # Output for CIFAR-10

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # Branch 1
        b1 = F.relu(F.max_pool2d(self.branch1_conv1(x), 2))
        b1 = F.relu(self.branch1_conv2(b1))
        b1 = F.relu(self.branch1_conv3(b1))
        b1 = self.branch1_drop(b1)
        b1 = b1.view(-1, self.num_flat_features(b1))
        b1 = F.relu(self.branch1_fc1(b1))
        
        # Branch 2
        b2 = F.relu(F.max_pool2d(self.branch2_conv1(x), 2))
        b2 = F.relu(self.branch2_conv2(b2))
        b2 = F.relu(self.branch2_conv3(b2))
        b2 = self.branch2_drop(b2)
        b2 = b2.view(-1, self.num_flat_features(b2))
        b2 = F.relu(self.branch2_fc1(b2))
        
        # Combine branches
        combined = torch.cat((b1, b2), dim=1)
        output = self.final_fc(combined)
        return F.log_softmax(output, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Create the model instance
model = TwoBranchCNN_CIFAR()
print(model)