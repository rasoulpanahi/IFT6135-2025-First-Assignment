'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt


class BasicBlock(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        """
            :param in_planes: input channels
            :param planes: output channels
            :param stride: The stride of first conv
        """
        super(BasicBlock, self).__init__()
        # Uncomment the following lines, replace the ? with correct values.
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # 1. Go through conv1, bn1, relu
        # 2. Go through conv2, bn
        # 3. Combine with shortcut output, and go through relu
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(out)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, stride):
        layers = []
        layers.append(BasicBlock(in_planes, planes, stride))
        layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, images):
        """ input images and output logits """
        out = F.relu(self.bn1(self.conv1(images)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.avg_pool2d(out, 4)  # Global Average Pooling (GAP)
        out = torch.flatten(out, 1)  # Flatten before Linear Layer
        logits = self.linear(out)
        return logits


    def visualize(self, logdir: str, layer_name: str):
        """ Visualize the kernel in the desired directory """

        # Find the layer by the name
        layer = dict(self.named_modules())[layer_name]

        # Ensure the layer has weight data
        if isinstance(layer, nn.Conv2d):
            # Extract the weights (kernels) from the layer
            kernels = layer.weight.data.cpu().numpy()

            # Normalize the kernel values for better visualization
            min_kernel = kernels.min()
            max_kernel = kernels.max()
            kernels = (kernels - min_kernel) / (max_kernel - min_kernel)  # Normalize between 0 and 1

            # Get the number of kernels and the size of each kernel
            num_kernels = kernels.shape[0]
            kernel_size = kernels.shape[2]  # Assuming square kernels, e.g., 3x3

            # Set up the grid for visualization
            grid_size = int(num_kernels ** 0.5)  # Arrange kernels in a square grid
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

            for i, ax in enumerate(axes.flat):
                if i < num_kernels:
                    # Plot each kernel (grayscale image)
                    ax.imshow(kernels[i, 0, :, :], cmap='gray')
                    ax.axis('off')
                else:
                    ax.axis('off')  # Empty axes if we don't have enough kernels

            # Save the figure in the specified logdir
            os.makedirs(logdir, exist_ok=True)
            plt.suptitle(f"Kernels of {layer_name}")
            save_path = os.path.join(logdir, f"{layer_name}_kernels.png")
            plt.savefig(save_path)
            plt.close(fig)
            print(f"Kernel visualization saved to {save_path}")
        else:
            print(f"Layer {layer_name} is not a Conv2d layer, skipping visualization.")