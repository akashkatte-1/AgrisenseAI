import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
#  RESNET9 ARCHITECTURE
#  (MATCHES YOUR .PTH FILE)
# -----------------------

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        # follow training architecture: downsample between blocks
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        # -----------------------------------------
        # CRITICAL FIX → MATCHES YOUR TRAINED MODEL
        # classifier.0 = MaxPool2d
        # classifier.1 = Flatten
        # classifier.2 = Linear(512 → 38)
        # -----------------------------------------
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = out + self.res1(out)

        out = self.conv3(out)
        out = self.conv4(out)
        out = out + self.res2(out)

        out = self.classifier(out)
        return out


