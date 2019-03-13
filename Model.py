import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from matplotlib import pyplot as plt


def tensor_to_gpu(tensor, is_cuda):
    if (is_cuda):
        return tensor.cuda()
    else:
        return tensor


def tensor_to_cpu(tensor, is_cuda):
    if (is_cuda):
        return tensor.cpu()
    else:
        return tensor


def flow_to_image(flow, original_image):
    use_cuda = torch.cuda.is_available()
    # forward path - interpolation
    X, Y = np.meshgrid(np.arange(flow.size(2)), np.arange(flow.size(3)))
    if use_cuda:
        meshgrid_tensor = torch.stack([torch.stack([torch.cuda.FloatTensor(Y), torch.cuda.FloatTensor(X)])] * original_image.size(0)).permute(0, 1, 3, 2)
    else:
        meshgrid_tensor = torch.stack([torch.stack([torch.FloatTensor(Y), torch.FloatTensor(X)])] * original_image.size(0)).permute(0, 1, 3, 2)
    flow += meshgrid_tensor
    # clamping
    flow_clamped = flow.clone()
    flow_clamped[:, 0] = flow_clamped[:, 0] / (original_image.size(3) - 1)
    flow_clamped[:, 1] = flow_clamped[:, 1] / (original_image.size(2) - 1)
    flow_clamped[:, 0] = flow_clamped[:, 0] * 2 - 1
    flow_clamped[:, 1] = flow_clamped[:, 1] * 2 - 1
    output_image = F.grid_sample(original_image, flow_clamped.permute(0, 2, 3, 1))
    return output_image


def show_images(original_image, output_image):
    original_image = tensor_to_cpu(original_image, torch.cuda.is_available())
    output_image = tensor_to_cpu(output_image, torch.cuda.is_available())
    for i in range(original_image.size(0)):
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.title('original image')
        plt.imshow(original_image[i].data.permute(1, 2, 0).numpy().astype('uint8'))
        fig.add_subplot(1, 2, 2)
        plt.title('deformed image')
        plt.imshow(output_image[i].data.permute(1, 2, 0).numpy().astype('uint8'))
        plt.show()


class NetBasicCNN(nn.Module):
    def __init__(self):
        super(NetBasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        original_image = x # keep for interpolation
        # forward path - until uxuy
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        #show_images(original_image, output_image)
        return output_image, uxuy

# stride 2 -> conv
class NetStride1(nn.Module):
    def __init__(self):
        super(NetStride1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64*4, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64*4)
        self.ps = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        original_image = x  # keep for interpolation
        # forward path - until uxuy
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.ps(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        #show_images(original_image, output_image)
        return output_image, uxuy


# conv -> stride 2 -> conv
class NetStride2(nn.Module):
    def __init__(self):
        super(NetStride2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64*4, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64*4)
        self.ps = nn.PixelShuffle(2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        original_image = x  # keep for interpolation
        # forward path - until uxuy
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.ps(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        #show_images(original_image, output_image)
        return output_image, uxuy

# stride 4 -> conv
class NetStride3(nn.Module):
    def __init__(self):
        super(NetStride3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64*16, 3, stride=4, padding=1)
        self.bn1 = nn.BatchNorm2d(64*16)
        self.ps = nn.PixelShuffle(4)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        original_image = x  # keep for interpolation
        # forward path - until uxuy
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.ps(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        #show_images(original_image, output_image)
        return output_image, uxuy


# conv -> stride 4 -> conv
class NetStride4(nn.Module):
    def __init__(self):
        super(NetStride4, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64*16, 3, stride=4, padding=1)
        self.bn2 = nn.BatchNorm2d(64*16)
        self.ps = nn.PixelShuffle(4)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        original_image = x  # keep for interpolation
        # forward path - until uxuy
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.ps(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        #show_images(original_image, output_image)
        return output_image, uxuy

# stride 2 & stride 4 -> conv
class NetStride5(nn.Module):
    def __init__(self):
        super(NetStride5, self).__init__()
        self.conv_s2 = nn.Conv2d(3, 64*4, 3, stride=2, padding=1)
        self.bn_s2 = nn.BatchNorm2d(64*4)
        self.ps2 = nn.PixelShuffle(2)
        self.conv_s4 = nn.Conv2d(3, 64*16, 3, stride=4, padding=1)
        self.bn_s4 = nn.BatchNorm2d(64*16)
        self.ps4 = nn.PixelShuffle(4)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        original_image = x  # keep for interpolation
        # forward path - until uxuy
        x2 = F.relu(self.bn_s2(self.conv_s2(x)))
        x2 = self.ps2(x2)
        x4 = F.relu(self.bn_s4(self.conv_s4(x)))
        x4 = self.ps4(x4)
        x = torch.cat((x2, x4), 1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        #show_images(original_image, output_image)
        return output_image, uxuy


# stride 2 -> conv (resnet)
class NetStride1_resnet(nn.Module):
    def __init__(self):
        super(NetStride1_resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64*4, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64*4)
        self.ps = nn.PixelShuffle(2)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        original_image = x  # keep for interpolation
        # forward path - until uxuy
        x = F.relu(self.bn1(self.conv1(x)))
        x_l1 = self.ps(x)
        x_l2 = F.relu(self.bn2(self.conv2(x_l1)))
        x_l3 = F.relu(self.bn3(self.conv3(x_l2)) + x_l1)
        x_l4 = F.relu(self.bn4(self.conv4(x_l3)) + x_l2)
        x_l5 = F.relu(self.bn5(self.conv5(x_l4)) + x_l3)
        x_l6 = F.relu(self.bn6(self.conv6(x_l5)) + x_l4)
        x_l7 = F.relu(self.bn7(self.conv7(x_l6)) + x_l5)
        x = self.conv8(x_l7)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        #show_images(original_image, output_image)
        return output_image, uxuy


# conv -> stride 2 -> conv (resnet)
class NetStride2_resnet(nn.Module):
    def __init__(self):
        super(NetStride2_resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64*4, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64*4)
        self.ps = nn.PixelShuffle(2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 2, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        original_image = x  # keep for interpolation
        # forward path - until uxuy
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_l1 = self.ps(x)
        x_l2 = F.relu(self.bn3(self.conv3(x_l1)))
        x_l3 = F.relu(self.bn4(self.conv4(x_l2)) + x_l1)
        x_l4 = F.relu(self.bn5(self.conv5(x_l3)) + x_l2)
        x_l5 = F.relu(self.bn6(self.conv6(x_l4)) + x_l3)
        x_l6 = F.relu(self.bn7(self.conv7(x_l5)) + x_l4)
        x = self.conv8(x_l6)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        #show_images(original_image, output_image)
        return output_image, uxuy


# stride 8 -> conv
class NetStride6(nn.Module):
    def __init__(self):
        super(NetStride6, self).__init__()
        self.conv1 = nn.Conv2d(3, 64 * 64, 3, stride=8, padding=1)
        self.bn1 = nn.BatchNorm2d(64 * 64)
        self.ps = nn.PixelShuffle(8)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        original_image = x  # keep for interpolation
        # forward path - until uxuy
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.ps(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        # show_images(original_image, output_image)
        return output_image, uxuy


# conv -> stride 2 -> conv (resnet)
class NetStride2_resnet_ker5(nn.Module):
    def __init__(self):
        super(NetStride2_resnet_ker5, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64 * 4, 5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64 * 4)
        self.ps = nn.PixelShuffle(2)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, 5, padding=2)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 2, 5, padding=2)

    def forward(self, x):
        original_image = x  # keep for interpolation
        # forward path - until uxuy
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_l1 = self.ps(x)
        x_l2 = F.relu(self.bn3(self.conv3(x_l1)))
        x_l3 = F.relu(self.bn4(self.conv4(x_l2)) + x_l1)
        x_l4 = F.relu(self.bn5(self.conv5(x_l3)) + x_l2)
        x_l5 = F.relu(self.bn6(self.conv6(x_l4)) + x_l3)
        x_l6 = F.relu(self.bn7(self.conv7(x_l5)) + x_l4)
        x = self.conv8(x_l6)
        uxuy = x.clone()
        output_image = flow_to_image(x, original_image)
        # debug - imshow
        # show_images(original_image, output_image)
        return output_image, uxuy