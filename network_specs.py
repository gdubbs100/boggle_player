import torch
from torch import nn
from torchvision.ops import distance_box_iou_loss, RoIPool


class ObjectDetector(nn.Module):

    def __init__(self, num_objects):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # bbox regressor
        self.fc1 = nn.Linear(16 * 101 * 101, 4 * num_objects)

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # all dimensions except for batch
        x = torch.relu(self.fc1(x))
        return x

class ResidualLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        y = torch.relu(self.fc(x))
        x = y + x
        return x

class ObjectDetector2(nn.Module):

    def __init__(self, num_objects):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 18, 6)
        # bbox regressor
        self.fc1 = nn.Linear(18 * 48 * 48, 1024)
        self.fc2 = nn.Linear(1024, 4 * num_objects)

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # all dimensions except for batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


class ObjectDetector2_res(nn.Module):

    def __init__(self, num_objects):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 18, 6)
        # bbox regressor
        self.fc1 = nn.Linear(18 * 48 * 48, 1024)
        self.res1 = ResidualLayer(1024)
        self.fc2 = nn.Linear(1024, 4 * num_objects)

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # all dimensions except for batch
        x = torch.relu(self.fc1(x))
        x = self.res1(x)
        x = torch.relu(self.fc2(x))
        return x

class ObjectDetector2_res_deep(nn.Module):

    def __init__(self, num_objects):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 18, 6)
        # bbox regressor
        self.fc1 = nn.Linear(18 * 48 * 48, 1024)
        self.res1 = ResidualLayer(1024)
        self.fc2 = nn.Linear(1024, 4 * num_objects)
        self.res2 = ResidualLayer(4 * num_objects)
        self.res3 = ResidualLayer(4 * num_objects)
        self.res4 = ResidualLayer(4 * num_objects)

    def forward(self,x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # all dimensions except for batch
        x = torch.relu(self.fc1(x))
        x = self.res1(x)
        x = torch.relu(self.fc2(x))
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return x

class ObjectDetectorROI(nn.Module):

    def __init__(self, num_objects):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.roi_pool = RoIPool(16, 2**-1)
        # bbox regressor
        self.fc1 = nn.Linear(16 * 101 * 101, 4 * num_objects)
        self.roi_conv1 = nn.Conv2d(3, 1, 4)
        self.fc2 = nn.Linear(169, 4 * num_objects)

    def forward(self,img):
        x = self.pool(torch.relu(self.conv1(img)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # all dimensions except for batch
        x = torch.relu(self.fc1(x))
        bboxes = torch.tensor_split(x.view(-1, 4, 16), 64, 0) # BATCH_SIZE = 64 hardcoded
        # print(img.size())
        # print(len(bboxes), bboxes[0].size())
        # print(list(bboxes))
        rois = self.roi_pool(img, list(bboxes))
        x = torch.relu(self.roi_conv1(rois))
        x = torch.flatten(x, 1)
        bboxes = torch.relu(self.fc2(x))
        # print(rois.size(), bboxes.size())
        # print(rois[0])

        return bboxes


# try using resnet pretrained model - takes a lot of memory
class resnet_ObjectDetector(nn.Module):

    def __init__(self, base_mod, num_objects):
        super().__init__()
        self.base_mod = base_mod
        self.fc1 = nn.Linear(1000, 4 * num_objects)
        self.fc2 = nn.Linear(4 * num_objects, 4 * num_objects)


    def forward(self,x):
        x = self.base_mod(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x