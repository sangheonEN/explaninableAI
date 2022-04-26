import torch
import torch.nn as nn
from collections import OrderedDict

class CNN(nn.Module):
    def __init__(self, img_size, num_class):
        super(CNN, self).__init__()
        # self.conv = nn.ModuleDict({
        #     'conv1' : nn.Conv2d(3, 32, 3, 1, 1),
        #     'leakyrelu1' : nn.LeakyReLU(0.2),
        #     'conv2' : nn.Conv2d(32, 64, 3, 1, 1),
        #     'leakyrelu2' : nn.LeakyReLU(0.2),
        #     'maxpool1' : nn.MaxPool2d(2, 2),
        #     'conv3' : nn.Conv2d(64, 128, 3, 1, 1),
        #     'leakyrelu3' : nn.LeakyReLU(0.2),
        #     'conv4' : nn.Conv2d(128, 256, 3, 1, 1),
        #     'leakyrelu4' : nn.LeakyReLU(0.2),
        #     'maxpool2' : nn.MaxPool2d(2, 2),
        #     'conv5' : nn.Conv2d(256, 512, 3, 1, 1),
        #     'leakyrelu5' : nn.LeakyReLU(0.2),
        #     'maxpool3' : nn.MaxPool2d(2, 2),
        #     'conv6' : nn.Conv2d(512, num_class, 3, 1, 1),
        #     'leakyrelu6' : nn.LeakyReLU(0.2)
        # })

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512, num_class, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

        # input image_size 128 // 8 = 16이니 16 x 16 kernel size로 average pooling하면,
        self.avg_pool = nn.AvgPool2d(img_size // 8)
        self.classifier = nn.Linear(num_class, num_class)

    def forward(self, x):
        features = self.conv(x)
        flatten = self.avg_pool(features).view(features.size(0), -1)
        # avg_pool 후 class num에 맞게 차원이 맞춰진다고 하더라도, Fully Connected Layer를 수행해야함.
        # 왜냐하면 avg_pool 수행한 결과는 단순히 그전 feature map에서 kernel size로 pooling 한것임. 그래서 feature를 모두 연결 지을 수 없음.
        # 따라서, FC layer를 통해서 확보한 feature 모두를 연결시켜서 연산함.
        output = self.classifier(flatten)
        return output, features

    def save_output_forward_hooking(self, layer_id):
        """
        augment
        hook(module, input, output) -> None or modified output
        """
        forwardhook = self.conv[layer_id].register_forward_hook()

        print(forwardhook)

    def save_output_forward_prehooking(self, layer_id):
        forwardprehook = self.conv[layer_id].register_forward_pre_hook()

        print(forwardprehook)

    def save_gradient_backward_hooking(self, layer_id):
        backwardhook = self.conv[layer_id].register_full_backward_hook()

        print(backwardhook)




