import os
import argparse
import cv2
import torch
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as tf

import utils
import model

def create_cam(config):
    if not os.path.exists(config.result_path):
        os.mkdir(config.result_path)

    test_loader, num_class = utils.get_testloader(config.dataset,
                                                  config.dataset_path,
                                                  config.img_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # CNN 클래스 생성자 cnn 선언해주고 -> 55 Line
    cnn = model.CNN(img_size=config.img_size, num_class=num_class).to(device)
    cnn.load_state_dict(
        torch.load(os.path.join(config.model_path, config.model_name))
                        )
    finalconv_name = "conv"

    feature_blobs = []

    def hook_feature(module, input, output):
        feature_blobs.append(output.cpu().data.numpy())

        """
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
        self.avg_pool = nn.AvgPool2d(img_size // 8)
        self.classifier = nn.Linear(num_class, num_class)
        
        # input image_size 128 // 8 = 16이니 16 x 16 kernel size로 average pooling하면,
        self.avg_pool = nn.AvgPool2d(img_size // 8)
        self.classifier = nn.Linear(num_class, num_class)
        
        1. model.py에서 정의한 sequence layer는 위와 같이 정의되는데 
        2. self.conv, self.avg_pool, self.classifier 이런식으로 nn.클래스 생성자의 명칭으로 지정된다.
        3. 그리고 나중에 내가 원하는 layer를 뽑아 내려면 cnn = CNN() 생성자를 선언하고 
        4. cnn._modules.get("layer명") -> ex cnn._modules.get("conv"), cnn._modules.get("avg_pool") 등등
        5. cnn._modules.get(finalconv_name).register_forward_hook(hook_feature): register_forward_hook->순전파일때 layer의 값을 확인함
        6. def hook_feature(module, input, output): -> output은 마지막 conv layer의 feature map의 값이 출력된다.
        7. 그래서 feature_blobs.append(output.cpu().data.numpy()) -> feature_blobs list에 GAP 이전 feature map을 저장해놓고
        8. cnn.parameters()[-2]로 -> FC layer의 weight를 불러와서
        9. 최종적으로 feature map and fc layer weight를 dot product 하여 cam을 구함. 
        """

    cnn._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(cnn.parameters())
    # get layer only from last layer (softmax layer) -> params[-2] -> softmax layer
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        """
        1. CAM의 수식인 weight_softmax, feature_conv의 행렬곱연산 후 CAM을 얻고
        2. 0-1 사이의 값을 가질 수 있게 normalization 실시 (cam - np.min(cam), cam / np.max(cam))
        3. 255를 곱해 visiualization 할수있게 Scale 보정

        :param feature_conv: Global Average Pooling 전 class num만큼 차원이 만들어진 종단 feature
        :param weight_softmax: softmax function 후 최종 weight map
        :param class_idx: softmax function 후 내가 제일 확신하는 class idx
        :return: Class Activation Map
        """
        size_upsample = (config.img_size, config.img_size)
        _, nc, h, w = feature_conv.shape
        output_cam = []
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    for i, (image_tensor, label) in enumerate(test_loader):
        image_PIL = tf.ToPILImage()(image_tensor[0])
        image_PIL.save(os.path.join(config.result_path, 'img%d.png' % (i + 1)))

        image_tensor = image_tensor.to(device)
        # cnn(image_tensor argment 전달하면 CNN의 forward() 메서드로 전달됨.) 모델 선언하고 모델에 텐서넣어주면 자동으로 forward가 실행됩니당 그렇게 되게 nn.module이 생겼어요 그래서 모델은 nn.module을 상속받아야하지요
        logit, _ = cnn(image_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        print("True label : %d, Predicted label : %d, Probability : %.2f" % (label.item(), idx[0].item(), probs[0].item()))
        CAMs = returnCAM(feature_blobs[0], weight_softmax, [idx[0].item()])
        # 결과 이미지를 불러와서 그 이미지에 cam value를 연산해줌.
        img = cv2.imread(os.path.join(config.result_path, 'img%d.png' % (i + 1)))
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(os.path.join(config.result_path, 'cam%d.png' % (i + 1)), result)
        if i + 1 == config.num_result:
            break
        feature_blobs.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR', choices=['STL', 'CIFAR', 'OWN'])
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--model_name', type=str, default='model.pth')

    parser.add_argument('--result_path', type=str, default='./result')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_result', type=int, default=1)

    config = parser.parse_args()
    print(config)

    create_cam(config)







