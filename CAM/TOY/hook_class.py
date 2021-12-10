"""
 - Deep Neural Network was composed many layer

 - So, we extract the layer value -> feature map value

 - because, we utilize the layer value and weight value.

 - pytorch offer each layers of feature value.

 - It is named Hook Class

 - step: Model Net -> Layer approach -> extract layer feature map

"""

import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_1 = nn.Linear(2, 2)
        self.sig_1 = nn.Sigmoid()
        self.fc_2 = nn.Linear(2, 2)
        self.sig_2 = nn.Sigmoid()

        self.fc_1.weight = torch.nn.Parameter(torch.Tensor([[0.15, 0.2], [0.250, 0.30]]))
        self.fc_1.bias = torch.nn.Parameter(torch.Tensor([0.35]))
        self.fc_2.weight = torch.nn.Parameter(torch.Tensor([[0.4, 0.45], [0.5, 0.55]]))
        self.fc_2.bias = torch.nn.Parameter(torch.Tensor([0.6]))


    def forward(self, x):
        x = self.fc_1(x)
        x = self.sig_1(x)
        x = self.fc_2(x)
        x = self.sig_2(x)
        return x


class Hook():
    def __init__(self, module, backward=False):

        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


if __name__ == "__main__":

    net = Net()
    print(net)

    # model layer save to list
    list_layer = []
    for name, layer in net._modules.items():
        list_layer.append(layer)
    print(list_layer)

    # model layer argment to Hook Class -> forward or backward
    # backward와 forward의 layer 선언 순서는 다르다.
    # forward는 앞 layer부터 순차적으로 존재하고 backward는 뒷 layer부터 순차적으로 존재
    hook_f = Hook(list_layer[0], backward=False)
    hook_b = Hook(list_layer[0], backward=True)

    # Throw data -> check the result
    data = torch.Tensor([0.05, 0.1])

    out = net(data)
    """
    https://hongl.tistory.com/158?category=927704 -> retain_graph 설명
    """
    out.backward(torch.tensor([1,1], dtype=torch.float32), retain_graph=True)


    print("debug_flag_1")


























