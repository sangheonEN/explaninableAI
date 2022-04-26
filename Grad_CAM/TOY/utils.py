"""

forward hook()의 input output은 순전파일때 layer를 통과하기 전의 feature map value와 통과한 후 feature map value를 반환함.
backward hook()의 input output은 역전파일때 layer를 통과하기 전 Gradient와 통과한 후 Gradient를 반환함.

"""

class Hook():
    def __init__(self, module, back_ward=False):
        if back_ward == False:
            self.hook = module.register_forward_hook(self.forward_hookfn)
        else:
            self.hook = module.register_full_backward_hook(self.backward_hookfn)

    def forward_hookfn(self, module, input, output):
        self.inputf = input
        self.outputf = output

    def backward_hookfn(self, module, input, output):
        self.inputb = input
        self.outputb = output

    def close(self):
        self.hook.remove()

