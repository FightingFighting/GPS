import torch
from timm.utils.clip_grad import dispatch_clip_grad


class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm',
                 parameters=None, create_graph=False,
                 mask=None, model=None, tuning_mode = None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)

        if tuning_mode == "part":
            self.maks_gradient(mask, model)
        self._scaler.step(optimizer)
        self._scaler.update()

    def maks_gradient(self, mask, model):
        for name, param in model.named_parameters():
            mask_ = mask[name]
            grad_tensor = param.grad.data
            grad_tensor = torch.where(mask_ == 1.0, mask_ - 1.0, grad_tensor)
            param.grad.data = grad_tensor


    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
