from typing import Tuple, Optional, List, Callable, Any

import torch.nn
import torchvision
from torch import nn, Tensor
from torchvision.models import Inception_V3_Weights, Inception3
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


class Nilception(torchvision.models.Inception3):
    def __init__(self, num_classes: int = 1000,
                 aux_logits: bool = True,
                 transform_input: bool = False,
                 inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
                 init_weights: Optional[bool] = None,
                 dropout: float = 0.5) -> None:
        super().__init__(num_classes, aux_logits, transform_input, inception_blocks, init_weights, dropout)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux


@handle_legacy_interface(weights=("pretrained", Inception_V3_Weights.IMAGENET1K_V1))
def nilception(*, weights: Optional[Inception_V3_Weights] = None, progress: bool = True, **kwargs: Any) -> Inception3:
    """
    Inception v3 model architecture from
    `Rethinking the Inception Architecture for Computer Vision <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        weights (:class:`~torchvision.models.Inception_V3_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.Inception_V3_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.Inception3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Inception_V3_Weights
        :members:
    """
    weights = Inception_V3_Weights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", True)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = Nilception(**kwargs)

    if weights is not None:
        # sd = model.state_dict()
        # wd = weights.get_state_dict(progress)
        ## TODO a veure si això la lia o no... Convertim els pesos de la primera conv [32,3,3,3] a [32, 1, 3, 3]
        # w0 = wd['Conv2d_1a_3x3.conv.weight'][:, 0, :, :]
        # w1 = wd['Conv2d_1a_3x3.conv.weight'][:, 1, :, :]
        # w2 = wd['Conv2d_1a_3x3.conv.weight'][:, 2, :, :]
        # conv0_weight = (w0+w1+w2)/3
        # conv0_weight = torch.reshape(conv0_weight, [32, 1, 3, 3])
        #
        # wd['Conv2d_1a_3x3.conv.weight'] = conv0_weight
        #
        # model.load_state_dict(wd)
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None

    return model
