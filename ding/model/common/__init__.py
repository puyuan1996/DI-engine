<<<<<<< HEAD
from .head import DiscreteHead, DuelingHead, DuelingHeadM, DistributionHead, RainbowHead, QRDQNHead, \
    QuantileHead, RegressionHead, ReparameterizationHead, MultiHead, head_cls_map
from .encoder import ConvEncoder, FCEncoder
=======
from .head import DiscreteHead, DuelingHead, DistributionHead, RainbowHead, QRDQNHead, \
    QuantileHead, FQFHead, RegressionHead, ReparameterizationHead, MultiHead, head_cls_map
from .encoder import ConvEncoder, FCEncoder, IMPALAConvEncoder
>>>>>>> e0aa2b24cec226db18c64e237177df391d13e26c
from .utils import create_model
