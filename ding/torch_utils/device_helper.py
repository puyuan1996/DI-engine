"""
Copyright 2020 Sensetime X-lab. All Rights Reserved.

Device helper utilities for automatic detection of NPU and GPU devices.
Supports Huawei Ascend NPU (torch_npu) and NVIDIA GPU (torch.cuda).
"""

import torch
from typing import Tuple, Optional
import logging

# Try to import torch_npu for Huawei NPU support
try:
    import torch_npu
    TORCH_NPU_AVAILABLE = True
except ImportError:
    TORCH_NPU_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_available_device() -> Tuple[str, bool]:
    """
    Overview:
        Automatically detect the available device (NPU or GPU or CPU).
        Priority: NPU > GPU > CPU
    Returns:
        - device_type (:obj:`str`): Device type string, one of 'npu', 'cuda', 'cpu'
        - is_accelerator (:obj:`bool`): Whether an accelerator (NPU/GPU) is available
    Examples:
        >>> device_type, is_accelerator = get_available_device()
        >>> print(f"Using device: {device_type}")
    """
    # Check for NPU first (Huawei Ascend)
    if TORCH_NPU_AVAILABLE and torch.npu.is_available():
        npu_count = torch.npu.device_count()
        logger.info(f"Detected {npu_count} NPU device(s), using NPU")
        return 'npu', True

    # Check for CUDA GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected {gpu_count} CUDA GPU device(s), using GPU")
        return 'cuda', True

    # Fallback to CPU
    logger.info("No NPU or GPU detected, using CPU")
    return 'cpu', False


def get_device_count(device_type: str) -> int:
    """
    Overview:
        Get the number of available devices for the specified device type.
    Arguments:
        - device_type (:obj:`str`): Device type, one of 'npu', 'cuda', 'cpu'
    Returns:
        - count (:obj:`int`): Number of available devices
    """
    if device_type == 'npu' and TORCH_NPU_AVAILABLE:
        return torch.npu.device_count()
    elif device_type == 'cuda':
        return torch.cuda.device_count()
    else:
        return 1  # CPU always has 1 "device"


def move_to_device(model: torch.nn.Module, device_type: str, rank: int = 0) -> torch.nn.Module:
    """
    Overview:
        Move a PyTorch model to the specified device.
        Supports NPU, CUDA, and CPU devices.
    Arguments:
        - model (:obj:`torch.nn.Module`): The model to move
        - device_type (:obj:`str`): Device type, one of 'npu', 'cuda', 'cpu'
        - rank (:obj:`int`): Device rank for multi-device setups
    Returns:
        - model (:obj:`torch.nn.Module`): The model moved to the device (in-place operation)
    """
    if device_type == 'npu' and TORCH_NPU_AVAILABLE:
        device_count = torch.npu.device_count()
        device_id = rank % device_count if device_count > 0 else 0
        model.npu(device_id)
        logger.debug(f"Moved model to NPU device {device_id}")
    elif device_type == 'cuda':
        device_count = torch.cuda.device_count()
        device_id = rank % device_count if device_count > 0 else 0
        model.cuda(device_id)
        logger.debug(f"Moved model to CUDA device {device_id}")
    # CPU case: no need to move
    return model


def get_device_string(device_type: str, rank: int = 0) -> str:
    """
    Overview:
        Get the device string for PyTorch tensor operations.
    Arguments:
        - device_type (:obj:`str`): Device type, one of 'npu', 'cuda', 'cpu'
        - rank (:obj:`int`): Device rank for multi-device setups
    Returns:
        - device_str (:obj:`str`): Device string like 'npu:0', 'cuda:0', or 'cpu'
    """
    if device_type in ['npu', 'cuda']:
        device_count = get_device_count(device_type)
        device_id = rank % device_count if device_count > 0 else 0
        return f'{device_type}:{device_id}'
    else:
        return 'cpu'


def auto_device_init(cfg_device: Optional[str], rank: int = 0) -> Tuple[str, bool, str]:
    """
    Overview:
        Initialize device settings based on config.
        Supports automatic detection, explicit device type, or legacy 'cuda' boolean.
    Arguments:
        - cfg_device (:obj:`Optional[str]`): Device configuration from config.
            Can be 'auto', 'npu', 'cuda', 'cpu', or None (defaults to 'auto')
        - rank (:obj:`int`): Device rank for multi-device setups
    Returns:
        - device_type (:obj:`str`): Detected device type ('npu', 'cuda', or 'cpu')
        - use_accelerator (:obj:`bool`): Whether an accelerator is being used
        - device_str (:obj:`str`): Full device string for PyTorch operations
    Examples:
        >>> device_type, use_accelerator, device_str = auto_device_init('auto')
        >>> # Returns ('npu', True, 'npu:0') if NPU available
        >>> # Returns ('cuda', True, 'cuda:0') if GPU available
        >>> # Returns ('cpu', False, 'cpu') otherwise
    """
    # Default to auto detection if not specified
    if cfg_device is None or cfg_device == 'auto':
        device_type, use_accelerator = get_available_device()
    else:
        # Explicit device type specified
        device_type = cfg_device.lower()

        # Validate the device type is available
        if device_type == 'npu':
            if TORCH_NPU_AVAILABLE and torch.npu.is_available():
                use_accelerator = True
                logger.info("Using NPU as explicitly configured")
            else:
                logger.warning("NPU requested but not available, falling back to CPU")
                device_type = 'cpu'
                use_accelerator = False
        elif device_type == 'cuda':
            if torch.cuda.is_available():
                use_accelerator = True
                logger.info("Using CUDA GPU as explicitly configured")
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                device_type = 'cpu'
                use_accelerator = False
        else:
            # CPU or any other value
            device_type = 'cpu'
            use_accelerator = False
            logger.info("Using CPU as configured")

    device_str = get_device_string(device_type, rank)

    return device_type, use_accelerator, device_str


def is_npu_available() -> bool:
    """
    Overview:
        Check if Huawei NPU is available.
    Returns:
        - available (:obj:`bool`): True if NPU is available
    """
    return TORCH_NPU_AVAILABLE and torch.npu.is_available()


def is_cuda_available() -> bool:
    """
    Overview:
        Check if NVIDIA CUDA GPU is available.
    Returns:
        - available (:obj:`bool`): True if CUDA is available
    """
    return torch.cuda.is_available()
