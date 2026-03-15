"""
Neural-Guided Beamformer.
Real-time speech enhancement via neural priors and restricted TFLC optimization.
"""

from .inference import enhance_audio

__all__ = ["enhance_audio"]
