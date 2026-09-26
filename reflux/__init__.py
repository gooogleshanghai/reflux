"""reflux_min: minimal deep-to-shallow feedback for frozen Transformers."""

from .hooks import Capture, Inject, decoder_layers, shift_prev
from .write import feedback_write
from .streaming import ReFluxStreaming
from .recirculation_v1 import RecirculationV1
from .windows import build_windows
from .metrics import window_ppl
from .models import load_model

__all__ = [
    "Capture", "Inject", "decoder_layers", "shift_prev",
    "feedback_write", "ReFluxStreaming", "RecirculationV1",
    "build_windows", "window_ppl", "load_model",
]
