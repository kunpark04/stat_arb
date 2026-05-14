from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeParams:
    leverage:        float
    entry_threshold: float
    exit_threshold:  float
    stop_loss:       float
    kappa_min:       float
    max_position:    float
    vol_target:      float


REGIME_PARAMS = {
    0: RegimeParams(leverage=1.2,  entry_threshold=1.75, exit_threshold=0.50, stop_loss=3.0, kappa_min=4.0, max_position=0.18, vol_target=0.12),
    1: RegimeParams(leverage=1.0,  entry_threshold=2.00, exit_threshold=0.50, stop_loss=3.0, kappa_min=5.0, max_position=0.15, vol_target=0.10),
    2: RegimeParams(leverage=0.3,  entry_threshold=2.75, exit_threshold=0.75, stop_loss=2.5, kappa_min=8.0, max_position=0.08, vol_target=0.05),
}

DEFAULT_REGIME = 1  # used during warmup NaN period
