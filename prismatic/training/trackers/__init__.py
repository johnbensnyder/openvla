"""
tensorboard.py

TensorBoard tracker with support for scalar metrics and video logging.
"""

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


class TensorBoardTracker:
    def __init__(self, run_id: str, run_dir: Path, hparams: Dict[str, Any]) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.hparams = hparams
        self.writer = None
        self.initialize()

    @overwatch.rank_zero_only
    def initialize(self) -> None:
        log_dir = self.run_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(log_dir))

    @overwatch.rank_zero_only
    def write_hyperparameters(self) -> None:
        if self.writer:
            self.writer.add_text("hparams", str(self.hparams), 0)

    @overwatch.rank_zero_only
    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, global_step)

    @overwatch.rank_zero_only
    def write_video(self, tag: str, video: np.ndarray, global_step: int, fps: int = 10) -> None:
        """
        Log video to TensorBoard.
        
        Args:
            tag: Video tag/name
            video: numpy array of shape (T, H, W, C) with uint8 values [0, 255]
            global_step: Training step
            fps: Frames per second for playback
        """
        if self.writer and video is not None and len(video) > 0:
            # TensorBoard expects (N, T, C, H, W) format
            video_tensor = np.transpose(video, (0, 3, 1, 2))  # (T, H, W, C) -> (T, C, H, W)
            video_tensor = np.expand_dims(video_tensor, 0)  # (1, T, C, H, W)
            self.writer.add_video(tag, video_tensor, global_step, fps=fps)

    @overwatch.rank_zero_only
    def flush(self) -> None:
        if self.writer:
            self.writer.flush()

    def finalize(self) -> None:
        if overwatch.is_rank_zero() and self.writer:
            self.writer.close()
