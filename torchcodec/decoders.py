from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch


class AudioDecoder:
    def __init__(self, uri: str | pathlib.Path):
        self._path = pathlib.Path(uri)
        self.metadata = SimpleNamespace(sample_rate=None)
        self._data = None

    def get_all_samples(self) -> SimpleNamespace:
        samples, sr = sf.read(str(self._path), always_2d=True)
        self.metadata.sample_rate = sr
        tensor = torch.from_numpy(samples.T.astype(np.float32))
        return SimpleNamespace(data=tensor)
