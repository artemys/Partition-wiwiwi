from __future__ import annotations

import pathlib

import soundfile as sf
import torch


class AudioEncoder:
    def __init__(self, data: torch.Tensor, *, sample_rate: int):
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        self._data = data.detach().cpu()
        self._sample_rate = sample_rate

    def to_file(self, uri: str | pathlib.Path, *, bit_rate: int | None = None) -> None:
        target = pathlib.Path(uri)

        tensor = self._data
        if tensor.ndim == 1:
            samples = tensor.numpy()
        else:
            if tensor.shape[0] <= tensor.shape[1]:
                samples = tensor.numpy().T
            else:
                samples = tensor.numpy().T
        sf.write(str(target), samples, self._sample_rate)
