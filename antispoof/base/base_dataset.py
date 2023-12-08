import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from antispoof.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            spec_augs=None,
            limit=None
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs
        self.spec_augs = spec_augs
        self.target_length = 4*self.config_parser["preprocessing"]["sr"]

        index = self._filter_records_from_dataset(index, limit)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path).squeeze(0)
        if len(audio_wave) < self.target_length:
            repeats = self.target_length // len(audio_wave)
            remainder = self.target_length % len(audio_wave)
            audio_wave = audio_wave.repeat(repeats)
            audio_wave = torch.cat([audio_wave, audio_wave[:remainder]])
        else:
            audio_wave = audio_wave[:self.target_length]
        
        return {
            "audio": audio_wave,
            "audio_path": audio_path,
            "type": data_dict["type"]
        }

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            return audio_tensor_wave

    @staticmethod
    def _filter_records_from_dataset(
            index: list, limit
    ) -> list:
        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

def random_crop_tensor(tensor, crop_length):
    # Check if crop length is less than original tensor length
    # Calculate the maximum starting index for cropping

    # Generate a random starting index within the valid range
    start_idx = torch.randint(0, len(tensor) - crop_length, (1,)).item()

    # Perform the random crop
    cropped_tensor = tensor[start_idx:start_idx + crop_length]

    return cropped_tensor