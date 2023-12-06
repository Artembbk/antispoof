import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from antispoof.base.base_dataset import BaseDataset
from antispoof.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)

class ASVDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        self._data_dir = Path(data_dir)
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        if part == "train":
            index_path = self._data_dir / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{part}.trn.txt"
        else:
            index_path = self._data_dir / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{part}.trl.txt"

        with index_path.open() as f:
            index = f.readlines()
            index = self.convert_to_dicts(index)
        return index
    
    def convert_to_dicts(self, input_list):
        result = []
        for line in input_list:
            parts = line.split()
            if len(parts) >= 4:
                path = self._data_dir / "ASVspoof2019_LA_cm_protocols" / f"{parts[1]}.flac"
                file_type = 1 if parts[-1].strip() == 'bonafide' else 0
                result.append({"path": path, "type": file_type})
        return result

