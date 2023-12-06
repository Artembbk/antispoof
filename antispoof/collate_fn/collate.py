import logging
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {
        "audio": [],
        'audio_path': [],
        "type": []
    }

    dataset_items = sorted(dataset_items, key=lambda x: -x["audio"].shape[1])

    for item in dataset_items: 
        result_batch["audio"].append(item["audio"])
        result_batch['audio_path'].append(item['audio_path'])
        result_batch['type'].append(item['type'])

    print(result_batch['audio'][0].shape)
    result_batch['audio'] = pad_sequence(result_batch['audio'], True, 0)
    result_batch['type'] = torch.tensor(result_batch["type"])
    
    return result_batch