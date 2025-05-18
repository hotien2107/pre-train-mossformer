from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset, Dataset


@dataclass
class DatasetConfig:
    dataset_name: str = "audio-team/speech-separation-dataset"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    sample_rate: int = 8000
    max_duration: float = 4.0  # giây
    num_sources: int = 2  # số lượng nguồn âm

class SpeechSeparationDataset(Dataset):
    def __init__(
            self,
            split: str = "train",
            config: DatasetConfig = DatasetConfig()
    ):
        """
        Args:
            split: Loại split ('train', 'validation', 'test')
            config: Cấu hình dataset
        """
        self.config = config
        self.split = self._validate_split(split)
        self.max_samples = int(config.sample_rate * config.max_duration)

        # Load dataset từ Hugging Face
        self.dataset = load_dataset(
            config.dataset_name,
            split=self.split,
            trust_remote_code=True
        )

        # Validate cấu trúc dataset
        self._validate_dataset_structure()

    def _validate_split(self, split):
        """Đảm bảo split hợp lệ và chuyển đổi tên nếu cần"""
        split = split.lower()
        if split == "val":
            return "validation"
        if split == "dev":
            return "validation"
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'validation', 'test']")
        return split

    def _validate_dataset_structure(self):
        """Kiểm tra cấu trúc dataset có đúng định dạng"""
        sample = self.dataset[0]
        required_keys = {'mix', 'sources'}

        if not required_keys.issubset(sample.keys()):
            missing = required_keys - sample.keys()
            raise KeyError(f"Dataset missing required keys: {missing}")

        if len(sample['sources']) != self.config.num_sources:
            raise ValueError(
                f"Dataset contains {len(sample['sources'])} sources "
                f"but expected {self.config.num_sources}"
            )

    def __len__(self):
        return len(self.dataset)

    def _process_audio(self, audio):
        """Xử lý audio array"""
        # Chuyển đổi sang numpy array và chuẩn hóa
        audio = np.asarray(audio, dtype=np.float32)

        # Cắt hoặc padding
        if len(audio) > self.max_samples:
            start = np.random.randint(0, len(audio) - self.max_samples)
            return audio[start:start+self.max_samples]
        else:
            return np.pad(audio, (0, max(0, self.max_samples - len(audio))),
                          mode='constant')

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Xử lý mix
        mix = self._process_audio(sample['mix']['array'])

        # Xử lý các nguồn âm
        sources = [
            self._process_audio(s['array'])
            for s in sample['sources'][:self.config.num_sources]
        ]

        return {
            'mix': torch.from_numpy(mix),
            'sources': torch.stack([torch.from_numpy(s) for s in sources]),
            'sample_id': sample.get('id', str(idx)),
            'metadata': sample.get('metadata', {})
        }

    @staticmethod
    def collate_fn(batch):
        """Tạo batch từ các sample có độ dài khác nhau"""
        max_length = max(x['mix'].shape[0] for x in batch)

        padded_batch = {
            'mix': [],
            'sources': [],
            'sample_ids': [],
            'metadata': []
        }

        for item in batch:
            # Padding cho mix
            padded_mix = torch.nn.functional.pad(
                item['mix'],
                (0, max_length - item['mix'].shape[0])
            )

            # Padding cho các nguồn âm
            padded_sources = torch.stack([
                torch.nn.functional.pad(s, (0, max_length - s.shape[0]))
                for s in item['sources']
            ])

            padded_batch['mix'].append(padded_mix)
            padded_batch['sources'].append(padded_sources)
            padded_batch['sample_ids'].append(item['sample_id'])
            padded_batch['metadata'].append(item['metadata'])

        return {
            'mix': torch.stack(padded_batch['mix']),
            'sources': torch.stack(padded_batch['sources']),
            'sample_ids': padded_batch['sample_ids'],
            'metadata': padded_batch['metadata']
        }
