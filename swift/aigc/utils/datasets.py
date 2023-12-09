import csv
import os
import random
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm


class AnimateDiffDataset(Dataset):

    VIDEO_ID = 'videoid'
    NAME = 'name'
    CONTENT_URL = 'contentUrl'

    def __init__(
        self,
        csv_path,
        video_folder,
        sample_size=256,
        sample_stride=4,
        sample_n_frames=16,
        dataset_sample_size=10000,
    ):
        print(f'loading annotations from {csv_path} ...')
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        dataset = []
        for d in tqdm(self.dataset):
            content_url = d[self.CONTENT_URL]
            file_name = content_url.split('/')[-1]
            if os.path.isfile(os.path.join(video_folder, file_name)):
                dataset.append(d)
            if dataset_sample_size is not None and len(
                    dataset) > dataset_sample_size:
                break

        self.dataset = dataset
        self.length = len(self.dataset)
        print(f'data scale: {self.length}')

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames

        sample_size = tuple(sample_size) if not isinstance(
            sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])

    def get_batch(self, idx):
        video_dict: Dict[str, str] = self.dataset[idx]
        name = video_dict[self.NAME]

        content_url = video_dict[self.CONTENT_URL]
        file_name = content_url.split('/')[-1]
        video_dir = os.path.join(self.video_folder, file_name)
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)

        clip_length = min(video_length,
                          (self.sample_n_frames - 1) * self.sample_stride + 1)
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx,
            start_idx + clip_length - 1,
            self.sample_n_frames,
            dtype=int)

        pixel_values = torch.from_numpy(
            video_reader.get_batch(batch_index).asnumpy()).permute(
                0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                logger.error(f'Error loading dataset batch: {e}')
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample