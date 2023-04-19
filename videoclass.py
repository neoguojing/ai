import torch
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.datasets.video_utils import VideoClips
from torch.utils.data import DataLoader

class TSN(torch.nn.Module):
    def __init__(self, num_classes, num_segments):
        super(TSN, self).__init__()
        self.num_segments = num_segments
        self.backbone = r3d_18(pretrained=True)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Reshape input tensor to (batch_size * num_segments, channel, depth, height, width)
        x = x.view((-1, self.num_segments) + x.shape[-3:])
        # Permute tensor to (batch_size, num_segments, channel, depth, height, width)
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous()
        # Flatten tensor to (batch_size * num_segments, channel, depth, height, width)
        x = x.view((-1,) + x.shape[2:])
        # Pass tensor through backbone
        x = self.backbone(x)
        # Reshape tensor to (batch_size, num_segments, num_classes)
        x = x.view((-1, self.num_segments, x.shape[-1]))
        # Average predictions across segments
        x = torch.mean(x, dim=1)
        # Pass tensor through softmax
        x = self.softmax(x)
        return x

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_list, clip_length, nclips, step_size, is_val=False):
        self.video_list = video_list
        self.clip_length = clip_length
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val
        self.video_clips = VideoClips(video_list, clip_length=clip_length, step_size=step_size)

    def __getitem__(self, index):
        video, audio, info, video_idx = self.video_clips.get_clip(index)
        video = self.normalize(video)
        label = self.get_label(video_idx)
        return video, label

    def __len__(self):
        return self.video_clips.num_clips()

    def get_label(self, video_idx):
        # Implement logic to get label from video_idx
        return label

    def normalize(self, video):
        # Implement normalization
        return video

class VideoDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(VideoDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self.collate

    def collate(self, batch):
        # Stack video tensors and labels
        videos, labels = zip(*batch)
        videos = torch.stack(videos, dim=0)
        labels = torch.tensor(labels)
        return videos, labels
