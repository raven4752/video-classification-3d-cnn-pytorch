import torch
from torch.autograd import Variable

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding
import numpy as np


def classify_video(video_dir, video_name, class_names, model, opt):
    assert opt.mode == 'feature'

    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(video_dir, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)

    video_outputs = []
    video_segments = []
    with torch.no_grad():

        for i, (inputs, segments) in enumerate(data_loader):
            inputs = Variable(inputs)
            outputs = model(inputs)

            video_outputs.append(outputs.cpu().data)
            video_segments.append(segments)

    video_outputs = torch.cat(video_outputs)
    # video_segments = torch.cat(video_segments)
    results = []

    for i in range(video_outputs.size(0)):
        clip_results = np.expand_dims(video_outputs[i].numpy(), axis=0)

        results.append(clip_results)
    results = np.concatenate(results, axis=0)
    return results
