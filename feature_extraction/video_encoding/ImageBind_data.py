#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import pkg_resources

import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
import cv2
from imagebind.models.multimodal_preprocessors import SimpleTokenizer
from torchvision.io import read_video
import torchvideo

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds


def return_bpe_path():
    return pkg_resources.resource_filename(
        "imagebind", "bpe/bpe_simple_vocab_16e6.txt.gz"
    )


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_outputs = []

    data_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=return_bpe_path())
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens


def load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)


def crop_boxes(boxes, x_offset, y_offset):
    """
    Perform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to perform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes



class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


def load_and_transform_video_data(
    video_paths,
    device,
    chunk_size_frames=64, 
):
    if video_paths is None:
        return None

    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224), 
            NormalizeVideo( 
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    spatial_cropper = SpatialCrop(crop_size=224, num_crops=1) #single crop from each frame (originally 3 for augmentation)

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        total_frames_from_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_from_cv2 = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if total_frames_from_cv2 == 0 or fps_from_cv2 == 0:
            print(f"Could not read total_frames ({total_frames_from_cv2}) or fps ({fps_from_cv2:.2f}) for {video_path}. Skipping.")
            empty = torch.empty((1, 3, 0, 224, 224), device=device)
            video_outputs.append(empty)
            continue

        try:
            video_object = EncodedVideo.from_path(
                video_path,
                decoder="decord",
                decode_audio=False,
            )
        except Exception as e:
            print(f"Error loading video {video_path}: {e}. Skipping.")
            empty = torch.empty((1, 3, 0, 224, 224), device=device)
            video_outputs.append(empty)
            continue
            
        video_duration_from_pv = video_object.duration # duration in seconds from pytorchvideo
        
        processed_video_chunks = []
        num_frames_processed = 0
        
        print(f"Processing video: {video_path}, CV2 Total Frames: {total_frames_from_cv2}, CV2 FPS: {fps_from_cv2:.2f}, PV Duration: {video_duration_from_pv:.2f}s")

        while num_frames_processed < total_frames_from_cv2:
            start_frame = num_frames_processed
            target_end_frame_exclusive = min(start_frame + chunk_size_frames, total_frames_from_cv2)
            
            # Calculate start and end times for get_clip
            start_time_sec = start_frame / fps_from_cv2
            end_time_sec = target_end_frame_exclusive / fps_from_cv2

            end_time_sec = min(end_time_sec, video_duration_from_pv)
            if start_time_sec >= end_time_sec and start_frame < target_end_frame_exclusive :
                 # failsafe
                 num_frames_this_chunk_ideal = target_end_frame_exclusive - start_frame
                 end_time_sec = start_time_sec + (num_frames_this_chunk_ideal / fps_from_cv2)
                 end_time_sec = min(end_time_sec, video_duration_from_pv)


            if start_time_sec >= video_duration_from_pv or start_time_sec >= end_time_sec:
                if num_frames_processed < total_frames_from_cv2:
                    print(f"  Stopping chunking for {video_path}: start_time {start_time_sec:.3f}s >= end_time {end_time_sec:.3f}s or video_duration {video_duration_from_pv:.3f}s. "
                          f"Processed {num_frames_processed}/{total_frames_from_cv2} frames.")
                break 
            
            try:
                clip_data = video_object.get_clip(start_sec=start_time_sec, end_sec=end_time_sec)
            except Exception as e:
                print(f"  Error during get_clip for {video_path} ({start_time_sec:.2f}s-{end_time_sec:.2f}s): {e}. Stopping chunks for this video.")
                break


            if clip_data is None or clip_data.get("video") is None or clip_data["video"].shape[1] == 0:
                if num_frames_processed < total_frames_from_cv2:
                     print(f"  Warning: get_clip returned 0 frames for {video_path} ({start_time_sec:.2f}s-{end_time_sec:.2f}s). "
                           f"Processed {num_frames_processed}/{total_frames_from_cv2}. Stopping chunks for this video.")
                break 
            
            video_chunk_tensor = clip_data["video"] # Shape: C, T_chunk_actual, H_orig, W_orig
            actual_frames_in_this_chunk = video_chunk_tensor.shape[1]
            
            video_chunk_tensor = video_chunk_tensor / 255.0
            transformed_chunk = video_transform(video_chunk_tensor) 
            
            spatially_cropped_chunk_list = spatial_cropper([transformed_chunk])
            processed_video_chunks.append(spatially_cropped_chunk_list[0])

            num_frames_processed += actual_frames_in_this_chunk


        if not processed_video_chunks:
            if total_frames_from_cv2 > 0 : 
                 print(f"Video {video_path} (expected {total_frames_from_cv2} frames) resulted in NO processed chunks.")
            full_video_tensor = torch.empty((3, 0, 224, 224), device=device) # C, T=0, H, W
        else:
            full_video_tensor = torch.cat(processed_video_chunks, dim=1) # C, T_total_processed, H_crop, W_crop
        
        # Final check on processed frames vs. expected frames from CV2
        if full_video_tensor.shape[1] != total_frames_from_cv2:
            print(f"Processed {full_video_tensor.shape[1]} frames, but cv2 reported {total_frames_from_cv2} frames. "
                  f"(Num frames added up: {num_frames_processed})")
        elif total_frames_from_cv2 > 0 : 
            print(f"Finished processing {video_path}: {full_video_tensor.shape[1]}/{total_frames_from_cv2} frames.")

        full_video_tensor = full_video_tensor.unsqueeze(0) # Shape: 1, C, T_total_processed, H_crop, W_crop
        video_outputs.append(full_video_tensor)

    if not video_outputs:
        pass

    return torch.stack(video_outputs, dim=0).to(device)        
        
        
        
        
"""        
        video_duration = video.duration
        clip_sampler = ConstantClipsPerVideoSampler(clip_duration=video.duration/6, clips_per_video=6)
        # Get number of frames from actual decoded clip
        decoded_clip = video.get_clip(0, video.duration)
        num_decoded_frames = decoded_clip["video"].shape[1]

        #frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=num_decoded_frames)


        all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)
        print(all_clips_timepoints)
        all_video = []
            #for clip_timepoints in all_clips_timepoints:
        for clip_timepoints in all_clips_timepoints:
            # read clip
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            #checkpoint for number of frames in clip
            num_frames = clip["video"].shape[1]
            print(f"Clip {i+1}: {num_frames} frames")
            i+=1
            #get frames
            video_clip = clip["video"]
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video = [video_transform(clip) for clip in all_video]
        all_video = SpatialCrop(224, num_crops=1)(all_video)

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0).to(device) 
"""