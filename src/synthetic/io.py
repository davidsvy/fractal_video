import cv2
import numpy as np
import os
from PIL import Image
import shlex
import subprocess


def save_video_cv2(frames, path, fps, crf=24, scale=True):
    # frames -> [T, H, W, C]
    height, width = frames.shape[1: 3]
    is_rgb = frames.ndim > 3
    if scale:
        frames *= 255

    frames = frames.astype(np.uint8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(
        path, fourcc, fps, (width, height), is_rgb)

    for frame in frames:
        video.write(frame)
    video.release()


def save_video_ffmpeg(frames, path, fps, crf=24, scale=True):
    """Adapted from:
    https://www.faqcode4u.com/faq/126138/how-to-output-x265-compressed-video-with-cv2-videowriter

    ffmpeg formats list:
    https://ffmpeg.org/pipermail/ffmpeg-devel/2007-May/035617.html
    """
    # frames -> [T, H, W, C]
    height, width = frames.shape[1: 3]
    assert height % 2 == 0 and width % 2 == 0

    format = 'gray' if frames.ndim == 3 else 'bgr24'
    if scale:
        frames *= 255

    frames = frames.astype(np.uint8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    command = (
        f'ffmpeg -y -s {width}x{height} -pixel_format {format} '
        f'-f rawvideo -r {fps} -i pipe: -vcodec libx265 '
        f'-pix_fmt yuv420p -crf {crf} {path}'
    )

    process = subprocess.Popen(
        shlex.split(command), 
        stdin=subprocess.PIPE, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
    )
    
    for frame in frames:
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()
    process.terminate()
    
def save_video(frames, path, fps, crf=24, scale=True, lib='ffmpeg'):
    if lib == 'ffmpeg':
        save_video_ffmpeg(
            frames=frames, path=path, fps=fps, crf=crf, scale=scale)
    elif lib == 'cv2':
        save_video_cv2(
            frames=frames, path=path, fps=fps, crf=crf, scale=scale)
    else:
        raise ValueError(f'Unknwon video library: "{lib}"')


def save_imgs(paths, frames, scale=True):
    for path, frame in zip(paths, frames):
        if scale:
            frame = np.clip(frame, 0, 1) * 255
            
        frame = frame.astype(np.uint8)
        mode = 'RGB' if frame.ndim == 3 else 'L'
        frame = Image.fromarray(frame, mode=mode)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frame.save(path)