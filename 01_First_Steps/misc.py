import subprocess
import wget
from pathlib import Path
import os
import time

def download_video(video_url, max_attempts=10, retry_delay=1.0):
    time.sleep(2.0)
    os.makedirs("videos", exist_ok=True)

    # The server may still be finalizing the recording right after the experiment
    # stops, in which case the download is empty. Retry until we get actual data.
    for attempt in range(max_attempts):
        input_path = wget.download(video_url, out="videos/")
        if os.path.getsize(input_path) > 0:
            break
        os.remove(input_path) # so that wget reuses the same file name next time
        time.sleep(retry_delay)
    else:
        raise RuntimeError(f"Downloaded video is still empty after {max_attempts} attempts: {video_url}")

    output_path = Path(input_path).parent / (str(Path(input_path).stem) + '.mp4')
    convert_flv_to_mp4(input_path, output_path)
    return output_path

def convert_flv_to_mp4(input_path, output_path):
    """
    Convert an FLV file to MP4 using FFmpeg.

    :param input_path: Path to the input FLV file.
    :param output_path: Path to the output MP4 file.
    """
    time.sleep(2.0)

    command = [
        "ffmpeg",
        "-i", input_path,    # Input file
        "-c:v", "copy",      # Copy video stream
        "-c:a", "copy",      # Copy audio stream
        output_path          # Output file
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode == 0:
        print(f"Conversion successful: {output_path}")
    else:
        print(f"Error during conversion: {process.stderr.decode()}")