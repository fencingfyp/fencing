from moviepy import VideoFileClip
import sys
import os

def crop_video_time(input_path, output_path, start_time, end_time):
    # Load video
    clip = VideoFileClip(input_path)
    duration = clip.duration

    # Clamp end_time if it exceeds duration
    if end_time > duration:
        print(f"⚠️ End time ({end_time}s) exceeds video length ({duration:.2f}s). Using {duration:.2f}s instead.")
        end_time = duration

    # Ensure valid range
    if start_time >= end_time or start_time < 0:
        raise ValueError("Invalid start/end time range.")

    # Crop and save
    subclip = clip.subclipped(start_time, end_time)
    subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    clip.close()
    subclip.close()
    print(f"✅ Saved cropped video to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python crop_video_time.py input.mp4 output.mp4 start_time end_time")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    start = float(sys.argv[3])
    end = float(sys.argv[4])

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    crop_video_time(input_file, output_file, start, end)
