from PIL import Image

# Load image
img = Image.open(r"phd-project/sea horse.png").convert("RGBA")

# Convert white to transparent
datas = img.getdata()
new_data = []
for item in datas:
    # If pixel is "white enough", make it transparent
    if item[0] > 20 and item[1] > 20 and item[2] > 20:
        new_data.append((255, 255, 255, 0))  # transparent
    else:
        new_data.append(item)

# Save new image
img.putdata(new_data)
img.save(r"phd-project/sea horse transparent.png", "PNG")

# %%
import numpy as np
import cv2

video_path = r"media/videos/phd_project/1080p60/Microscope.mp4"
def save_last_frame(video_path, output_path="last_frame.png"):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set position to last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    # Read the last frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read the last frame from the video.")

    # Save frame as PNG
    cv2.imwrite(output_path, frame)
    print(f"Last frame saved to {output_path}")

save_last_frame(video_path, r"phd_project/last_frame.png")

# %%
import matplotlib.colors as mcolors
def rgb_to_hue(r, g, b):
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c

    if delta == 0:
        return 0
    elif max_c == r:
        hue = ((g - b) / delta) % 6
    elif max_c == g:
        hue = (b - r) / delta + 2
    else:
        hue = (r - g) / delta + 4

    return hue / 6  # Normalize to [0, 1]


rgb = mcolors.to_rgb("#e48f47")  # (r, g, b) in [0, 1]
TARGET_HUE = rgb_to_hue(*rgb)