import pyrealsense2 as rs2
import numpy as np
import cv2


file = "./data/20230518_111059.bag"
config = rs2.config()
pipeline = rs2.pipeline()

config.enable_device_from_file(file, repeat_playback=False)
config.enable_stream(rs2.stream.color, rs2.format.rgb8, 30)

pipeline_profile = pipeline.start(config)
playback = pipeline_profile.get_device().as_playback()
playback.set_real_time(False)
count = 0
success = True

while success:
    try:
        frames = pipeline.wait_for_frames()
        fnum = str(frames.get_frame_number())

        color_frame = frames.get_color_frame()

        rgb_image = np.asanyarray(color_frame.get_data())
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        imgname = "data/cimg-" + fnum.zfill(3) + ".png"
        # Render image in opencv window
        cv2.imwrite(imgname, bgr_image)

    except:
        success = False
