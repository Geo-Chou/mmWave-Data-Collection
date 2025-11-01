import pyrealsense2 as rs
import numpy as np
import cv2
import os
import glob

data_root = "./data/"

bag_files =  glob.glob(os.path.join(data_root, "*.bag"))
for bag_file in bag_files:
    save_path = os.path.join(data_root, os.path.basename(bag_file).replace('.bag', ''), 'realsense')
    os.makedirs(save_path, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_file, repeat_playback=False)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    frame_id = 0
    try:
        while True:
            frame_id += 1
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not depth_frame or not color_frame:
                break

            timestamp = frames.get_timestamp()* 1e6  
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            np.save(os.path.join(save_path, f"{timestamp:.0f}.npy"), depth_image)
            cv2.imwrite(os.path.join(save_path, f"{timestamp:.0f}.png"), color_image)
    except Exception as e:
        print("End of bag:", e)
    finally:
        pipeline.stop()
        print("Finished processing bag file.")
        print("Total frames processed:", frame_id)
