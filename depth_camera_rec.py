import pyrealsense2 as rs
from more_itertools import time_limited
import itertools
import os

def recorder(SAVE_DIR, duration, stop_evt=None):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_record_to_file(f"{SAVE_DIR}/realsense/recorded.bag")


    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    for i in time_limited(duration, itertools.count()):

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
    
    pipeline.stop()

if __name__ == "__main__":
    SAVE_DIR = './data/'
    os.makedirs(os.path.join(SAVE_DIR, 'realsense'), exist_ok=True)
    RECORD_DURATION = 120  # seconds
    recorder(SAVE_DIR, RECORD_DURATION)