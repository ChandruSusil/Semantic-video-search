import cv2
import os
import shutil
from typing import List, Tuple

class FrameExtractor:
    def __init__(self, extraction_interval: int = 2):
        """
        Args:
            extraction_interval: Extract one frame every X seconds.
        """
        self.extraction_interval = extraction_interval

    def extract_frames(self, video_path: str, output_folder: str) -> List[Tuple[str, float]]:
        """
        Extracts frames from the video.
        Returns a list of tuples: (frame_filepath, timestamp_in_seconds).
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        else:
            # Clear existing frames to avoid confusion
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Timestamp in seconds
            timestamp = frame_count / fps
            
            # Check if we should extract this frame (based on interval)
            # We want roughly one frame every `extraction_interval` seconds
            # Simplest way: check if timestamp is close to a multiple of interval
            if int(frame_count % int(fps * self.extraction_interval)) == 0:
                # Skip black/dark frames (often transitions or errors)
                if frame.mean() < 5.0: # Threshold for "blackness"
                    frame_count += 1
                    continue

                filename = f"frame_{len(frames_data):04d}.jpg"
                filepath = os.path.join(output_folder, filename)
                cv2.imwrite(filepath, frame)
                frames_data.append((filepath, timestamp))
            
            frame_count += 1
            
        cap.release()
        return frames_data
