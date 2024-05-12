import cv2
import numpy as np
import torch as t
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, file_path, max_frames=None, frame_height=None, frame_width=None):
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.video_data = self._load_video(file_path, max_frames)
        self.time_length, self.height, self.width = self.video_data.shape  # Removed self.channels
        print(f"Loaded video '{file_path}' with dimensions: {self.video_data.shape}")

    def __len__(self):
        return self.time_length * self.height * self.width

    def __getitem__(self, idx):
        h = idx // self.width
        w = idx % self.width
    
        # Normalize time
        time = np.arange(self.time_length) / self.time_length
    
        # Get pixel intensity value across all frames
        output_data = self.video_data[:, h, w] / 255.0  # Assuming intensity values are in the range [0, 255]
    
        return (t.from_numpy(time.astype(np.float32)).to(self.device), 
            t.from_numpy(output_data.astype(np.float32)).unsqueeze(-1).to(self.device))  # Add an extra dimension to output_data
    def _load_video(self, file_path, max_frames):
        cap = cv2.VideoCapture(file_path)
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
                if self.frame_height is not None and self.frame_width is not None:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            else:
                break
        cap.release()
        video_data = np.array(frames)
        print(f"Video '{file_path}' loaded into memory with shape: {video_data.shape}")
        return video_data

    def play_video(self, window_name='Video Playback', delay=30):
        for frame in self.video_data:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()