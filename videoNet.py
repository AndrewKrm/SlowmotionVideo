import torch as t
import torch.nn as nn
import cv2
import numpy as np

class PixelNet(nn.Module):
    def __init__(self):
        super(PixelNet, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = t.relu(self.fc1(x))
        x = t.relu(self.fc2(x))
        x = t.sigmoid(self.fc3(x))  # Use sigmoid to ensure output is in the range [0, 1]
        return x

class VideoNet(nn.Module):
    def __init__(self, height, width):
        super(VideoNet, self).__init__()
        self.height = height
        self.width = width
        print(f'\tCreating {height}x{width} video network')
        self.pixel_nets = nn.ModuleList([PixelNet() for _ in range(height * width)])
        print(f'\tCreated {height * width} pixel networks')
        # Use CUDA if it's available
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, pixel_idx):
        return self.pixel_nets[pixel_idx](x.to(self.device))

    def train_net(self, dataset, num_epochs=2000):
        print("\tStarting training...")
        optimizer = t.optim.Adam(self.parameters())
        criterion = nn.MSELoss()
        total_pixels = dataset.height * dataset.width
        print(f'\tTraining on {total_pixels} pixels')

        # Move the entire dataset to the device once
        dataset = [(t.unsqueeze(1).to(self.device), c.to(self.device)) for t, c in dataset]

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(total_pixels):
                print(f'\r\t\tProcessing pixel {i+1}/{total_pixels}', end='')
                time, color = dataset[i]  # Data and labels are already on the device

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self(time, i)
                loss = criterion(outputs, color)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / total_pixels
            print(f'\tEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss}')

        print('\tFinished Training')

    def save_model(self, path):
        t.save(self.state_dict(), path+str(self.height)+'x'+str(self.width)+'.pth')
        print(f'\tModel saved to {path}')

    def load_model(self, path):
        self.load_state_dict(t.load(path, map_location=self.device))  # Load model to the device
        print(f'\tModel loaded from {path}')

    def generate_video(self, time_steps):
        video = np.zeros((len(time_steps), self.height, self.width))
        time_steps = time_steps.unsqueeze(1).to(self.device)  # Make time_steps a 2D tensor and move it to the device
        for i in range(self.height * self.width):
            pixels = self(time_steps, i).detach().cpu().numpy()
            video[:, i // self.width, i % self.width] = pixels.squeeze()
        return video
    
    def save_video(self, video, path, fps=60):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(path, fourcc, fps, (self.width, self.height))
        for frame in video:
            frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
            out.write(frame_bgr)
        out.release()
        print(f'Video saved to {path}')
