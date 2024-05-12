import torch as t
from videoDataset import VideoDataset
from videoNet import VideoNet

TRAIN=True

def main():
    if TRAIN:
        dataset = VideoDataset('videos/andrewditcoucou.mp4', frame_height=13, frame_width=24)
        net = VideoNet(dataset.height, dataset.width)
        print("NN created")
        net.train_net(dataset)
        print("NN trained")
        net.save_model('models/andrewditcoucou')
        print("NN saved")
        video=net.generate_video(t.arange(0, 1, 0.01))
        net.save_video(video, 'videos/andrewditcoucou_generated.mp4')
    else:
        net = VideoNet(9, 16)
        net.load_model('models/andrewditcoucou9x16.pth')
        dataset = VideoDataset('videos/andrewditcoucou.mp4', frame_height=9, frame_width=16)
        net.train_net(dataset)
        print("NN trained")
        net.save_model('models/andrewditcoucou')
        print("NN saved")
        video=net.generate_video(t.arange(0, 1, 0.01))
        net.save_video(video, 'videos/andrewditcoucou_generated.mp4')
        video=net.generate_video(t.arange(0, 1, 0.01))
        net.save_video(video, 'videos/andrewditcoucou_generated.mp4')

if __name__ == '__main__':
    main()