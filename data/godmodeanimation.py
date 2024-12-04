from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer
import platform


class GodModeAnimation(Dataset):
    def __init__(self, data_folder, tokenizer, train_or_test='train', num_frames=16, res=256):
        self.data = glob.glob(data_folder + "/*.npz")
        self.data.sort()
        self.train_or_test = train_or_test
        print('GodModeAnimation len(self.data):', len(self.data))
        self.tokenizer = tokenizer
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename = self.data[idx]
        data = np.load(filename)
        video = data['video'].astype(np.float32)
        text = data['text']
        
        start_frame = 0
        # if video.shape[0] - self.num_frames == 0:
        #     start_frame = 0
        # else:
        #     start_frame = np.random.randint(0, video.shape[0]-self.num_frames)
        video = video[start_frame:start_frame + self.num_frames]
        
        text = str(text)
        text_input = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")#.input_ids

        if False:
            print('video:', video.shape, video.min(), video.max())
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.imshow(video[0, 0])
            plt.subplot(1, 2, 2)
            plt.imshow(video[1, 0])
            plt.show()

        if self.train_or_test == 'train':
            return video, text_input
        else:
            return video, text_input, text
