from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer
import platform



class Multimodal_CelebA(Dataset):
    def __init__(self, data_folder, tokenizer, train_or_test='train', res=256):
        self.data = glob.glob(data_folder + "/*.npz")
        self.data.sort()
        self.train_or_test = train_or_test
        print('len(self.data):', len(self.data))

        self.tokenizer = tokenizer
        self.res = res


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        data = np.load(self.data[idx])
        image = data['image'].astype(np.float32)
        text = data['text']
        # print('text:', text)

        text = str(text)
        text_input = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")#.input_ids
        
        if False:
            print('image:', image.shape, image.min(), image.max(), text)
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.imshow(image[0])
            plt.subplot(1, 2, 2)
            plt.imshow(image[1])
            plt.show()
        
        if self.train_or_test == 'train':
            return image, text_input
        else:
            return image, text_input, text
        