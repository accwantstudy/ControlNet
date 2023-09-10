import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

class MyCustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        for label in range(10):  # assuming you have 10 classes
            class_folder = os.path.join(root_dir, f'class{label}')
            for filename in os.listdir(class_folder):
                if filename.endswith('.jpg') or filename.endswith('.png'):  # assuming images are in jpg or png format
                    self.image_paths.append(os.path.join(class_folder, filename))
                    self.labels.append(label)
                    
        self.white_image = torch.ones((64, 64, 3), dtype=torch.float32)
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        # Get the label as text
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        label_text = classes[label]
        
        # Apply transformations
        image = np.array(image, dtype=np.float32)/127.5 - 1

        return dict(jpg=image, hint = self.white_image, txt=label_text)


if __name__ == '__main__':
    dataset = MyCustomDataset(root_dir='data\cifar10')

    # You can then use this dataset with a DataLoader to handle batching, shuffling, etc.
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    print(batch['jpg'].shape)
    print(batch['txt'])
    print(batch['hint'].shape)