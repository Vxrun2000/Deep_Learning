import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.transform import resize
from sklearn.utils import shuffle as sklearn_shuffle

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
    
        # Load labels from JSON.Dictionary has mapping of image file name(as 15,8..) to label(integer)
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        # Get Image ids[Keys].
        self.image_ids = list(self.labels.keys())
        self.num_images = len(self.image_ids)
        if self.shuffle:
            random.shuffle(self.image_ids)

        #Initialize iteration variables

        self.index = 0
        self.epoch = 0
        self.epoch_completed = False 

    def next(self):

        image_batch = []
        label_batch = []
        self.epoch_completed = False
    
        for _ in range(self.batch_size):
          if self.index >= self.num_images:
              self.index = 0
              if self.shuffle:
                  random.shuffle(self.image_ids)
              self.epoch += 1
              self.epoch_completed = True
        
          img_id = self.image_ids[self.index] #Image id
          img = np.load(os.path.join(self.file_path, f"{img_id}.npy"))
          label = self.labels[img_id] #Label

          img_resized = resize(img, self.image_size, mode='reflect', anti_aliasing=True)#Resize
          img_resized = self.augment(img_resized)
          image_batch.append(img_resized)
          label_batch.append(label)

          self.index += 1     
          
        image_batch = np.array(image_batch)
        label_batch = np.array(label_batch)

        return image_batch,label_batch
        

    def augment(self,img):
        
        if self.mirroring and random.choice([True, False]):
          img = np.fliplr(img)

       # Random rotation by 90, 180, or 270 degrees if rotation is enabled
        if self.rotation:
          k = random.choice([1, 2, 3])  # 0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°
          img = np.rot90(img, k)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch

    def class_name(self, label):
        return self.class_dict.get(label,"Not found")
        
    def show(self):
        
        images, labels = self.next()
        batch_size = images.shape[0]
        rows = 3
        cols =4
        fig, axes = plt.subplots(rows,cols,figsize=(cols*3,rows*3))
        axes = axes.flatten()
        for i in range(batch_size):
                ax = axes[i] if batch_size > 1 else axes
                ax.imshow(images[i]) 
                ax.set_title(self.class_name(labels[i]))
                ax.axis('off')

        for j in range(batch_size, len(axes)):
            axes[j].axis('off')

        plt.show()

            

