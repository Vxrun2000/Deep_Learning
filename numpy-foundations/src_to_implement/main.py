import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
from generator import ImageGenerator

class Checker:
    def __init__(self, resolution,tile_size):
        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution must be divisible by 2")
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):  # Instance method
        num_tiles = self.resolution//self.tile_size
        base_tile = np.array([[0,1],[1,0]])
        checkerboard = np.tile(base_tile,(num_tiles//2,num_tiles//2))
        self.output = np.kron(checkerboard,np.ones((self.tile_size,self.tile_size)))
        return self.output.copy()

    def show(self):  # Instance method
        if self.output is not None:
            plt.imshow(self.output, cmap='grey')
            plt.title("Checkerboard pattern")
            plt.axis('off') 
        plt.show()


class Circle:
    def __init__(self,resolution, radius, position):
        pass  # Instance method
        self.resolution= resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):  # Instance method
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        xx,yy = np.meshgrid(x,y)
        dist = (xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2
        circle = dist <= self.radius ** 2
        self.output = circle.astype(np.uint8)  # 0 (black) or 1 (white)
        return self.output.copy()
        

    def show(self):  # Instance method
        if self.output is not None:
            plt.imshow(self.output,cmap='grey')
            plt.title("Circle pattern")
            plt.axis('off')
            plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        # Create linearly spaced values from 0 to 1
        r = np.linspace(0.0, 1.0, self.resolution)
        g = np.linspace(0.0, 1.0, self.resolution)  
        b = np.linspace(1.0, 0.0, self.resolution)
        #Create 2D arrays
        R = np.tile(r, (self.resolution, 1))                # Red varies horizontally
        G = np.tile(g[:, np.newaxis], (1, self.resolution)) # Green varies vertically
        B = np.tile(b, (self.resolution, 1))  

        self.output = np.stack((R, G, B), axis=2) #axis 2 beacuse of color channels
        return self.output.copy()

    def show(self):
        if self.output is not None:
            plt.imshow(self.output)
            plt.axis('off')
            plt.title('RGB Color Spectrum')
            plt.show()


if __name__ == "__main__":
    generator = ImageGenerator(
        file_path=r"C:\Users\Varun\Downloads\src_to_implement\data\exercise_data",                 
        label_path=os.path.join(os.getcwd(),'Labels.json'), 
        batch_size=12,
        image_size=[64,64,3],
        rotation=False,
        mirroring=False,
        shuffle=True
    )

    generator.show()
