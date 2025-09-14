# Import Checker and Circle classes from main.py
from main import Checker, Circle, Spectrum

def create_patterns():
    # Checkerboard pattern
    checker_board = Checker(resolution=200,tile_size=20)
    checker_board.draw()
    checker_board.show()

    #Circle pattern
    circle = Circle(resolution=200, radius=20,position=(50,50))
    circle.draw()
    circle.show()

    #Spectrum
    spectrum = Spectrum(resolution=256)
    spectrum.draw()
    spectrum.show()

if __name__ == "__main__":
    create_patterns()
