import numpy as np

class Position:
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y

    def distance(self, position):
        return np.sqrt((self.x-position.x)**2 + (self.y-position.y)**2)

    def __add__(self, o):
        self.x += o.x
        self.y += o.y
        return 
    
    def __str__(self) -> str:
        return f"{self.x}:{self.y}"
    
    def to_tuple(self) -> tuple:
        return (self.x, self.y)