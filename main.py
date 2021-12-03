import cv2
from math import sin, pi
from numba import njit
import numpy as np
from typing import Callable


class Sin2D:
    def __init__(self, omega_x: float = 1, phi_x: float = 0, omega_y: float = 1, phi_y: float = 0):
        self.omega_x = omega_x
        self.phi_x = phi_x

        self.omega_y = omega_y
        self.phi_y = phi_y

    def x(self, t: float) -> float:
        return sin(self.omega_x * t + self.phi_x)

    def y(self, t: float) -> float:
        return sin(self.omega_y * t + self.phi_y)


class Variable:
    def __init__(self, name: str, start: float, end: float):
        self.name = name
        self.start = start
        self.end = end


class Lissajous:
    def __init__(self, f: Sin2D, w: int = 800, h: int = 800, variables: set[Variable, ] = None, info: bool = True):
        self.f = f
        self.w = w
        self.h = h
        self.variables = [] if variables is None else variables
        self.info = info

    @staticmethod
    def linmap(value, actual_bounds, desired_bounds):
        return desired_bounds[0] + (value - actual_bounds[0]) * (desired_bounds[1] - desired_bounds[0]) / (
                    actual_bounds[1] - actual_bounds[0])

    def str(self, num: int, length: int) -> str:
        string = str(num)
        return (length - len(string)) * '0' + string

    def var(self, variable: str, start: float, end: float, file: str = 'output.mp4', frames: int = 300, fps: int = 30, **kwargs):
        len_frames = max(len(str(frames)), 4)
        out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.h, self.w), False)
        for i in range(frames):
            self.f.__setattr__(variable, self.linmap(i, (0, frames), (start, end)))
            arr = self.img(**kwargs)
            out.write(arr)
            if self.info:
                print(f'\r[INFO] render | {round(100 * (i + 1)/frames)}%', end='')
        out.release()

    def render(self, seconds: float = 10, file: str = 'output.mp4', codec: str = 'mp4v', fade: float = 1, steps: float = 1, t_max: float = 2 * pi, fps: int = 30) -> np.ndarray:
        arr: np.ndarray = np.zeros((self.h, self.w))
        steps = round(self.w * self.h * steps * seconds / 50)
        mult = 1 - 25 * fade/steps

        out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*codec), fps, (self.h, self.w), False)
        for step in range(steps):
            t = t_max * step/steps
            for variable in self.variables:
                self.f.__setattr__(variable.name, self.linmap(step, (0, steps), (variable.start, variable.end)))
            x, y = self.f.x(t), self.f.y(t)
            x, y = round(self.linmap(x, (-1.02, 1.02), (0, self.w))), round(self.linmap(y, (1.02, -1.02), (0, self.h)))
            if 0 <= x < self.w and 0 <= y < self.h:
                arr[y, x] = 255
                arr *= mult
            if step % round(steps/(seconds * fps)) == 0:
                print(f'\r[INFO] render | {round(100 * step/steps, 2)}%', end='')
                out.write(arr.astype(np.uint8))
        print(f'\r[INFO] render | released to {file!r}', end='')
        out.release()
        return arr


if __name__ == '__main__':
    sin2d = Sin2D()
    lissajous = Lissajous(sin2d, 1080, 1080, {Variable('omega_x', 2.256, 7.1325), Variable('omega_y', 3.1262, 9.1523)})
    lissajous.render(15, t_max=64*pi, steps=4)
