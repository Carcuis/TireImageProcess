import glob
import os
import time
from typing import Callable

import cv2


class FPSCounter:
    def __init__(self):
        self._fps: float = 0
        self._count: int = 0
        self._last_time: float = time.time()

    def update(self):
        self._count += 1
        if self._count == 20:
            now_time = time.time()
            self._fps = self._count / (now_time - self._last_time)
            self._last_time = now_time
            self._count = 0

    @property
    def fps(self) -> float:
        return self._fps


class Util:
    fps_counter = FPSCounter()

    @staticmethod
    def create_window() -> None:
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera', 800, 600)

    @staticmethod
    def init() -> None:
        Util.create_window()

    @classmethod
    def show(cls, frame) -> None:
        cls.fps_counter.update()
        Util.display_txt(frame, f"FPS: {cls.fps_counter.fps:.2f}",
                         (10, frame.shape[0] - 10), (255, 255, 255), thickness=4)
        cv2.imshow('Camera', frame)

    @staticmethod
    def display_txt(frame, text: str, position: tuple[int, int], color: tuple[int, int, int],
                    font_size: float = 2.5, thickness: float = 6) -> None:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)

    @staticmethod
    def measure_time(func: Callable):
        def wrapper(*args, **kwargs):
            start = time.perf_counter_ns()
            res = func(*args, **kwargs)
            end = time.perf_counter_ns()
            print(f"Function {func.__name__} used {(end - start) / 1e6} ms.")
            return res
        return wrapper

    @staticmethod
    def get_time_str(fmt: str = "%Y%m%d_%H%M%S") -> str:
        return time.strftime(fmt, time.localtime())

    @staticmethod
    def get_date_str() -> str:
        return Util.get_time_str("%Y%m%d")

    @staticmethod
    def get_file_name_without_ext(file_path: str) -> str:
        return os.path.splitext(os.path.basename(file_path))[0]

    @staticmethod
    def ensure_folder_exist(*folders: str) -> None:
        for folder_name in folders:
            if not os.path.exists(folder_name):
                print(f"Create folder: {folder_name}")
                os.makedirs(folder_name)

    @staticmethod
    def generate_new_save_path(parent_dir: str) -> str:
        """
        生成一个新的保存路径: 父级目录/日期/序号(新)

        :param parent_dir: 父级目录
        :return: 新的保存路径
        """

        time_folder_str = Util.get_date_str()
        folder_name = os.path.join(os.getcwd(), parent_dir, time_folder_str)
        Util.ensure_folder_exist(folder_name)
        num = 1
        while True:
            path = os.path.join(folder_name, str(num))
            # 找到一个空文件夹
            if os.path.exists(path) and len(os.listdir(path)) > 0:
                num += 1
                continue
            Util.ensure_folder_exist(path)
            return path

    @staticmethod
    def get_file_list_in_dir_by_ext(dir_path: str, ext: str) -> list[str]:
        """
        获取指定目录下的所有指定文件类型

        :param dir_path: 图片所在目录
        :param ext: 文件类型后缀
        :return: 图片文件列表
        """
        return glob.glob(os.path.join(dir_path, f'*.{ext}'))
