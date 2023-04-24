import threading
import time
import os

import cv2
from numpy.typing import NDArray

from util import FPSCounter, Util


class Camera(threading.Thread):
    def __init__(self, camera_id: int = 0) -> None:
        super().__init__()
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError("Error: Cannot open camera")
        self.set_camera_props()

        self._frame: NDArray = self.cap.read()[1]
        self.frame_height: int = self._frame.shape[0]
        self.frame_width: int = self._frame.shape[1]
        self.frame_dim: int = self._frame.shape[2]
        print(f"Frame size: {self.frame_width}x{self.frame_height}, dim: {self.frame_dim}")

        self.daemon = True
        self._lock = threading.Lock()
        self.time_after_last_shoot: float = time.time()
        self.photos_num: int = 0
        self.photo_save_path: str = ""
        self.fps_counter = FPSCounter()
        self.inner_fps: float = 0

        self.start()

    def set_camera_props(self) -> None:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        self.cap.set(cv2.CAP_PROP_GAIN, 255)
        print(f"Camera inner FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Camera exposure: {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        print(f"Camera gain: {self.cap.get(cv2.CAP_PROP_GAIN)}")

    def run(self) -> None:
        while True:
            self.update_frame()
            self.fps_counter.update()
            self.inner_fps = self.fps_counter.fps

    def update_frame(self) -> None:
        success, frame = self.cap.read()
        with self._lock:
            self._frame = frame

    def get_new_frame(self) -> NDArray:
        success, frame = self.cap.read()
        return frame

    @property
    def frame(self) -> NDArray:
        with self._lock:
            return self._frame.copy()

    def take_photos(self, interval: int, total: int) -> bool:
        """
        在间隔 interval 秒内，拍摄 total 张照片

        :param interval: 间隔时间，单位秒
        :param total: 总张数
        :return: 是否拍摄完成
        """

        if self.photos_num >= total:
            self.photos_num = 0
            self.photo_save_path = ""
            return True
        if time.time() - self.time_after_last_shoot <= interval:
            return False
        self.time_after_last_shoot = time.time()

        # 以当前时间作为文件名
        time_filename_str = Util.get_time_str()
        filename = f"{time_filename_str}.jpg"

        if self.photo_save_path == "":
            # 保存在img/tire/[date]/[num]文件夹下
            new_dir = Util.generate_new_save_path(os.path.join("img", "tire"))
            print(f"Save photos to: {new_dir}")
            self.photo_save_path = new_dir

        # 保存图像
        filename = os.path.join(self.photo_save_path, filename)
        cv2.imwrite(filename, self.frame)
        self.photos_num += 1
        return False

    def __del__(self) -> None:
        print("Camera released.")
        self.cap.release()
