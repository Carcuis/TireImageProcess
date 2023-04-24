import os
import time

import cv2
from matplotlib import pyplot as plt

from util import Util


class Straightener:
    def __init__(self) -> None:
        self.save_path = Util.generate_new_save_path(os.path.join("img", "straighten"))

    def straighten(self, img_path: str, preview: bool = True, save: bool = True) -> None:
        """
        读取图片进行拼接

        :param img_path: 图片路径
        :param preview: 是否预览
        :param save: 是否保存
        """

        print(f"Straightening: {img_path}")
        img = cv2.imread(img_path)

        # 找到轮廓
        center = (img.shape[1] // 2, img.shape[0] + 750)
        radius = int(img.shape[1] * 1.15)
        # cv2.circle(img, center, radius, (0, 0, 255), 4)

        polar = cv2.warpPolar(img, (10000, 10000), center, radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG)
        polar_gray = cv2.cvtColor(polar, cv2.COLOR_BGR2GRAY)
        # 找到非0部分的坐标
        coords = cv2.findNonZero(polar_gray)
        # 计算非0部分的矩形区域
        x, y, w, h = cv2.boundingRect(coords)
        # 裁剪图片并放大
        crop = polar[y:y + h, x:x + w]
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        crop = crop[0:crop.shape[0] // 2, 0:crop.shape[1]]
        # cv2.rectangle(result, (580, 0), (result.shape[1] - 580, result.shape[0]), (0, 0, 255), 4)
        result = crop[0:crop.shape[0], 580:crop.shape[1] - 580]

        # 显示结果
        if preview:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.show()

        if save:
            file_name = f"{self.save_path}/straighten_{Util.get_file_name_without_ext(img_path)}.jpg"
            print(f"Saving to {file_name}...")
            cv2.imwrite(file_name, result)

    def straighten_from_folder(self, img_folder: str, preview: bool = True, save: bool = True) -> None:
        """
        从文件夹中读取图片进行拼接

        :param img_folder: 图片文件夹
        :param preview: 是否预览
        :param save: 是否保存
        """

        print(f"Straightening from folder: {img_folder}")
        for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder, img_name)
            start = time.time()
            self.straighten(img_path, preview, save)
            end = time.time()
            print(f"--Time spent: {end - start}")


def main():
    straightener = Straightener()
    start = time.time()
    straightener.straighten_from_folder("img/tire/20230412/5", preview=True, save=True)
    end = time.time()
    print("Total time spent: ", end - start)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
