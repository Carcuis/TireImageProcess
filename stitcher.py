import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from util import Util


class Stitcher:
    def __init__(self, top: int = 0, bottom: int = 50, left: int = 0, right: int = 600) -> None:
        self.top_padding = top
        self.bottom_padding = bottom
        self.left_padding = left
        self.right_padding = right
        self.save_path = Util.generate_new_save_path("stitch")

    def _stitch_manually(self, path1: str, path2: str, show_match=True) -> tuple[bool, NDArray]:
        """
        使用手动图片变换并拼接

        :param path1:
        :param path2:
        :param show_match: 是否显示匹配点
        :return: 是否拼接成功, 拼接后的图片
        """

        print(f"Stitching {path1} and {path2} using manual algorithm...")

        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)

        img1_size = img1.shape[:2]
        img2 = cv2.resize(img2, (img1_size[1], img1_size[0]))

        src_img = cv2.copyMakeBorder(
            img1,
            self.top_padding, self.bottom_padding, self.left_padding, self.right_padding,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        test_img = cv2.copyMakeBorder(
            img2,
            self.top_padding, self.bottom_padding, self.left_padding, self.right_padding,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        img1gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        img2gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        # find the key points and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1gray, None)
        kp2, des2 = sift.detectAndCompute(img2gray, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        # bf = cv2.BFMatcher()  # 采用暴力特征匹配
        # matches = bf.knnMatch(des1, des2, k=2)  # 匹配特征点
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matches_mask = [[0, 0] for _ in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
                matches_mask[i] = [1, 0]

        if show_match:
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matches_mask,
                               flags=0)
            match_result = cv2.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
            plt.imshow(cv2.cvtColor(match_result, cv2.COLOR_BGR2RGB))
            plt.show()

        MIN_MATCH_COUNT = 4
        if len(good) <= MIN_MATCH_COUNT:
            print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
            return False, NDArray

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        warp_img = cv2.warpPerspective(test_img, np.array(M), (test_img.shape[1], test_img.shape[0]),
                                       flags=cv2.WARP_INVERSE_MAP)
        return True, self._blend(src_img, warp_img)

    @staticmethod
    def _blend(src_img, warp_img) -> NDArray:
        """
        图片融合

        :param src_img: 左边原图
        :param warp_img: 右边将要变换的图
        :return: 融合后的图
        """
        rows, cols = src_img.shape[:2]

        left, right = 0, 0
        # 找到左右重叠区域
        for col in range(0, cols):
            if src_img[:, col].any() and warp_img[:, col].any():
                left = col
                break
        for col in range(cols - 1, 0, -1):
            if src_img[:, col].any() and warp_img[:, col].any():
                right = col
                break

        res = np.zeros([rows, cols, 3], np.uint8)
        alpha = np.zeros((rows, right - left, 3), dtype=float)
        for row in range(0, rows):
            for col in range(left, right):
                if not src_img[row, col].any():  # src 不存在
                    alpha[row, col - left, :] = 0
                elif not warp_img[row, col].any():  # warpImg 不存在
                    alpha[row, col - left, :] = 1
                else:  # src 和 warp 都存在
                    src_img_len = float(abs(col - left))
                    test_img_len = float(abs(col - right))
                    alpha[row, col - left, :] = test_img_len / (src_img_len + test_img_len)

        res[:, :left] = src_img[:, :left]
        res[:, right:] = warp_img[:, right:]
        res[:, left:right] = np.clip(
            src_img[:, left:right] * alpha +
            warp_img[:, left:right] * (np.ones_like(alpha) - alpha),
            0, 255
        )
        return res

    @staticmethod
    def _stitch_using_cv2_algorithm(img_list: list) -> tuple[bool, NDArray]:
        """
        使用 cv2.Stitcher 进行拼接

        :param img_list: 已经读取到内存的图像 list
        :return: 拼接成功, 拼接结果
        """

        modes = (cv2.Stitcher_PANORAMA, cv2.Stitcher_SCANS)
        stitcher = cv2.Stitcher.create(modes[1])
        try:
            status, result = stitcher.stitch(img_list)
        except cv2.error as e:
            print(f"Can't stitch images, error: {e}")
            return False, NDArray

        if status != cv2.Stitcher_OK:
            print(f"Can't stitch images, error code = {status}")
            return False, NDArray
        return True, result

    def stitch_two(self, img_folder: str, step: int = 1, preview=True, save=True, use_cv2_algorithm=False) -> None:
        """
        从文件夹中依次读取图片进行两两拼接

        :param img_folder: 图片文件夹
        :param step: 两次拼接之间的间隔
        :param preview: 是否预览
        :param save: 是否保存
        :param use_cv2_algorithm: 是否使用 opencv 内置的 Stitcher
        """

        print(f"Start stitching from folder: {img_folder}")
        img_list = os.listdir(img_folder)
        for i in range(0, len(img_list) - 1, step):
            img1_path = os.path.join(img_folder, img_list[i])
            img2_path = os.path.join(img_folder, img_list[i + 1])
            print(f"Start stitching {img1_path} and {img2_path}")
            start = time.time()
            if use_cv2_algorithm:
                success, result = self._stitch_using_cv2_algorithm([cv2.imread(img1_path), cv2.imread(img2_path)])
                file_name = f"{self.save_path}/bundle_stitch_{Util.get_time_str()}.jpg"
            else:
                success, result = self._stitch_manually(img1_path, img2_path, show_match=True)
                img1_file_name = Util.get_file_name_without_ext(img1_path)
                img2_file_name = Util.get_file_name_without_ext(img2_path)
                file_name = f"{self.save_path}/stitch_{img1_file_name}_{img2_file_name[-6:]}.jpg"

            end = time.time()
            print(f"--Time spent: {end - start}")

            if success:
                if save:
                    print(f"--Done. Save to: {file_name}")
                    cv2.imwrite(file_name, result)
                if preview:
                    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                    plt.show()

    def stitch_all_using_bundled_algorithm(self, img_folder: str, preview=True, save=True) -> None:
        """
        使用 opencv 内置的 Stitcher 从文件夹中读取全部图片进行一次性拼接

        :param img_folder: 图片文件夹
        :param preview: 是否预览
        :param save: 是否保存
        """

        imgs: list[NDArray] = []
        img_list = os.listdir(img_folder)
        print("Reading images...", end=" ")
        for i in range(0, len(img_list)):
            img = cv2.imread(os.path.join(img_folder, img_list[i]))
            imgs.append(img)
        print("done.")

        print("Start stitching using cv2.Stitcher...")
        success, result = self._stitch_using_cv2_algorithm(imgs)
        if success:
            if save:
                file_name = f"{self.save_path}/bundle_stitch_all_{Util.get_time_str()}.jpg"
                print(f"--Done. Save to: {file_name}")
                cv2.imwrite(file_name, result)
            if preview:
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.show()


def main():
    stitcher = Stitcher()

    img_folder = "straighten/20230415/2"

    start = time.time()
    stitcher.stitch_two(img_folder, step=1, preview=True, save=True, use_cv2_algorithm=False)
    # stitcher.stitch_all_using_bundled_algorithm(img_folder, preview=True, save=True)
    end = time.time()
    print("Total time spent: ", end - start)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
