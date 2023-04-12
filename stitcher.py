import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from util import Util


class Stitcher:
    def __init__(self):
        self.top_padding = 100
        self.bottom_padding = 100
        self.left_padding = 0
        self.right_padding = 800

    def stitch(self, path1: str, path2: str):
        """
        进行图片变换并拼接

        :param path1:
        :param path2:
        """
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
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                matches_mask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask,
                           flags=0)
        img3 = cv2.drawMatchesKnn(img1gray, kp1, img2gray, kp2, matches, None, **draw_params)
        plt.imshow(img3, ), plt.show()

        MIN_MATCH_COUNT = 4
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            warp_img = cv2.warpPerspective(test_img, np.array(M), (test_img.shape[1], test_img.shape[0]),
                                           flags=cv2.WARP_INVERSE_MAP)
            self.blend(src_img, warp_img)
        else:
            print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
            # matches_mask = None

    def blend(self, src_img, warp_img):
        """
        图片融合
        """
        rows, cols = src_img.shape[:2]
        # 找到左右重叠区域
        for col in range(0, cols):
            if src_img[:, col].any() and warp_img[:, col].any():
                self.left_padding = col
                break
        for col in range(cols - 1, 0, -1):
            if src_img[:, col].any() and warp_img[:, col].any():
                self.right_padding = col
                break

        res = np.zeros([rows, cols, 3], np.uint8)
        alpha = np.zeros((rows, self.right_padding - self.left_padding, 3), dtype=float)
        for row in range(0, rows):
            for col in range(self.left_padding, self.right_padding):
                if not src_img[row, col].any():  # src不存在
                    alpha[row, col - self.left_padding, :] = 0
                elif not warp_img[row, col].any():  # warpImg 不存在
                    alpha[row, col - self.left_padding, :] = 1
                else:  # src 和warp都存在
                    src_img_len = float(abs(col - self.left_padding))
                    test_img_len = float(abs(col - self.right_padding))
                    alpha[row, col - self.left_padding, :] = test_img_len / (src_img_len + test_img_len)

        res[:, :self.left_padding] = src_img[:, :self.left_padding]
        res[:, self.right_padding:] = warp_img[:, self.right_padding:]
        res[:, self.left_padding:self.right_padding] = np.clip(
            src_img[:, self.left_padding:self.right_padding] * alpha +
            warp_img[:, self.left_padding:self.right_padding] * (np.ones_like(alpha) - alpha),
            0, 255
        )

        # opencv is bgr, matplotlib is rgb
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        file_name = f"stitch/stitch_{Util.get_time_str()}.jpg"
        plt.imsave(file_name, res)
        plt.imshow(res)
        plt.show()


def main():
    stitcher = Stitcher()

    path1 = "img\\20230412\\6\\20230412_112133.jpg"
    path2 = "img\\20230412\\6\\20230412_112136.jpg"

    start = time.time()
    stitcher.stitch(path1, path2)
    end = time.time()
    print("Spent time: ", end - start)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
