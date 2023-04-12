import cv2
import numpy as np


class Straightener:
    def __init__(self) -> None:
        pass

    def straighten(self) -> None:
        # 1. 预处理
        img = cv2.imread("img\\20230412\\6\\20230412_112133.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 2. 轮廓检测
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 3. 筛选轮廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # 根据实际情况设置面积阈值
                continue
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if w < h:
                approx = approx.reshape((4, 2))
                approx = approx[[1, 2, 3, 0]]
            break

        # 4. 扭曲校正
        rect = cv2.minAreaRect(approx)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        M = cv2.getRotationMatrix2D(rect[0], angle, 1.0)
        img_rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

        # 5. 裁剪反投影
        dst_size = (int(rect[1][0]), int(rect[1][1]))
        dst_corners = np.array([[0, 0],
                                [dst_size[0], 0],
                                [dst_size[0], dst_size[1]],
                                [0, dst_size[1]]], dtype=np.float32)
        src_corners = cv2.boxPoints(rect).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        dst = cv2.warpPerspective(img_rot, M, dst_size)

        # 6. 输出
        cv2.imwrite("result.png", dst)


def main():
    straightener = Straightener()
    straightener.straighten()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
