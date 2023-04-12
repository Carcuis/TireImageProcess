import cv2

from camera import Camera
from util import Util


def main():
    Util.init()
    cam = Camera(0)

    finish = False
    capture = False
    photo_total = 20

    while True:
        frame = cam.frame

        if not capture and not finish:
            Util.display_txt(frame, f"Press C to take {photo_total} photos", (10, 80), (255, 255, 0))
        elif capture and not finish:
            Util.display_txt(frame, f"Taking photos... {cam.photos_num}/{photo_total}", (10, 80), (255, 0, 255))
        elif finish:
            Util.display_txt(frame, "Finish, press C to restart", (10, 80), (0, 255, 0))

        if capture:
            finish = cam.take_photos(3, photo_total)
            if finish:
                capture = False

        Util.display_txt(frame, f"Cam: {cam.inner_fps:.2f}",
                         (frame.shape[1] - 500, frame.shape[0] - 10), (255, 255, 255), thickness=4)
        Util.show(frame)

        key = cv2.waitKey(10)
        if key == ord('q') or key == 27:
            break
        elif key == ord('c') and not capture:
            capture = True
            cam.photos_num = 0


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted.')
