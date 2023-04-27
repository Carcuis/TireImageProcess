import os

from get_classes import VOCAnnotationCounter
from global_config import Global
from util import Util


class LabelImgRunner:
    def __init__(self) -> None:
        self.dataset_dir = Global.dataset_dir
        self.image_dir = os.path.join(self.dataset_dir, "images")
        if not os.path.exists(self.image_dir):
            print(f"Error: {self.image_dir} does not exist.")
            exit(1)
        self.annotations_dir = os.path.join(self.dataset_dir, "annotations")
        Util.ensure_folder_exist(self.annotations_dir)
        self.classes_txt = os.path.join(self.dataset_dir, "classes.txt")
        Util.ensure_file_exist(self.classes_txt)
        self.check_labelimg_installation()

    def run(self) -> None:
        command = f"labelImg {self.image_dir} {self.classes_txt} {self.annotations_dir}"
        print(f"Running command: {command}")
        os.system(command)
        print("Done.")

    def sort_classes_txt(self) -> None:
        with open(self.classes_txt, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = sorted(lines)
        with open(self.classes_txt, "w") as f:
            for line in lines:
                f.write(f"{line}\n")
        print(f"Sort {self.classes_txt} success.")

    @staticmethod
    def check_labelimg_installation() -> None:
        try:
            import labelImg
        except ImportError:
            print("Error: labelImg is not installed.")
            print("run: pip install labelImg")
            exit(1)


def main():
    runner = LabelImgRunner()
    runner.run()
    counter = VOCAnnotationCounter()
    counter.save_classes()


if __name__ == "__main__":
    main()
