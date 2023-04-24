import os

from util import Util


class LabelImgRunner:
    def __init__(self) -> None:
        self.current_dir = os.getcwd()
        self.dataset_dir = os.path.join(self.current_dir, "dataset", "images")
        if not os.path.exists(self.dataset_dir):
            print(f"Error: {self.dataset_dir} does not exist.")
            exit(1)
        self.annotations_dir = os.path.join(self.current_dir, "dataset", "annotations")
        Util.ensure_folder_exist(self.annotations_dir)
        self.classes_txt = os.path.join(self.current_dir, "dataset", "classes.txt")
        Util.ensure_file_exist(self.classes_txt)
        self.check_labelimg_installation()

    def run(self) -> None:
        command = f"labelImg {self.dataset_dir} {self.classes_txt} {self.annotations_dir}"
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
    runner.sort_classes_txt()


if __name__ == "__main__":
    main()
