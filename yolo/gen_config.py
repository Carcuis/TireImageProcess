import os
import time

import yaml

from global_config import Global


class YamlGenerator:
    def __init__(self) -> None:
        self.dataset_dir = Global.dataset_dir
        if not os.path.exists(self.dataset_dir):
            print(f"Error: {self.dataset_dir} not found, please check current directory.")
            exit(1)
        self.train = os.path.join("train", "images")
        self.val = os.path.join("val", "images")
        self.test = os.path.join("test", "images")
        self.classes_txt = os.path.join(self.dataset_dir, "classes.txt")
        self.config_yaml = os.path.join(os.getcwd(), "yolo", "config.yaml")

    def get_classes(self) -> dict[int, str]:
        if not os.path.exists(self.classes_txt):
            print(f"Error: {self.classes_txt} not found, use get_classes.py to generate first.")
            exit(1)
        print(f"Found {self.classes_txt}")
        classes = {}
        with open(self.classes_txt) as f:
            for i, line in enumerate(f):
                classes[i] = line.strip()
        return classes

    def generate_yaml(self) -> None:
        data = {
            "path": self.dataset_dir,
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "names": self.get_classes()
        }
        with open(self.config_yaml, "w") as f:
            f.write(f"# Generated by gen_config.py on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            yaml.dump(data, f, sort_keys=False)

        print(f"Generated {self.config_yaml} done.")


def main():
    yaml_generator = YamlGenerator()
    yaml_generator.generate_yaml()


if __name__ == "__main__":
    main()
