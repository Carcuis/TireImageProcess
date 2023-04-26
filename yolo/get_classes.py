import os
import xml.etree.ElementTree as ET

from global_config import Global
from util import Util


class VOCAnnotationCounter:
    def __init__(self) -> None:
        self.dataset_dir = Global.dataset_dir
        self.annotation_dir = os.path.join(self.dataset_dir, "annotations")
        if not os.path.exists(self.annotation_dir):
            raise FileNotFoundError(f"{self.annotation_dir} does not exist.")
        self.class_list = []
        self.class_instances = {}  # {class_name: instance_count}
        self.classes_txt = os.path.join(self.dataset_dir, "classes.txt")

        self.count_classes()
        self.print_instances()

    def count_classes(self) -> None:
        xml_files = Util.get_file_list_in_dir_by_ext(self.annotation_dir, "xml")
        for xml_file in xml_files:
            root = ET.parse(xml_file).getroot()
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in self.class_list:
                    self.class_list.append(class_name)
                    self.class_instances[class_name] = 0
                self.class_instances[class_name] += 1

    def print_classes(self) -> None:
        print(self.class_list)

    def print_instances(self) -> None:
        print(self.class_instances)

    def save_classes(self) -> None:
        with open(self.classes_txt, "w") as f:
            self.class_list.sort()
            for class_name in self.class_list:
                f.write(class_name + "\n")
        print(f"Saved classes to {self.classes_txt}")


def main():
    counter = VOCAnnotationCounter()
    counter.save_classes()


if __name__ == "__main__":
    main()
