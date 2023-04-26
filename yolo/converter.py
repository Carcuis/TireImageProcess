import os
import xml.etree.ElementTree as ET

from global_config import Global
from util import Util


class Converter:
    def __init__(self) -> None:
        self.dataset_dir = Global.dataset_dir
        self.classes_txt = os.path.join(self.dataset_dir, "classes.txt")
        if not os.path.exists(self.classes_txt):
            print(f"Error: {self.classes_txt} does not exist.")
            print("Please run get_classes.py first to generate a classes.txt.")
            exit(1)
        with open(self.classes_txt) as f:
            self.classes = tuple([line.strip() for line in f.readlines()])
        print(f"classes: {self.classes}")

        self.annotation_dir = os.path.join(self.dataset_dir, "annotations")
        self.output_dir = os.path.join(self.dataset_dir, "labels")
        Util.ensure_folder_exist(self.output_dir)
        print(f"Save converted labels to {self.output_dir}")
        self.xml_list = Util.get_file_list_in_dir_by_ext(self.annotation_dir, "xml")
        print(f"Total xml files: {len(self.xml_list)}")

    @staticmethod
    def _convert_coord(size: tuple[int, int], box: tuple[float, float, float, float]) \
            -> tuple[float, float, float, float]:
        """
        将 VOC 格式的坐标转换为 YOLO 格式

        :param size: 图片尺寸
        :param box: VOC 格式的坐标
        :return: YOLO 格式的坐标
        """
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def convert_annotation(self) -> None:
        """
        将 VOC 格式的标注文件转换为 YOLO 格式
        """
        for image_path in self.xml_list:
            basename_no_ext = Util.get_file_name_without_ext(image_path)

            with open(os.path.join(self.annotation_dir, basename_no_ext + ".xml"), encoding="utf-8") as in_file, \
                    open(os.path.join(self.output_dir, basename_no_ext + ".txt"), "w", encoding="utf-8") as out_file:
                tree = ET.parse(in_file)
                root = tree.getroot()
                size = root.find("size")
                w = int(size.find("width").text)
                h = int(size.find("height").text)

                for obj in root.iter("object"):
                    difficult = obj.find("difficult").text
                    class_name = obj.find("name").text
                    if class_name not in self.classes or int(difficult) == 1:
                        continue
                    class_id = self.classes.index(class_name)
                    xml_box = obj.find("bndbox")
                    voc_coords = (
                        float(xml_box.find("xmin").text),
                        float(xml_box.find("xmax").text),
                        float(xml_box.find("ymin").text),
                        float(xml_box.find("ymax").text)
                    )
                    yolo_coords = self._convert_coord((w, h), voc_coords)
                    out_file.write(str(class_id) + " " + " ".join([str(coord) for coord in yolo_coords]) + "\n")

        print("Done.")


def main():
    converter = Converter()
    converter.convert_annotation()


if __name__ == "__main__":
    main()
