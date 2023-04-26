import os
import random
import shutil

from global_config import Global
from util import Util


class SplitTrainVal:
    def __init__(self) -> None:
        self.dataset_dir = Global.dataset_dir
        self.images_folder = os.path.join(self.dataset_dir, "images")
        self.labels_folder = os.path.join(self.dataset_dir, "labels")
        if not os.path.exists(self.images_folder) or not os.path.exists(self.labels_folder):
            print("Error: 原始 images 或 labels 文件夹不存在")
            exit(1)
        self.train_folder = os.path.join(self.dataset_dir, "train")
        self.val_folder = os.path.join(self.dataset_dir, "val")
        self.test_folder = os.path.join(self.dataset_dir, "test")

        self.check_images_and_labels_pairs()

    def check_images_and_labels_pairs(self) -> None:
        """
        检查 images 和 labels 文件夹中的文件名是否一一对应
        """
        images_list = sorted(os.listdir(self.images_folder))
        labels_list = sorted(os.listdir(self.labels_folder))

        if len(images_list) == 0 or len(labels_list) == 0:
            print("Error: images 文件夹或 labels 文件夹中没有文件")
            exit(1)

        if len(images_list) != len(labels_list):
            print("Error: images 文件夹和 labels 文件夹中的文件数量不一致")
            images_without_ext = [Util.get_file_name_without_ext(image) for image in images_list]
            labels_without_ext = [Util.get_file_name_without_ext(label) for label in labels_list]
            for image in images_without_ext:
                if image not in labels_without_ext:
                    print(f"Error: images 文件夹中的 {image}.jpg 文件没有对应的 labels 文件")
            for label in labels_without_ext:
                if label not in images_without_ext:
                    print(f"Error: labels 文件夹中的 {label}.txt 文件没有对应的 images 文件")
            exit(1)
        else:
            print(f"images 文件夹和 labels 文件夹中的文件数量一致，共有 {len(images_list)} 个文件")
            for image, label in zip(images_list, labels_list):
                if Util.get_file_name_without_ext(image) != Util.get_file_name_without_ext(label):
                    print("Error: images 文件夹和 labels 文件夹中的文件名不一一对应")
                    exit(1)

        print("Check success: images 文件夹和 labels 文件夹中的文件名一一对应")

    def split_train_val(self) -> None:
        """
        进行拆分
        """
        images_list = sorted(os.listdir(self.images_folder))
        labels_list = sorted(os.listdir(self.labels_folder))
        num_data = len(images_list)
        num_train = int(num_data * 0.8)  # 训练集占60%
        num_val = int(num_data * 0.2)  # 验证集占20%
        # num_test = num_data - num_train - num_val  # 测试集占20%
        indices = list(range(num_data))
        random.shuffle(indices)  # 打乱索引顺序

        # 拷贝训练集数据到train文件夹中
        print(f"Copying {num_train} images and labels to {self.train_folder}")
        train_images_folder = os.path.join(self.train_folder, "images")
        train_labels_folder = os.path.join(self.train_folder, "labels")
        Util.ensure_folder_exist(train_images_folder, train_labels_folder)
        for i in indices[:num_train]:
            image_path = os.path.join(self.images_folder, images_list[i])
            label_path = os.path.join(self.labels_folder, labels_list[i])
            shutil.copy2(image_path, train_images_folder)
            shutil.copy2(label_path, train_labels_folder)

        # 拷贝验证集数据到val文件夹中
        print(f"Copying {num_val} images and labels to {self.val_folder}")
        val_images_folder = os.path.join(self.val_folder, "images")
        val_labels_folder = os.path.join(self.val_folder, "labels")
        Util.ensure_folder_exist(val_images_folder, val_labels_folder)
        for i in indices[num_train:num_train + num_val]:
            image_path = os.path.join(self.images_folder, images_list[i])
            label_path = os.path.join(self.labels_folder, labels_list[i])
            shutil.copy2(image_path, val_images_folder)
            shutil.copy2(label_path, val_labels_folder)

        # 拷贝测试集数据到test文件夹中
        # print(f"Copying {num_val} images and labels to {self.test_folder}")
        # test_images_folder = os.path.join(self.test_folder, "images")
        # test_labels_folder = os.path.join(self.test_folder, "labels")
        # Util.ensure_folder_exist(test_images_folder, test_labels_folder)
        # for i in indices[num_train + num_val:]:
        #     image_path = os.path.join(self.images_folder, images_list[i])
        #     label_path = os.path.join(self.labels_folder, labels_list[i])
        #     shutil.copy2(image_path, test_images_folder)
        #     shutil.copy2(label_path, test_labels_folder)

        print("Split done.")


def main():
    splitter = SplitTrainVal()
    splitter.split_train_val()


if __name__ == "__main__":
    main()
