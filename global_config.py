import os


class Global:
    dataset_name = "defect"
    dataset_dir = os.path.join(os.getcwd(), "dataset", dataset_name)
    if not os.path.exists(dataset_dir):
        print(f"Warning: dataset directory from config.py ` {dataset_dir} ` not found.")
