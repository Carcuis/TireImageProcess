import os

from ultralytics import YOLO


def main():
    model = YOLO("yolov8s.pt")

    config_yaml = os.path.join(os.getcwd(), "yolo", "config.yaml")
    if not os.path.exists(config_yaml):
        print(f"Error: {config_yaml} not found, run gen_config.py to generate first.")
        exit(1)
    model.train(data=config_yaml, epochs=50, batch=2, device=0)

    model.val()


if __name__ == '__main__':
    main()
