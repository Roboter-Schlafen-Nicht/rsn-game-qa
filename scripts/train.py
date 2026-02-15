from ultralytics import YOLO
import torch

if __name__ == "__main__":
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(device)

    model = YOLO("yolov8n.pt").to(device)

    results = model.train(
        data="F:/work/LastWarRobot/World Model V1.v1-roboflow-instant-1--eval-.yolov8/data.yaml",
        epochs=100,
        imgsz=640,
        device=device,
        amp=False,
    )

    print(results)
