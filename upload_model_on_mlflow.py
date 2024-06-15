import os
import torch
from ultralytics import YOLO, settings
from roboflow import Roboflow
import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri("http://192.168.0.18:5000/")
mlflow.set_experiment("YOLOv8 co")

settings.update({
    'mlflow': True,
    'clearml': False,
    'comet': False,
    'dvc': False,
    'hub': False,
    'neptune': False,
    'raytune': False,
    'tensorboard': False,
    'wandb': False
})

def download_dataset():
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key="")
    project = rf.workspace("ntu-department-of-civil-engineering").project("building_crack_obj")
    version = project.version(1)
    dataset = version.download("yolov8")
    return dataset.location + "/data.yaml"

def register_model(model_path):
    # Load the YOLOv8 model
    print("Loading YOLOv8 model from file...")
    model = YOLO(model_path)

    # Set the device to Metal (Apple Silicon)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Log the model to MLflow
    with mlflow.start_run():
        mlflow.pytorch.log_model(model, "model")
        print(f"Model logged to MLflow with run_id: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    # Сначала регистрируем обученную модель
    trained_model_path = "./HackBuilding/train/weights/best.pt"
    register_model(trained_model_path)

    # Затем выполняем обучение новой модели, если это необходимо
    print("Downloading dataset...")
    # data_path = download_dataset()
    # register_model(data_path)