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


def train_yolo(data_path, epochs=50, img_size=640):

    # Load the YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    # Set the device to Metal (Apple Silicon)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    model.to(device)

    # Train the model
    print("Starting training...")
    try:
        results = model.train(data=data_path, epochs=epochs, imgsz=img_size, device=device)
        print("Training completed.")

        # Логирование метрик
        if 'metrics' in results:
            metrics = results['metrics']
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

        # Save the model
        model_path = "yolov8_model"
        model.save(model_path)
        mlflow.pytorch.log_model(model, model_path)

    except mlflow.exceptions.MlflowException as e:
        print(f"MLflow Exception: {e}")


if __name__ == "__main__":
    print("Downloading dataset...")
    data_path = download_dataset()
    train_yolo(data_path)