from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score

# Move data to GPU if available
def move_to_gpu(data, device):
    if isinstance(data, (list, tuple)):
        return [move_to_gpu(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Get default device (GPU or CPU)
def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device = get_default_device()

# Evaluate model accuracy
def evaluate_model(model, dataloader):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = move_to_gpu(images, device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds) * 100
    return accuracy

# Measure inference time
def measure_inference_time(model, dataloader, num_batches=10):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            images = move_to_gpu(images, device)
            
            start_time = time.time()
            model(images)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return np.mean(times)

# Main execution block
if __name__ == '__main__':
    batch_size = 64
    directory = 'kaggle/input/trashnet/dataset-resized'

    # Transformations for the dataset
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(directory, transform=transformations)

    # Splitting the dataset
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    # DataLoader with num_workers for multiprocessing
    validation = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)

    # Load models
    original_model = torch.jit.load("ResnetModel.pth")
    quantized_model = torch.jit.load("ResnetModel_Quantized.pth")

    original_model.eval()
    quantized_model.eval()

    # Evaluate models
    original_accuracy = evaluate_model(original_model, validation)
    quantized_accuracy = evaluate_model(quantized_model, validation)

    # Model size comparison
    original_size = os.path.getsize("ResnetModel.pth") / (1024 * 1024)
    quantized_size = os.path.getsize("ResnetModel_Quantized.pth") / (1024 * 1024)

    # Inference time comparison
    original_inference_time = measure_inference_time(original_model, validation)
    quantized_inference_time = measure_inference_time(quantized_model, validation)

    # Display results
    print("===== Quantization Comparison =====")
    print(f"Original Model Accuracy: {original_accuracy:.2f}%")
    print(f"Quantized Model Accuracy: {quantized_accuracy:.2f}%")
    print(f"Original Model Size: {original_size:.2f} MB")
    print(f"Quantized Model Size: {quantized_size:.2f} MB")
    print(f"Original Model Inference Time: {original_inference_time:.4f} sec/batch")
    print(f"Quantized Model Inference Time: {quantized_inference_time:.4f} sec/batch")
