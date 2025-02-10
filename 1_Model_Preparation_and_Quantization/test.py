import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import random
import os

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset directory and classes
data_dir = 'kaggle/input/trashnet/dataset-resized'
garbage_classes = os.listdir(data_dir)
print("Classes found:", garbage_classes)

# Use deterministic transforms for testing (no random augmentation)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create dataset
dataset = ImageFolder(root=data_dir, transform=test_transforms)

# Load the TorchScript model
model_path = 'kaggle/output/ResnetModel.pth'
model = torch.jit.load(model_path, map_location=device)
model.to(device)
model.eval()

# Pick a random image from the dataset
idx = random.randint(0, len(dataset) - 1)
image, true_label = dataset[idx]

# Display the image with its ground truth label
plt.figure(figsize=(4, 4))
plt.imshow(image.permute(1, 2, 0))
plt.title("Ground Truth: " + dataset.classes[true_label])
plt.axis("off")
plt.show()

# Prepare the image and perform inference
image_batch = image.unsqueeze(0).to(device)
with torch.no_grad():
    outputs = model(image_batch)
    _, predicted_idx = torch.max(outputs, 1)
    predicted_class = dataset.classes[predicted_idx.item()]

print("Predicted Class:", predicted_class)
print("Ground Truth Class:", dataset.classes[true_label])
