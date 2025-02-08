import torch
from torch.quantization import prepare, convert
from torch.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

# Load the trained model
model = torch.jit.load("ResnetModel.pth")
model.eval()
model.cpu()  # Quantization works on CPU

# Define a custom QConfig with quant_min and quant_max
qconfig = torch.quantization.QConfig(
    activation=MinMaxObserver.with_args(quant_min=0, quant_max=255),
    weight=PerChannelMinMaxObserver.with_args(quant_min=-128, quant_max=127)
)

# Apply the custom QConfig
model.qconfig = qconfig
prepare(model, inplace=True)

# Calibration step
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

directory = 'kaggle/input/trashnet/dataset-resized'
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = ImageFolder(directory, transform=transformations)
validation = DataLoader(dataset, batch_size=64, shuffle=True)

# Run calibration with a few batches
with torch.no_grad():
    for images, _ in validation:
        model(images)
        break  # Calibration doesn't need the full dataset

# Convert to quantized model
quantized_model = convert(model, inplace=True)

# Save the quantized model
quantized_model_file = "ResnetModel_Quantized.pth"
torch.jit.save(torch.jit.script(quantized_model), quantized_model_file)

print("Static quantization complete. Model saved as ResnetModel_Quantized.pth")
