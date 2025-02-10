#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import multiprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# **KHỞI TẠO MÔ HÌNH**

# In[2]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassification(nn.Module):  
    def training_step(self, batch):  
        images, labels = batch
        out = self(images) 
        loss = F.cross_entropy(out, labels)
        return loss

    def validating(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'Validation Loss': loss.detach(), 'Validation Accuracy': acc}

    def validating_epoch_final(self, outputs):
        batch_loss = [x['Validation Loss'] for x in outputs] 
        # each batch of the validation data
        epoch_loss = torch.stack(batch_loss).mean() 
        batch_accuracy = [x['Validation Accuracy'] for x in outputs]
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {'Validation Loss': epoch_loss.item(), 'Validation Accuracy': epoch_accuracy.item()}

    def epoch_final(self, epoch, result):
        print("Epoch [{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}"
              .format(epoch + 1, result['Training Loss'], result['Validation Loss'], result['Validation Accuracy']))


# **SỬ DỤNG KIẾN TRÚC ResNet50 CHO VIỆC PHÂN LOẠI**

# In[3]:


class ResNet(ImageClassification):
    def __init__(self):
        super().__init__()
        # Using ResNet50 pretrained model
        self.network = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        features = self.network.fc.in_features
        self.network.fc = nn.Linear(features, len(garbage_classes))

    def forward(self, image):
        return torch.sigmoid(self.network(image))

    def training_step(self, batch):
        images, labels = batch  
        out = self(images) 
        loss = F.cross_entropy(out, labels)
        return loss

    def epoch_final(self, epoch, result):
        print("Epoch [{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}"
              .format(epoch + 1, result['Train Loss'], result['Validation Loss'], result['Validation Accuracy']))


# **THIẾT LẬP HÀM XUẤT CÁC THÔNG SỐ KẾT QUẢ**

# In[4]:


def export_classification_metrics(model, dataloader, classes):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = move_to_gpu(images, device)
            labels = labels.to(device)  
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy()) 

    for i, class_name in enumerate(classes):
        class_labels = [1 if label == i else 0 for label in all_labels]
        class_preds = [1 if pred == i else 0 for pred in all_preds]
        
        accuracy = accuracy_score(class_labels, class_preds) * 100
        precision = precision_score(class_labels, class_preds, zero_division=0) * 100
        recall = recall_score(class_labels, class_preds, zero_division=0) * 100
        f1 = f1_score(class_labels, class_preds, zero_division=0) * 100
        
        print(f'{class_name}, accuracy: {accuracy:.2f}%, precision: {precision:.2f}%, recall: {recall:.2f}%, F1 score: {f1:.2f}%')


# **THIẾT LẬP EARLY STOPPPING**

# In[5]:


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, restore_best_weights=True):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.restore_best_weights = restore_best_weights
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_weights = model.state_dict().copy()
            if self.verbose:
                print(f'Validation loss improved: {val_loss:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss did not improve: {val_loss:.4f}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_weights is not None:
                    model.load_state_dict(self.best_model_weights)
                    if self.verbose:
                        print("Restored best model weights.")


# **THIẾT LẬP HÀM ĐO HIỆU SUẤT CHÍNH XÁC GIỮA CÁC LOẠI RÁC**

# In[6]:


def plot_accuracy_per_class(model, dataloader, classes):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = move_to_gpu(images, device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(labels.cpu().numpy())

    accuracies = []
    for i, class_name in enumerate(classes):
        class_labels = [1 if label == i else 0 for label in all_labels]
        class_preds = [1 if pred == i else 0 for pred in all_preds]

        accuracy = accuracy_score(class_labels, class_preds) * 100
        accuracies.append(accuracy)

    plt.bar(classes, accuracies)
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Class')
    plt.xticks(rotation=45)
    plt.show()


# **LOAD BỘ DỮ LIỆU TRASHNET**

# In[7]:


directory = 'kaggle/input/trashnet/dataset-resized'
global garbage_classes
garbage_classes = os.listdir(directory)
print(garbage_classes)


# **BIẾN ĐỔI BỘ DỮ LIỆU**

# In[8]:


# Import Transforms để chỉnh sửa ảnh
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# Resize ảnh về kích thước 224x224 pixels và transform thành Tensor
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Load bộ dữ liệu và áp dụng transformations
dataset = ImageFolder(directory, transform=transformations)


# **HIỂN THỊ ẢNH NGẪU NGHIÊN VỚI LABEL ĐỂ TEST BỘ DỮ LIỆU**

# In[9]:


def display_test(image, label):
    print("Label:", dataset.classes[label], "(Class No: " + str(label) + ")")
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

# Hiển thị ảnh ngẫu nhiên từ bộ dữ liệu
image, label = dataset[random.randint(0, len(dataset))]
display_test(image, label)


# **THIẾT LẬP SEED NGẪU NHIÊN PHỤC VỤ CHO VIỆC TÁI TẠO**

# In[10]:


random_seed = 43
torch.manual_seed(random_seed)


# **CHIA BỘ DỮ LIỆU & THIẾT LẬP BATCH SIZE**

# In[11]:


# Chia bộ dữ liệu thành các tập train, validate, và test
train_size = int(0.6 * len(dataset))  # 60% bộ dữ liệu
val_size = int(0.2 * len(dataset))  # 20% bộ dữ liệu
test_size = len(dataset) - train_size - val_size  # 20% bộ dữ liệu còn lại
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
from torch.utils.data.dataloader import DataLoader
batch_size = 32


# **THIẾT LẬP DATALOADERS**

# In[12]:


train = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
validation = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)


# **BATCH VISUALIZATION**

# In[13]:


from torchvision.utils import make_grid
def batch_visualization(data):
    for image, labels in data:
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(image, nrow=16).permute(1, 2, 0))
        plt.show()
        break

batch_visualization(train)
model = ResNet()

# Port model sang GPU
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()

# Chuyển model sang GPU
def move_to_gpu(data, device):
    if isinstance(data, (list, tuple)):
        return [move_to_gpu(x, device) for x in data]
    return data.to(device, non_blocking=True)

# DataLoading Class
class DataLoad():
    def __init__(self, data, device):
        self.data = data
        self.device = get_default_device()

    def __iter__(self):
        for batch in self.data:
            yield move_to_gpu(batch, self.device)

    def __len__(self):
        return len(self.data)

device = get_default_device()
print('Processing via:', device)  # Note: Nếu output trả về cuda, thì có thể thực hiện huấn luyện qua GPU

train = DataLoad(train, device)
validation = DataLoad(validation, device)
move_to_gpu(model, device)


# **HUẤN LUYỆN MÔ HÌNH**

# In[14]:


@torch.no_grad()
def evaluate(model, validator):
    model.eval()
    outputs = [model.validating(batch) for batch in validator]
    return model.validating_epoch_final(outputs)

def opt(epochs, learning_rate, model, train_loader, validator, opt_func=torch.optim.SGD, patience = 5):
    training_history = []  # List dùng để chứa lịch sử huấn luyện
    optimizer = opt_func(model.parameters(), learning_rate)
    early_stopping = EarlyStopping(patience=patience, verbose=True)  # Thiết lập early stopping

    for epoch in range(epochs):
        model.train()  
        train_loss = []  # List dùng để lưu training loss
        for batch in train_loader:
            loss = model.training_step(batch)  # Tính toán độ loss
            train_loss.append(loss)  # Thêm loss vào danh sách
            loss.backward()  # Thực hiện truyền ngược (Backpropagate)
            optimizer.step()  # Cập nhật các tham số bằng cách sử dụng optimizer
            optimizer.zero_grad()  # Đặt lại gradient về 0 cho batch tiếp theo

        # Giai đoạn Validation
        result = evaluate(model, validator)  # Đánh giá mô hình trên tập xác thực
        result['Train Loss'] = torch.stack(train_loss).mean().item()  # Tinh toán trung bình của training loss
        model.epoch_final(epoch, result)  # Gọi phương thức epoch_final trên mô hình
        training_history.append(result)  # Thêm kết quả vào lịch sử huấn luyện
        
        # Kiểm tra early stopping
        early_stopping(result['Validation Loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break  # Dừng việc huấn luyện nếu thỏa điều kiện kích hoạt early stopping

    return training_history

model = move_to_gpu(ResNet(), device)
evaluate(model, validation)


# **THIẾT LẬP HYPERPARAMETERS**

# In[ ]:


epoch = 1
optimizer = torch.optim.Adam
learning_rate = 0.00005
patience = 3  # Số epoch chờ để xem tình hình cải thiện của mô hình trước khi dừng huấn luyện
model_history = opt(epoch, learning_rate, model, train, validation, optimizer, patience)


# **VẼ BIỂU ĐỒ ACCURACY VS EPOCHS & LOSS VS EPOCHS**

# In[ ]:


def plot_accuracy(model_history):
    accuracies = [x['Validation Accuracy'] for x in model_history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.show()
plot_accuracy(model_history)

def plot_loss(model_history):
    train_loss = [x.get('Train Loss') for x in model_history]
    validation_loss = [x['Validation Loss'] for x in model_history]
    plt.plot(train_loss, '-bx')
    plt.plot(validation_loss, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs Epochs')
    plt.show()
plot_loss(model_history)


# **XUẤT CÁC THÔNG SỐ KẾT QUẢ VÀ HIỆU SUẤT CHÍNH XÁC GIỮA CÁC LOẠI RÁC**

# In[ ]:


export_classification_metrics(model, validation, garbage_classes)
plot_accuracy_per_class(model, validation, garbage_classes)


# **VẼ CONFUSION MATRIX**

# In[ ]:


def plot_confusion_matrix(model, dataloader, classes):
    model.eval() 
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = move_to_gpu(images, device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy()) 

    # Tạo confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalize to get percentages
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=classes)

    # Vẽ confusion matrix
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()
plot_confusion_matrix(model, validation, garbage_classes)


# **TRỰC QUAN HÓA CÁC PREDICTIONS**

# In[ ]:


def predict(image, model):
    xb = move_to_gpu(image.unsqueeze(0), device)
    # Nhận dự đoán từ mô hình
    yb = model(xb)
    # Chọn index với xác xuất cao nhất
    _, preds = torch.max(yb, dim=1)
    # Truy xuất class label
    return dataset.classes[preds[0].item()]


# **TESTING NO.1**

# In[ ]:


img, label = random.choice(dataset)
plt.imshow(img.permute(1, 2, 0))
print('Testing No.1 - Class:', dataset.classes[label], ', Predicted Class:', predict(img, model))
plt.title('Testing No.1')
plt.show()


# **TESTING NO.2**

# In[ ]:


img, label = random.choice(dataset)
plt.imshow(img.permute(1, 2, 0))
print('Testing No.2 - Class:', dataset.classes[label], ', Predicted Class:', predict(img, model))
plt.title('Testing No.2')
plt.show()


# **TESTING NO.3**

# In[ ]:


img, label = random.choice(dataset)
plt.imshow(img.permute(1, 2, 0)) 
print('Testing No.3 - Class:', dataset.classes[label], ', Predicted Class:', predict(img, model))
plt.title('Testing No.3')
plt.show()


# **LƯU MÔ HÌNH**

# In[ ]:


# FILE = "kaggle/output/ResnetModel.pth"
# model_scripted = torch.jit.script(model)
# torch.jit.save(model_scripted, FILE)
torch.save(model.state_dict(), 'kaggle/output/ResnetModel.pth')


# # Create and Save a Simple PyTorch Model

# In[ ]:


import torch
import torch.nn as nn
import torchvision.models as models
import hls4ml
import numpy as np

# Example model definition (using ResNet50 as in your code)
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        # Use a pretrained ResNet50 and replace the final fully-connected layer.
        self.network = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        features = self.network.fc.in_features
        self.network.fc = nn.Linear(features, num_classes)
        # Replace AdaptiveAvgPool2d with a fixed average pooling (kernel size 7) for fixed input sizes
        self.network.avgpool = nn.AvgPool2d(kernel_size=7)

    def forward(self, x):
        # You may use a sigmoid (or softmax) at the end if needed.
        return torch.sigmoid(self.network(x))

# Suppose you have a list of class names (garbage_classes) from your dataset:
# garbage_classes = os.listdir(your_data_directory)
num_classes = len(garbage_classes)

# Instantiate your model and load trained weights
model = ResNet(num_classes)
# Load your trained model weights (adjust the path as needed)
model.load_state_dict(torch.load('kaggle/output/ResnetModel.pth', map_location='cpu'))
model.eval()

# Define the input shape without the batch dimension (channels, height, width)
input_shape = (3, 224, 224)

# Create an HLS4ML configuration from your PyTorch model.
config = hls4ml.utils.config_from_pytorch_model(
    model,
    input_shape=input_shape,
    granularity='name',                # 'name' granularity gives you per-layer control
    default_precision='ap_fixed<16,6>',  # Use fixed-point for better FPGA performance
    backend='Vitis'                    # Choose the backend; e.g., Vitis for Xilinx devices
)

# Optionally, adjust configuration parameters (for example, the reuse factor)
config['Model']['ReuseFactor'] = 16

# Convert the PyTorch model to an HLS4ML model.
hls_model = hls4ml.converters.convert_from_pytorch_model(
    model,
    hls_config=config,
    input_shape=input_shape,
    output_dir='hls4ml_project'  # Directory where the HLS project will be generated
)



# In[ ]:


print(hls_model)
# Compile the HLS model (this runs a C-simulation to check correctness)
hls_model.compile()

# You can now run a prediction using a dummy input.
dummy_input = np.random.randn(*input_shape).astype(np.float32)
output = hls_model.predict(dummy_input)
print("HLS4ML model output:", output)


# In[ ]:


import shutil

vitis_hls_path = shutil.which('vitis_hls')
if vitis_hls_path:
    print("Vitis HLS found at:", vitis_hls_path)
else:
    print("Vitis HLS not found in PATH. Please source the Vitis environment.")

