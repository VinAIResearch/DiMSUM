import os
import numpy as np
from PIL import Image
import torch
from torchvision.models import inception_v3
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn

# Path to your sketch dataset
name_dataset = 'sketch_64'
dataset_path = f'./data/{name_dataset}'

# Size of your images
image_size = (64, 64)

# Function to preprocess an image
def preprocess_image(image_path):
    image = Image.open(image_path)
    # image = image.resize(image_size, Image.ANTIALIAS)
    image = np.array(image)
    return image

# Function to load the sketch dataset
def load_sketch_dataset(dataset_path):
    sketch_data = []
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                if image_file.endswith('.jpg'):
                    sketch = preprocess_image(image_path)
                    sketch_data.append(sketch)
    return np.array(sketch_data)

# Load the sketch dataset
sketch_images = load_sketch_dataset(dataset_path)

# Load the pre-trained Inception-v3 model
model = inception_v3(pretrained=True, transform_input=False).to('cuda')
model.fc = nn.Identity()
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()
# model = torch.nn.Sequential(*list(model.children())[:-2])

# Define data preprocessing for the model
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),  # Inception-v3 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize lists to store feature activations
feature_activations = []

print('Now, process each sketch image and extract features')
# Process each sketch image and extract features
with torch.no_grad():
    # i = 1
    for sketch in tqdm(sketch_images):
        # print(i)
        # i += 1
        sketch_tensor = torch.tensor(sketch).permute(2, 0, 1).float()

        sketch_tensor = preprocess(sketch_tensor)
        sketch_tensor = sketch_tensor.to('cuda')  # Move to GPU if available
        
        # Extract features (activations from a specific layer)
        # sketch_tensor = torch.unsqueeze(sketch_tensor, 0)
        # print(sketch_tensor.shape)
        activations = model(sketch_tensor.unsqueeze(0))  # Add batch dimension again for the model
        # print(activations.shape)
        feature_activations.append(activations.cpu().numpy())

# Combine feature activations into a single array
feature_activations = np.vstack(feature_activations)

# Calculate 'mu' (mean) and 'sigma' (covariance)
mu = np.mean(feature_activations, axis=0)
sigma = np.cov(feature_activations, rowvar=False)

# Create a dictionary to store 'mu' and 'sigma'
sketch_stats = {'mu': mu, 'sigma': sigma}

# Save the statistics to a file
np.save(f'./pytorch_fid/{name_dataset}_stat.npy', sketch_stats)