import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from model import YourModel  # replace with your model import

# Load the trained model
model = YourModel()  # initialize your model
model.load_state_dict(torch.load('path/to/your/model.pth'))  # load the model weights
model.eval()

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # resize to desired size
    transforms.ToTensor(),
])

# Function to perform segmentation
def segment_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze().numpy()  # return the segmentation output

# Visualization function
def visualize(image_path, segmentation):
    image = Image.open(image_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation, cmap='gray')
    plt.title('Segmentation Output')
    plt.axis('off')
    
    plt.show()

# Example usage
if __name__ == '__main__':
    image_path = 'path/to/your/xray/image.jpg'  # replace with your image path
    segmentation = segment_image(image_path)
    visualize(image_path, segmentation)