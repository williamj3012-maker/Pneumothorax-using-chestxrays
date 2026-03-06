import cv2
import matplotlib.pyplot as plt
import numpy as np

# Function to apply CLAHE
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Function for histogram equalization
def apply_histogram_equalization(image):
    return cv2.equalizeHist(image)

# Function for bilateral filtering
def apply_bilateral_filtering(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

# Function for gamma correction
def apply_gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Load the chest X-ray image
image_path = 'path_to_your_xray_image.jpg'  # Update this path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply preprocessing techniques
clahe_image = apply_clahe(image)
he_image = apply_histogram_equalization(image)
bilateral_image = apply_bilateral_filtering(image)
gamma_corrected_image = apply_gamma_correction(image, gamma=2.0)  # Example gamma value

# Display the images
plt.figure(figsize=(15, 10))
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('CLAHE')
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Histogram Equalization')
plt.imshow(he_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Bilateral Filtering')
plt.imshow(bilateral_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Gamma Correction')
plt.imshow(gamma_corrected_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()