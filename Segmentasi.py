import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

def thresholding(image, threshold):
    return np.where(image > threshold, 255, 0)

input_image_path = 'F:\\Semester 5\\Pengolahan Citra Digital\\Project_segmentasi\\Gambar\\Hitler.jpg'  # Ganti dengan path citra Anda
image = imageio.imread(input_image_path)

if len(image.shape) == 3:
    image = np.mean(image, axis=2) 

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Citra Asli')
plt.imshow(image, cmap='gray')
plt.axis('off')

sobel_x = sobel(image, axis=1)  
sobel_y = sobel(image, axis=0)  

sobel_magnitude = np.hypot(sobel_x, sobel_y)

plt.subplot(1, 2, 2)
plt.title('Deteksi Tepi Sobel')
plt.imshow(sobel_magnitude, cmap='gray')
plt.axis('off')
plt.show()

threshold_value = 50  
segmented_image = thresholding(sobel_magnitude, threshold_value)

plt.figure(figsize=(6, 6))
plt.title('Hasil Segmentasi (Thresholding)')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')
plt.show()
