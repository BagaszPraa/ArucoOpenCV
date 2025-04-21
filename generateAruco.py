import cv2
import numpy as np
import matplotlib.pyplot as plt
import os  # Untuk bekerja dengan direktori

# Pastikan folder markerAruco ada
output_folder = 'markerAruco'
os.makedirs(output_folder, exist_ok=True)

# Define the dictionary we want to use
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate a marker
marker_id = 0
marker_size = 200  # Size in pixels
marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# Simpan gambar di folder markerAruco
output_path = os.path.join(output_folder, f'aruco_{marker_id}.png')
cv2.imwrite(output_path, marker_image)

plt.imshow(marker_image, cmap='gray', interpolation='nearest')
plt.axis('off')  # Hide axes
plt.title(f'ArUco Marker {marker_id}')
plt.show()
