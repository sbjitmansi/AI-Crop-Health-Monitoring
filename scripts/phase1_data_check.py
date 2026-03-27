import os
import cv2
import matplotlib.pyplot as plt

DATASET_PATH = "D:/Crop_Health/Crop_Health_Project/data/raw"

classes = os.listdir(DATASET_PATH)

print("Classes found:", classes)

for cls in classes:
    class_path = os.path.join(DATASET_PATH, cls)
    images = os.listdir(class_path)
    
    print(f"\nClass: {cls}")
    print("Number of images:", len(images))
    
    img_path = os.path.join(class_path, images[0])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")
    plt.show()
