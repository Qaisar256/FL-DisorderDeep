import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Disease distribution
diseases = ['Healthy', 'COPD', 'Asthma', 'Pneumonia', 'URTI', 'Bronchiectasis', 'Bronchiolitis', 'LRTI']
base_path = 'c://sampledata//images'
augmented_path = 'c://sampledata//augmented'

# Ensure the augmented directories exist
os.makedirs(augmented_path, exist_ok=True)
for disease in diseases:
    os.makedirs(os.path.join(augmented_path, disease), exist_ok=True)

# Initialize the ImageDataGenerator with some common augmentation techniques
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to augment images for each disease
def augment_images(disease):
    num_augmented_images = 50  # number of augmented images to generate per original image
    image_folder = os.path.join(base_path, disease)
    augmented_folder = os.path.join(augmented_path, disease)
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    for img_path in images:
        img = load_img(img_path)  # load an image
        img_array = img_to_array(img)  # convert to numpy array
        img_array = img_array.reshape((1,) + img_array.shape)  # reshape image

        i = 0
        for batch in data_gen.flow(img_array, batch_size=1, save_to_dir=augmented_folder, save_prefix='aug', save_format='jpg'):
            i += 1
            if i >= num_augmented_images:
                break

# Apply data augmentation
for disease in diseases:
    augment_images(disease)

# Note: The following code for visualizing distributions should be updated based on the actual counts of the augmented images,
# which will require reading the contents of the augmented directories.

# Visualize original distribution
plt.figure(figsize=(10, 6))
plt.bar(diseases, [total_counts[disease] for disease in diseases], color='b', alpha=0.5)
plt.title('Original Distribution of Disease Classes')
plt.xlabel('Disease Classes')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize augmented distribution (not implemented; placeholder)
plt.figure(figsize=(10, 6))
plt.bar(diseases, [total_counts[disease] + 50 for disease in diseases], color='r', alpha=0.5)  # Example: adding 50 samples for visualization
plt.title('Augmented Distribution of Disease Classes')
plt.xlabel('Disease Classes')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
