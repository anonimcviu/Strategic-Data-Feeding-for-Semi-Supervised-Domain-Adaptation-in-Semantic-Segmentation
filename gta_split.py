import os
import shutil
import random
random.seed(170997)
# Paths to your dataset
dataset_path = './datasets/gta/gta3/'
train_path = './datasets/gta/gtaV/train/'
val_path = './datasets/gta/gtaV/test/'

# Create directories for train and validation datasets
os.makedirs(os.path.join(train_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_path, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)

# List all images
image_folder = os.path.join(dataset_path, 'images')
label_folder = os.path.join(dataset_path, 'labels')

image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]  # Adjust extensions if needed

# Shuffle the list of image files
random.shuffle(image_files)

# Split the data
split_ratio = 0.80
split_index = int(len(image_files) * split_ratio)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Function to copy files
def copy_files(file_list, src_folder, dest_folder):
    for file_name in file_list:
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        shutil.copy(src_file, dest_file)

# Copy training files
copy_files(train_files, image_folder, os.path.join(train_path, 'images'))
copy_files(train_files, label_folder, os.path.join(train_path, 'labels'))

# Copy validation files
copy_files(val_files, image_folder, os.path.join(val_path, 'images'))
copy_files(val_files, label_folder, os.path.join(val_path, 'labels'))

print(f"Dataset split into {len(train_files)} training and {len(val_files)} validation files.")
