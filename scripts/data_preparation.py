import tensorflow as tf
import numpy as np
import os
from data_paths import paths, config  

def download_and_extract_data():
    # Check if the directories for the images already exist
    if paths['anchor_images_path'].exists() and paths['positive_images_path'].exists():
        print("Data already exists. Skipping download.")
        return

    # If data does not exist, proceed with downloading and extracting
    print("Data not found. Downloading now...")
    import subprocess
    subprocess.run(["gdown", "--id", "1jvkbTr_giSP3Ru8OwGNCg6B4PvVbcO34"])
    subprocess.run(["gdown", "--id", "1EzBZUb_mh_Dp_FKD0P4XiYYSd0QBH5zW"])
    subprocess.run(["unzip", "-oq", "left.zip", "-d", str(paths['cache_dir'])])
    subprocess.run(["unzip", "-oq", "right.zip", "-d", str(paths['cache_dir'])])

def preprocess_image(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (config["TARGET_HEIGHT"], config["TARGET_WIDTH"]))  # Use dimensions from config
    return image

def preprocess_triplets(anchor, positive, negative):
    return preprocess_image(anchor), preprocess_image(positive), preprocess_image(negative)

def create_datasets():
    anchor_images = sorted([str(paths['anchor_images_path'] / f) for f in os.listdir(paths['anchor_images_path'])])
    positive_images = sorted([str(paths['positive_images_path'] / f) for f in os.listdir(paths['positive_images_path'])])
    
    rng = np.random.RandomState(seed=42)
    rng.shuffle(anchor_images)
    rng.shuffle(positive_images)
    
    negative_images = anchor_images + positive_images
    np.random.RandomState(seed=32).shuffle(negative_images)

    dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(anchor_images),
                                    tf.data.Dataset.from_tensor_slices(positive_images),
                                    tf.data.Dataset.from_tensor_slices(negative_images)))
    dataset = dataset.shuffle(buffer_size=1024).map(preprocess_triplets)

    image_count = len(anchor_images)
    train_dataset = dataset.take(round(image_count * 0.8)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = dataset.skip(round(image_count * 0.8)).batch(32).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

download_and_extract_data()