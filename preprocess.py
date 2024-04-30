import os
import cv2
import argparse
import shutil
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Data Augmentation')
    parser.add_argument('--input_dir', type=str, default=None, help='Path to input images.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to output directory.')
    args = parser.parse_args()
    return args

def preprocess_image(input_img_path, output_dir, img_name):
    pic = cv2.imread(input_img_path)

    pic_processed = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic_processed = cv2.equalizeHist(pic_processed)
    pic_processed = cv2.cvtColor(pic_processed, cv2.COLOR_GRAY2BGR)

    pic_processed = cv2.GaussianBlur(pic_processed, (5, 5), 0)
    pic_processed = cv2.flip(pic_processed, 1)

    cv2.imwrite(output_dir  + img_name, pic)
    cv2.imwrite(output_dir + 'aug_' + img_name, pic_processed)

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    train_dir = os.path.join(output_dir, 'train/')
    valid_dir = os.path.join(output_dir, 'valid/')
    test_dir = os.path.join(output_dir, 'test/')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'images/'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels/'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'images/'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'labels/'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'images/'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'labels/'), exist_ok=True)

    all_images = []
    all_labels = []

    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if name.endswith('.jpg') or name.endswith('.png'):
                all_images.append(os.path.join(root, name))

    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if name.endswith('.txt'):
                all_labels.append(os.path.join(root, name))

    for img_path in all_images:
        img_name = os.path.basename(img_path)
        preprocess_image(img_path, output_dir, img_name)

    for label_path in all_labels:
        label_name = os.path.basename(label_path)
        shutil.copy(label_path, os.path.join(output_dir, label_name))
        shutil.copy(label_path, os.path.join(output_dir, 'aug_' + label_name))

    all_images = []
    for root, dirs, files in os.walk(output_dir):
        for name in files:
            if name.endswith('.jpg') or name.endswith('.png'):
                all_images.append(os.path.join(root, name))

    # split and move images and labels into train/valid/test dirs
    random.shuffle(all_images)
    train_size = int(0.7 * len(all_images))
    test_size = int(0.15 * len(all_images))
    valid_size = len(all_images) - train_size - test_size

    train_images = all_images[:train_size]
    test_images = all_images[train_size:train_size + test_size]
    valid_images = all_images[train_size + test_size:]

    for filename in train_images:
        shutil.move(filename, os.path.join(train_dir, 'images/'))
        shutil.move(os.path.splitext(filename)[0] + '.txt', os.path.join(train_dir, 'labels/'))

    for filename in test_images:
        shutil.move(filename, os.path.join(test_dir, 'images/'))
        shutil.move(os.path.splitext(filename)[0] + '.txt', os.path.join(test_dir, 'labels/'))

    for filename in valid_images:
        shutil.move(filename, os.path.join(valid_dir, 'images/'))
        shutil.move(os.path.splitext(filename)[0] + '.txt', os.path.join(valid_dir, 'labels/'))

if __name__ == '__main__':
    main()

                