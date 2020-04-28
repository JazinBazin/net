import os
from skimage.io import imread
import matplotlib.pyplot as plt

from albumentations import (Blur, Compose, HorizontalFlip, IAAEmboss, OneOf,
                            RandomBrightnessContrast, RandomCrop, RandomGamma,
                            RandomRotate90, ShiftScaleRotate, Transpose, VerticalFlip,
                            ElasticTransform, GridDistortion, OpticalDistortion)


def show_image(image_path):
    image_data = imread(image_path)
    plt.imshow(image_data)
    plt.show()


def aug_with_crop(width=640, height=480, crop_prob=1):
    return Compose([
        # RandomCrop(width=480, height=640, p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        IAAEmboss(p=0.25),
        Blur(p=0.01, blur_limit=3),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8)
    ], p=1)


# def aug_with_crop(image_size=256, crop_prob=1):
#     return Compose([
#         RandomCrop(width=image_size, height=image_size, p=crop_prob),
#         HorizontalFlip(p=0.5),
#         VerticalFlip(p=0.5),
#         RandomRotate90(p=0.5),
#         Transpose(p=0.5),
#         ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
#         RandomBrightnessContrast(p=0.5),
#         RandomGamma(p=0.25),
#         IAAEmboss(p=0.25),
#         Blur(p=0.01, blur_limit=3),
#         OneOf([
#             ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#             GridDistortion(p=0.5),
#             OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
#         ], p=0.8)
#     ], p=1)


def plot_training_history(history, model_name):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(history.epoch, history.history["iou_score"], label="Train iou")
    ax_acc.plot(history.epoch, history.history["val_iou_score"], label="Validation iou")
    ax_acc.legend()
    plt.savefig(
        os.path.join(
            'training_plots',
            model_name + '.png'
        )
    )


def test_images_shape(image_dir, width, height):
    for file in (os.path.join(image_dir, f) for f in os.listdir(image_dir)):
        image = imread(file)
        if image.shape[0] != width or image.shape[1] != height:
            print(file)
    print('Done')


def check_images(input_dir, output_dir):
    for img1_name, img2_name in zip(os.listdir(input_dir), os.listdir(output_dir)):
        if img1_name != img2_name:
            print(img1_name, img2_name)
            break
    print('Done')
