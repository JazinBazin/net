import os
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from segmentation_models import Unet
from keras.optimizers import Adam
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from data_generator import DataGeneratorFolder
from utils import aug_with_crop, plot_training_history
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.io import imread
from albumentations import Resize
from skimage.transform import resize
import datetime


def predict_window(source_image_path, model_path):
    image_data = imread(source_image_path).astype(np.float32) / 255
    image_data = image_data.reshape(1, image_data.shape[0], image_data.shape[1], 3).astype(np.float32)
    model = get_model(model_path)
    prediction = model.predict(image_data)
    prediction_image = prediction.reshape(image_data.shape[1], image_data.shape[2])
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    plt.axis('off')
    axs[0].imshow(prediction_image, cmap="Blues")
    axs[1].imshow(image_data.squeeze(0))
    plt.show()


def scale_image(image_data, max_image_size=1500):
    width_scale_factor = image_data.shape[0] / max_image_size
    height_scale_factor = image_data.shape[1] / max_image_size
    scale_factor = max(width_scale_factor, height_scale_factor)
    if scale_factor > 1:
        image_data = resize(image_data,
                            (image_data.shape[0] // scale_factor,
                             image_data.shape[1] // scale_factor),
                            anti_aliasing=True)
    augmented = Resize(height=(image_data.shape[0] // 32) * 32,
                       width=(image_data.shape[1] // 32) * 32)(image=image_data)
    image_data = augmented['image']
    image_data = image_data.reshape(1, image_data.shape[0], image_data.shape[1], 3).astype(np.float32)
    return image_data


def get_model(model_path):
    model = load_model(
        model_path,
        custom_objects={
            'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss,
            'iou_score': iou_score
        })
    return model


def predict_multiple_images(image_paths, model_path, save_result=False):
    model = get_model(model_path)
    fig, axs = plt.subplots(len(image_paths), 2, figsize=(20, 10))
    fig.canvas.set_window_title('Результаты обнаружения дорог')
    for i in range(len(image_paths)):
        image_data = imread(image_paths[i]).astype(np.float32) / 255
        image_data = scale_image(image_data)
        prediction = model.predict(image_data)
        prediction_image = prediction.reshape(image_data.shape[1], image_data.shape[2])
        axs[i, 0].axis('off')
        axs[i, 1].axis('off')
        axs[i, 0].imshow(image_data.squeeze(0))
        axs[i, 1].imshow(prediction_image, cmap="Blues")

    plt.subplots_adjust(left=0.2, bottom=0.01,
                        right=0.8, top=0.99,
                        wspace=0, hspace=0.03)
    if save_result:
        plt.savefig(
            os.path.join(
                'multiple_results',
                str(datetime.datetime.now().strftime('%d.%m.%Y_%H:%M:%S')) + '.png',
            )
        )
    plt.show()


def predict_single_image(source_image_path, model_path, save_result=False):
    image_data = imread(source_image_path).astype(np.float32) / 255
    image_data = scale_image(image_data)
    model = get_model(model_path)
    prediction = model.predict(image_data)
    prediction_image = prediction.reshape(image_data.shape[1], image_data.shape[2])
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    plt.axis('off')
    axs[0].imshow(prediction_image, cmap="Blues")
    axs[1].imshow(image_data.squeeze(0))
    if save_result:
        plt.savefig(
            os.path.join(
                'single_results',
                os.path.basename(source_image_path)
            )
        )
    plt.show()


def train_model():
    train_generator = DataGeneratorFolder(root_dir='./dataset/training',
                                          image_folder='input/',
                                          mask_folder='output/',
                                          augmentation=aug_with_crop,
                                          batch_size=4,
                                          # image_size=512,
                                          nb_y_features=1)

    test_generator = DataGeneratorFolder(root_dir='./dataset/testing',
                                         image_folder='input/',
                                         mask_folder='output/',
                                         batch_size=1,
                                         nb_y_features=1)

    lr_reducer = ReduceLROnPlateau(factor=0.1,
                                   cooldown=10,
                                   patience=10,
                                   verbose=1,
                                   min_lr=0.1e-5)

    mode_auto_save = ModelCheckpoint("checkpoints/saved_model_epoch_{epoch:02d}_iou_{val_iou_score:.2f}.h5",
                                     monitor='val_iou_score',
                                     verbose=1,
                                     save_best_only=False,
                                     save_weights_only=False,
                                     mode='auto',
                                     period=1)

    early_stopping = EarlyStopping(patience=10,
                                   verbose=1,
                                   mode='auto')

    # tensor_board = TensorBoard(log_dir='./logs/tensor_board',
    #                            histogram_freq=0,
    #                            write_graph=False,
    #                            write_images=False)

    callbacks = [mode_auto_save, lr_reducer, early_stopping]
    # callbacks = [mode_auto_save, lr_reducer, tensor_board, early_stopping]

    model = Unet(backbone_name='efficientnetb0',
                 encoder_weights='imagenet',
                 encoder_freeze=False)

    model.compile(optimizer=Adam(),
                  loss=bce_jaccard_loss,
                  metrics=[iou_score])

    epochs = 100

    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        # workers=4,
        # use_multiprocessing=True,
        validation_data=test_generator,
        shuffle=True,
        verbose=1,
        callbacks=callbacks)

    model_name = 'trained_model_on_' + str(epochs) + '_epochs_' + \
                 str(datetime.datetime.now().strftime('%d.%m.%Y_%H:%M:%S'))

    model.save(os.path.join(
        'trained_models',
        model_name + '.h5')
    )

    plot_training_history(history, model_name)

    """
        Reduce learning rate when a metric has stopped improving.
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced. new_lr = lr * factor
        patience: number of epochs that produced the monitored quantity with no improvement after which training will be stopped. Validation quantities may not be produced for every epoch, if the validation frequency (model.fit(validation_freq=5)) is greater than one.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on significant changes.
        cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """ \
 \
    """
        Save the model after every epoch.
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
        mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is made based on either the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """ \
 \
    """
        Stop training when a monitored quantity has stopped improving.
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
        patience: number of epochs that produced the monitored quantity with no improvement after which training will be stopped. Validation quantities may not be produced for every epoch, if the validation frequency (model.fit(validation_freq=5)) is greater than one.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.
        restore_best_weights: whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.
    """ \
 \
    """
        TensorBoard basic visualizations.
        tensorboard --logdir=/full_path_to_your_logs
        log_dir: the path of the directory where to save the log files to be parsed by TensorBoard.
        histogram_freq: frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations.
        batch_size: size of batch of inputs to feed to the network for histograms computation.
        write_graph: whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True.
        write_grads: whether to visualize gradient histograms in TensorBoard. histogram_freq must be greater than 0.
        write_images: whether to write model weights to visualize as image in TensorBoard.
        embeddings_freq: frequency (in epochs) at which selected embedding layers will be saved. If set to 0, embeddings won't be computed. Data to be visualized in TensorBoard's Embedding tab must be passed as embeddings_data.
        embeddings_layer_names: a list of names of layers to keep eye on. If None or empty list all the embedding layer will be watched.
        embeddings_metadata: a dictionary which maps layer name to a file name in which metadata for this embedding layer is saved. See the details about metadata files format. In case if the same metadata file is used for all embedding layers, string can be passed.
        embeddings_data: data to be embedded at layers specified in embeddings_layer_names. Numpy array (if the model has a single input) or list of Numpy arrays (if the model has multiple inputs). Learn more about embeddings.
        update_freq: 'batch' or 'epoch' or integer. When using 'batch', writes the losses and metrics to TensorBoard after each batch. The same applies for 'epoch'. If using an integer, let's say 10000, the callback will write the metrics and losses to TensorBoard every 10000 samples. Note that writing too frequently to TensorBoard can slow down your training.
    """
