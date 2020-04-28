import os
from network import train_model, predict_single_image, predict_multiple_images, predict_window
from utils import show_image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    # show_image('multiple_results/09.12.2019_12:42:49.png')
    # predict_multiple_images(
    #     ('/home/admin_03/PycharmProjects/AerialPhotographs/testing/input/img-1.png',
    #      '/home/admin_03/PycharmProjects/AerialPhotographs/testing/input/img-3.png'),
    #     'trained_models/roads_prediction_model_06.12.2019_01:50:39.h5',
    #     save_result=True)
    predict_window("/home/admin_03/PycharmProjects/Roads/dataset/testing/input/model2_bricks12.jpg",
                   "/home/admin_03/PycharmProjects/Roads/checkpoints/saved_model_epoch_33_iou_0.82.h5")
    # predict_single_image('dataset/testing/input/img-1.png',
    #                      'trained_models/roads_prediction_model_06.12.2019_01:50:39.h5',
    #                      save_result=True)
    # predict_image('tests/1.JPG', 'trained_models/roads_prediction_model_06.12.2019_01:50:39.h5')
    # train_model()
