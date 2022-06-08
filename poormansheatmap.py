"""
              (  .      )
           )           (              )
                 .  '   .   '  .  '  .
        (    , )       (.   )  (   ',    )
         .' ) ( . )    ,  ( ,     )   ( .
      ). , ( .   (  ) ( , ')  .' (  ,    )
     (_,) . ), ) _) _,')  (, ) '. )  ,. (' )
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^°°°
         Feel the HEAT, feel the MAP.

Contact: mark.schutera@mailbox.org or find me on LinkedIn - Mark Schutera
Explainable and Trustworthy AI remains to be a central topic during deployment of deep neural networks.
This project is intended to show an easy and simplistic approach to make classifications more interpretable.
Contributions are welcome.

"""

import cv2  # pip install opencv-python
import tensorflow as tf
import numpy as np


def maptheheat(model, image, mode='gaussian', class_under_test=-1, columns=6, rows=12):
    """
    Central function to generate the heatmap

    :arguments:
        :param model: Input model to generate heatmap from, pretrained and preloaded with keras
        :param image: Input image to generate heatmap from, loaded with the keras utils
        :param mode:    Specifies the mode of the value throughout the heatmap tiles
                        e.g. 'gaussian', 'None' meaning discrete
        :param class_under_test:    Specifies the class under test (starts with 0),
                                    defaults (for -1) to the predicted class,
                                    but allows to see tiles that might have influenced the prediction
                                    towards other classes.
        :param columns: Number of tile columns
        :param rows: Number of tile rows

    :return:
        input: Original input image
        heat: Array associating a grid tile (row, column) with prediction confidence reached without it.
        heatmap: Heat visualized as grid tiles
        overlay: Heatmap mapped over the original input image
    """

    # Predict on the input
    input = yourpreprocessing(image)
    heatmap = np.copy(input)
    prediction = tf.nn.softmax(model.predict(input))

    if int(class_under_test) > len(prediction[0]) or class_under_test == -1:
        print('Warning! The class you specified exceeds the available number of classes your model has been trained on')
        print('Proceeding as if class under test has not been specified.')
        class_under_test = np.argmax(prediction, axis=1)[0]
    print('Class under test is, ', class_under_test)

    prediction_value = prediction[0][class_under_test]

    print('Prediction is: ', prediction)
    print('Prediction confidence on the sample is: ', prediction_value)

    # get image dimensions
    # print('Input shape: ', np.shape(input))
    width = np.shape(input)[1]
    height = np.shape(input)[2]
    tile_width = width / columns
    tile_height = height / rows

    # get tile results
    heat = []
    for r in range(0, rows):
        x1 = int(r * tile_height)
        x2 = int((r+1) * tile_height)
        for c in range(0, columns):
            y1 = int(c * tile_width)
            y2 = int((c+1) * tile_width)

            noised_input = overlaynoise(input, x1, y1, x2, y2)
            prediction = tf.nn.softmax(model.predict(noised_input))
            tile_value = np.abs(prediction_value - prediction[0][class_under_test])
            heat.append([r, c, tile_value])
            heatmap = addtheheat(tile_value, x1, y1, x2, y2, heatmap)

    heatmap = heatmapconfig(heatmap[0], mode)

    return input[0], heat, heatmap, input[0]+heatmap


def heatmapconfig(heatmap, mode='gaussian'):
    if mode == 'gaussian':
        heatmap = cv2.GaussianBlur(heatmap, (9, 9), cv2.BORDER_DEFAULT)
    else:
        print('The mode you have specified is not yet implemented. Progressing with the discrete mode')
        pass
    return heatmap

def yourpreprocessing(image):
    # Add any preprocessing steps you might have used on your images before they enter your model during training
    # e.g. in our case the tensorflow hub models expect images as float inputs [0,1] with a specific height and width
    image = tf.keras.preprocessing.image.img_to_array(image)

    # Resizing
    img_height = 224
    img_width = 224
    resizing_layer = tf.keras.layers.Resizing(img_height, img_width)
    re_image = resizing_layer(image)

    # Normalization
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    norm_image = normalization_layer(re_image)

    prep_input = np.array([norm_image])
    return prep_input


def overlaynoise(image, x1, y1, x2, y2):
    '''
    :param image: original image
    :param x1, y1, x2, y2: specified bounding box coordinates
    :return: Overlays the image with gaussian noise in the specified bounding box
    '''
    # print('image tile', image[0][x1:x2, y1:y2, 0])
    im = np.copy(image)
    im[0][x1:x2, y1:y2, 0] = np.random.rand(x2-x1, y2-y1)
    im[0][x1:x2, y1:y2, 1] = np.random.rand(x2-x1, y2-y1)
    im[0][x1:x2, y1:y2, 2] = np.random.rand(x2-x1, y2-y1)
    # print('image with noise', image)
    return im


def addtheheat(tile_value, x1, y1, x2, y2, heatmap):
    heatmap[0][x1:x2, y1:y2, 0] = 0
    heatmap[0][x1:x2, y1:y2, 1] = 0
    heatmap[0][x1:x2, y1:y2, 2] = tile_value
    return heatmap


if __name__ == '__main__':
    '''
    This is an exemplary run of the package.
    '''

    # //////////////////////////////////////// Load image
    yourimage = tf.keras.utils.load_img(
        "./data/validation/tulip/tulip1.jpg",
        grayscale=False,
        color_mode='rgb',
        target_size=None,
        interpolation='nearest'
    )

    # //////////////////////////////////////// Load model
    # load a model trained on the flower dataset, specs for the provided one:
    # Epoch 10/10
    # 92/92 [==============================]
    # - 162s 2s/step - loss: 0.1436 - acc: 0.9646 - val_loss: 0.3542 - val_acc: 0.8856
    model_name = "flower_model"
    import_path = "./tmp/saved_models/{}".format(model_name)
    yourmodel = tf.keras.models.load_model(import_path)

    # //////////////////////////////////////// Heatmap
    input, heat, heatmap, overlay = maptheheat(yourmodel, yourimage, mode='gaussian', class_under_test=-1, columns=6, rows=12)

    cv2.imshow("overlay", overlay)
    cv2.waitKey(0)
