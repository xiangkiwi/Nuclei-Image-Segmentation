# Nuclei Segmentation Image

The dataset is provided by Kaggle: https://www.kaggle.com/c/data-science-bowl-2018.
The project is built to do segmentation on nuclei image.

## Methodologies
This project is done under Spyder IDE with TensorFlow Keras functional API framework.

In the provided dataset, it has been distributed to train and test datasets. The train dataset will be splited into 80:20, where 80% is for training the model and remaining 20% is for validating. Meanwhile, the test datasets is used for testing.

All the numpy array data is then converted into tensor slice to zip the tensor slice into dataset.

```sh
train = tf.data.Dataset.zip((train_x, train_y))
validation = tf.data.Dataset.zip((test_x,test_y))
test = tf.data.Dataset.zip((ori_test_x, ori_test_y))
```

This part of coding can be found inside the `nuclei_image_segmentation.py`.

To train this model, mobilenet_v2 process is used. The training epochs is set to 20.

```sh
base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
```

This part of coding can be found inside the `concrete_crack_classify.py`.

## Results
After running with 20 epochs, the accuracy can reach around 97%. 

✨Result after running 20 epochs✨

![training result](https://user-images.githubusercontent.com/34246703/163669454-3d7f4e06-3fce-4c41-a6e2-1107fbd70da2.PNG)

✨Some examples of actual value of test data vs prediction value✨

![result 1](https://user-images.githubusercontent.com/34246703/163669462-21611c8c-59ca-448c-af78-d403e1e30ff1.PNG)
![result 2](https://user-images.githubusercontent.com/34246703/163669469-b45928be-1881-4ff6-9738-351800716980.PNG)
