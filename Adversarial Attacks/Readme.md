### env
```
!pip3 install --upgrade pip
!pip install tensorflow==1.14.0
!pip install keras==2.3.1
```



### ijcnn19attacks/src/main.py
```
root_dir = 'data/dl-tsc'
archive_name = 'TSC'
```

### ijcnn19attacks/src/cleverhans_tutorials/tsc_tutorial_keras_tf.py
**113th to 126th lines**
```
tf.keras.backend.set_learning_phase(0)

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    # tf.random.set_seed(1234)
    tf.set_random_seed(1234)

    if not hasattr(backend, "tensorflow_backend"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    if keras.backend.image_data_format() != 'channels_last':
        keras.backend.set_image_data_format('channels_last')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")
              
    sess = tf.compat.v1.Session()
    keras.backend.set_session(sess)

    root_dir = 'data/dl-tsc/'
```
