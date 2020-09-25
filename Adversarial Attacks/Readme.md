### create directory for dataset and model 
```
import os
dataset_path = "data/dl-tsc/archives/TSC/HAR/"
pretrain_model_path = "data/dl-tsc/results/resnet/TSC/HAR/"

os.makedirs(pretrain_model_path)
```


### ijcnn19attacks/src/main.py
```
root_dir = 'data/dl-tsc'
archive_name = 'TSC'
```

### ijcnn19attacks/src/cleverhans_tutorials/tsc_tutorial_keras_tf.py
***import settings***
```
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend
```


**113th to 126th lines**
```
    tf.keras.backend.set_learning_phase(False)

    if not hasattr(backend, "backend"):
      raise RuntimeError("This tutorial requires keras to be configured to use the TensorFlow backend.")

    if keras.backend.image_data_format() != 'channels_last':
      keras.backend.set_image_data_format('channels_last')
      print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'th', temporarily setting to 'tf'")
              
    sess = tf.Session()
    backend.set_session(sess)

    root_dir = 'data/dl-tsc/'
```



### env
```
!pip install tensorflow==1.14.0
!pip install keras==2.3.1
```
