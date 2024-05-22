# %%
# %pip uninstall tensorflow[and-cuda]
# %pip install tensorflow[and-cuda] --upgrade
# %pip install opencv-python numpy matplotlib pillow
# %pip install keras
# %pip show keras

# !export CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# !export LD_LIBRARY_PATH=${CUDNN_PATH}/lib


# %%
import cv2, numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
import os
from tensorflow.keras.layers import  Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras import Layer, Model
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model

from time import sleep


# %%
tf.config.list_physical_devices('GPU') 

# %%
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    ## Magic happens here - similarity calculation
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)
    

model = load_model("file_modelatm2.keras",custom_objects={'L1Dist':L1Dist})
model.training = False

for layer in model.layers: 
    print(layer.get_config(), layer.get_weights())
    
    # print(layer.get_weights())

model.summary()

# %%
lays = model.layers

lays_dict = dict(  [ (lay.name,lay) for lay in lays]     )

print(lays_dict['input_img'],lays_dict['validation_img'],sep='\n')

embedding_weights = lays_dict['Embedding_Layer'].get_weights()
dense_weights, dense_biases = lays_dict['dense_1'].get_weights()

# %%
def Embedding_Layer(): 
    input_layer = Input(shape=(100,100,3), name='input_image')
    
    #first block
    convolution_layer1 = Conv2D(64, (10,10), activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(64, (2,2), padding='same')(convolution_layer1)
    
    #second block
    convolution_layer2 = Conv2D(128, (7,7), activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(64, (2,2), padding='same')(convolution_layer2)
    
    #third block 
    convolution_layer3 = Conv2D(128, (4,4), activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling2D(64, (2,2), padding='same')(convolution_layer3)
    
    #fourth & final block
    convolution_layer4 = Conv2D(256, (4,4), activation='relu')(max_pooling3)
    flatten_layer1 = Flatten()(convolution_layer4)
    dense_layer1 = Dense(4096, activation='sigmoid')(flatten_layer1)
    
    
    return Model(inputs=input_layer, outputs=dense_layer1, name='Embedding_Layer')
    
model1= Embedding_Layer()

def siamese_model(): 
    
    # base image input into the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siameselayer = L1Dist(name='l1_layer')
    dist = siameselayer([model1(input_image), model1(validation_image)])
    
    # Classification layer 
    classify = Dense(1, activation='sigmoid',name='output_layer')(dist)
    
    return Model(inputs=[input_image, validation_image], outputs=classify, name='SiameseNetwork')

# for t in dense_weights:
#     t*=-1

SiameseModel = siamese_model()
SiameseModel.get_layer('Embedding_Layer').set_weights(embedding_weights)
SiameseModel.get_layer('output_layer').set_weights((dense_weights,dense_biases))

for layer in SiameseModel.layers: 
    print(layer.get_config(), layer.get_weights())


# %%
SiameseModel.summary()


# %%
def preprocessing_func(file_path):
    
    # Read in image from file path
    byte_image = tf.io.read_file(file_path)
    # Load image 
    imag = tf.io.decode_jpeg(byte_image)
    
    # Preprocessing: resize image -> 100*100*3
    imag = tf.image.resize(imag, (100,100))
    # Scale image to be between 0 and 1 
    imag = imag / 255.0
    imag = np.expand_dims(imag,axis=0)
    return imag


def preprocess_identical(input_img, validation_img, label):
    #label: indicates whether the input and validation images are identical or not
    return(preprocessing_func(input_img), preprocessing_func(validation_img), label)

# %%
for layer in model.layers: 
    print(layer.get_config(), layer.get_weights())


# %%
print(SiameseModel.get_layer('Embedding_Layer').get_weights() )

# %%
def get_face(img):
    face_classifier = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    face = face_classifier.detectMultiScale( 
    img, scaleFactor=1.1, minNeighbors=9) 

# %%
filename_known1 = '../Known_Faces/20180596_Yazan-Matarweh.png'
# filename_known1 = '20180596_Yazan-Matarweh.png'
filename_unknown1 = "../../../../../Pictures/Camera Roll/WIN_20240517_22_16_08_Pro.jpg"

filename_known2 = '../Known_Faces/20021736_Anne-Hathaway.png'

filename_known3 = '../Known_Faces/19850192_Rosie-O\'Donnell.jpg'

# plt.imshow(plt.imread(filename_unknown1)) 

known1 = preprocessing_func(filename_known1)
unknown1 = preprocessing_func(filename_unknown1)

known2 = preprocessing_func(filename_known2)

known3 = preprocessing_func(filename_known3)
# label = filename_known.split('/')[-1].split('_')[1]

challenges =  ([known1,unknown1],[known2,unknown1],[known3,unknown1])

for challengne in challenges:
    prediction = SiameseModel.predict(challengne)
    print(prediction)



# %%



