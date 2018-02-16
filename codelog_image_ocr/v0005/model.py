from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, Concatenate
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, CSVLogger

# Network parameters
conv_filters = 16
kernel_size = (3, 3)
pool_size = 3
time_dense_size = 32
rnn_size = 512

def create_tensor_io(img_w, img_h, channel_count, label_count):
    input_shape = (img_w, img_h, channel_count)
    act = 'relu'

    tensor_in = Input(shape=input_shape, name='the_input')

    tensor = tensor_in

    tensor = Conv2D(filters=32, kernel_size=1,     padding='same', activation=act, name='a0')(tensor)
    tensor = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation=act, name='a1')(tensor)
    tensor = Conv2D(filters=32, kernel_size=1,     padding='same', activation=act, name='a2')(tensor)
    tensor = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation=act, name='a3')(tensor)
    tensor = Conv2D(filters=32, kernel_size=1,     padding='same', activation=act, name='a4')(tensor)
    tensor = Conv2D(filters=32, kernel_size=(5,5), padding='same', activation=act, name='a5')(tensor)
    tensor = Conv2D(filters=32, kernel_size=1,     padding='same', activation=act, name='a6')(tensor)

    shape = K.int_shape(tensor)
    tensor = Reshape(target_shape=(shape[1],shape[2]*shape[3]), name='b0')(tensor)
    tensor = Dense(512,activation=act, name='b1')(tensor)
    #tensor = Dropout(2-PHI, name='b2')(tensor)

    tensor_0 = GRU(512, return_sequences=True, go_backwards=False, name='c00')(tensor)
    tensor_1 = GRU(512, return_sequences=True, go_backwards=True , name='c01')(tensor)
    tensor = Concatenate(axis=2, name='c02')([tensor_0,tensor_1])
    tensor = Dense(512,activation=act, name='c03')(tensor)
    #tensor = Dropout(2-PHI, name='c04')(tensor)

    tensor_0 = GRU(512, return_sequences=True, go_backwards=False, name='c10')(tensor)
    tensor_1 = GRU(512, return_sequences=True, go_backwards=True , name='c11')(tensor)
    tensor = Concatenate(axis=2, name='c12')([tensor_0,tensor_1])
    tensor = Dense(512,activation=act, name='c13')(tensor)
    #tensor = Dropout(2-PHI, name='c14')(tensor)

    tensor_0 = GRU(512, return_sequences=True, go_backwards=False, name='c20')(tensor)
    tensor_1 = GRU(512, return_sequences=True, go_backwards=True , name='c21')(tensor)
    tensor = Concatenate(axis=2, name='c22')([tensor_0,tensor_1])
    tensor = Dense(512,activation=act, name='c23')(tensor)
    #tensor = Dropout(2-PHI, name='c24')(tensor)

    tensor = Dense(512, activation=act, name='d0')(tensor)
    #tensor = Dropout(2-PHI, name='d2')(tensor)
    tensor = Dense(512, activation=act, name='d3')(tensor)
    #tensor = Dropout(2-PHI, name='d5')(tensor)
    tensor = Dense(512, activation=act, name='d6')(tensor)
    #tensor = Dropout(2-PHI, name='d8')(tensor)
    #tensor = Dense(label_count, activity_regularizer=regularizers.l1(0.01/(label_count*input_shape[1])), name='d9')(tensor)
    #tensor = Dense(label_count, activity_regularizer=regularizers.l1(0.01), name='d9')(tensor)
    tensor = Dense(label_count, name='d9')(tensor)

    tensor = Activation('softmax', name='e0')(tensor)

    tensor_out = tensor

    return tensor_in, tensor_out

if __name__ == '__main__':
    input_data, y_pred = create_tensor_io(160, 32, 3, 11)
    model = Model(inputs=input_data, outputs=y_pred)
    model.summary()
    