from keras.models import Model
from keras.layers import Input, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D


def Lnet ( input_height=321, input_width=481, nChannels=18, optimizer=None ): 
    
    inputs = Input((input_height, input_width, nChannels))
    inputa = AveragePooling2D()(inputs)
    
    conv1 = SeparableConv2D(8, 3, activation='relu')(inputa)
    conv1 = MaxPooling2D()(conv1)
    
    conv2 = SeparableConv2D(16, 3, activation='relu')(conv1)
    conv2 = MaxPooling2D()(conv2)
    
    conv3 = SeparableConv2D(32, 3, activation='relu')(conv2)
    conv3 = MaxPooling2D()(conv3)
    
    conv4 = SeparableConv2D(64, 3, activation='relu')(conv3)
    conv4 = MaxPooling2D()(conv4)
    
    conv5 = SeparableConv2D(1 , 1, activation='relu')(conv4)
    conv5 = GlobalAveragePooling2D()(conv5)
    
    model = Model(inputs=inputs, outputs=conv5)
    
    if not optimizer is None:
        model.compile(loss='mean_squared_error', optimizer= optimizer )
    
    return model


# def Lnet ( input_height=321, input_width=481, nChannels=18, optimizer=None ): 
    
    # inputs = Input((input_height, input_width, nChannels))
    # inputa = AveragePooling2D()(inputs)
    
    # conv1 = SeparableConv2D( 4, 3, activation='relu')(inputa)
    # conv1 = MaxPooling2D()(conv1)
    
    # conv2 = SeparableConv2D( 8, 3, activation='relu')(conv1)
    # conv2 = MaxPooling2D()(conv2)
    
    # conv3 = SeparableConv2D(16, 3, activation='relu')(conv2)
    # conv3 = MaxPooling2D()(conv3)
    
    # conv4 = SeparableConv2D( 1, 1, activation='relu')(conv3)
    # conv4 = GlobalAveragePooling2D()(conv4)
    
    # model = Model(inputs=inputs, outputs=conv4)
    
    # if not optimizer is None:
        # model.compile(loss='mean_squared_error', optimizer= optimizer )
    
    # return model