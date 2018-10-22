def resnet_module(X, modify_dim = False, channels = 64):
    if modify_dim is False:
        X2 = X
    elif modify_dim is True:
        X2 = Conv2D(channels, (1,1),padding='same',strides=1)(X)        
    X1 = BatchNormalization()(X)
    X1 = Activation('relu')(X1)
    X1 = Conv2D(channels, (3,3),padding='same')(X1)
    X1 = BatchNormalization()(X1)
    X1 = Activation('relu')(X1) 
    X1 = Conv2D(channels, (3,3),padding='same')(X1)
    out = Add()([X1,X2])
    return out
