# Layer by layer pretraining Models

# Layer 1
input_img = Input(shape = (784, ))
distorted_input1 = Dropout(.1)(input_img)
encoded1 = Dense(800, activation = 'sigmoid')(distorted_input1)
encoded1_bn = BatchNormalization()(encoded1)
decoded1 = Dense(784, activation = 'sigmoid')(encoded1_bn)

autoencoder1 = Model(input = input_img, output = decoded1)
encoder1 = Model(input = input_img, output = encoded1_bn)

# Layer 2
encoded1_input = Input(shape = (800,))
distorted_input2 = Dropout(.2)(encoded1_input)
encoded2 = Dense(400, activation = 'sigmoid')(distorted_input2)
encoded2_bn = BatchNormalization()(encoded2)
decoded2 = Dense(800, activation = 'sigmoid')(encoded2_bn)

autoencoder2 = Model(input = encoded1_input, output = decoded2)
encoder2 = Model(input = encoded1_input, output = encoded2_bn)

# Layer 3 - which we won't end up fitting in the interest of time
encoded2_input = Input(shape = (400,))
distorted_input3 = Dropout(.3)(encoded2_input)
encoded3 = Dense(200, activation = 'sigmoid')(distorted_input3)
encoded3_bn = BatchNormalization()(encoded3)
decoded3 = Dense(400, activation = 'sigmoid')(encoded3_bn)

autoencoder3 = Model(input = encoded2_input, output = decoded3)
encoder3 = Model(input = encoded2_input, output = encoded3_bn)

# Deep Autoencoder
encoded1_da = Dense(800, activation = 'sigmoid')(input_img)
encoded1_da_bn = BatchNormalization()(encoded1_da)
encoded2_da = Dense(400, activation = 'sigmoid')(encoded1_da_bn)
encoded2_da_bn = BatchNormalization()(encoded2_da)
encoded3_da = Dense(200, activation = 'sigmoid')(encoded2_da_bn)
encoded3_da_bn = BatchNormalization()(encoded3_da)
decoded3_da = Dense(400, activation = 'sigmoid')(encoded3_da_bn)
decoded2_da = Dense(800, activation = 'sigmoid')(decoded3_da)
decoded1_da = Dense(784, activation = 'sigmoid')(decoded2_da)

deep_autoencoder = Model(input = input_img, output = decoded1_da)

# Not as Deep Autoencoder
nad_encoded1_da = Dense(800, activation = 'sigmoid')(input_img)
nad_encoded1_da_bn = BatchNormalization()(nad_encoded1_da)
nad_encoded2_da = Dense(400, activation = 'sigmoid')(nad_encoded1_da_bn)
nad_encoded2_da_bn = BatchNormalization()(nad_encoded2_da)
nad_decoded2_da = Dense(800, activation = 'sigmoid')(nad_encoded2_da_bn)
nad_decoded1_da = Dense(784, activation = 'sigmoid')(nad_decoded2_da)

nad_deep_autoencoder = Model(input = input_img, output = nad_decoded1_da)

sgd1 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)
sgd2 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)
sgd3 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)

autoencoder1.compile(loss='binary_crossentropy', optimizer = sgd1)
autoencoder2.compile(loss='binary_crossentropy', optimizer = sgd2)
autoencoder3.compile(loss='binary_crossentropy', optimizer = sgd3)

encoder1.compile(loss='binary_crossentropy', optimizer = sgd1)
encoder2.compile(loss='binary_crossentropy', optimizer = sgd1)
encoder3.compile(loss='binary_crossentropy', optimizer = sgd1)

deep_autoencoder.compile(loss='binary_crossentropy', optimizer = sgd1)
nad_deep_autoencoder.compile(loss='binary_crossentropy', optimizer = sgd1)

# What will happen to the learnning rates under this decay schedule?
lr = 5
for i in range(12):
    lr = lr - lr * .15
    print(lr)    