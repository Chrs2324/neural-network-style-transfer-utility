import os
import sys
import random
import numpy as np
from PIL import Image
from random import randint
from copy import deepcopy
import matplotlib.pyplot as plt

import keras.backend as K
from keras import initializers
from keras.utils import plot_model
from keras.layers.core import Activation
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam, SGD, Nadam, Adamax
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, MaxPooling2D, Deconvolution2D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Lambda, Concatenate

class Generator(object):
    def __init__(self, width = 28, height= 28, channels = 1):
        
        self.W = width
        self.H = height
        self.C = channels
        self.SHAPE = (width,height,channels)

        self.Generator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.Generator.compile(loss='binary_crossentropy', optimizer=self.OPTIMIZER,metrics=['accuracy'])

        # self.save_model()
        self.summary()

    def model(self):
        input_layer = Input(shape=self.SHAPE)
        
        down_1 = Convolution2D(64  , kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)
        norm_1 = InstanceNormalization()(down_1)

        down_2 = Convolution2D(64*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_1)
        norm_2 = InstanceNormalization()(down_2)

        down_3 = Convolution2D(64*4, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_2)
        norm_3 = InstanceNormalization()(down_3)

        down_4 = Convolution2D(64*8, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_3)
        norm_4 = InstanceNormalization()(down_4)


        upsample_1 = UpSampling2D()(norm_4)
        up_conv_1 = Convolution2D(64*4, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_1)
        norm_up_1 = InstanceNormalization()(up_conv_1)
        add_skip_1 = Concatenate()([norm_up_1,norm_3])

        upsample_2 = UpSampling2D()(add_skip_1)
        up_conv_2 = Convolution2D(64*2, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_2)
        norm_up_2 = InstanceNormalization()(up_conv_2)
        add_skip_2 = Concatenate()([norm_up_2,norm_2])

        upsample_3 = UpSampling2D()(add_skip_2)
        up_conv_3 = Convolution2D(64, kernel_size=4, strides=1, padding='same',activation='relu')(upsample_3)
        norm_up_3 = InstanceNormalization()(up_conv_3)
        add_skip_3 = Concatenate()([norm_up_3,norm_1])

        last_upsample = UpSampling2D()(add_skip_3)
        output_layer = Convolution2D(3, kernel_size=4, strides=1, padding='same',activation='tanh')(last_upsample)
        
        return Model(input_layer,output_layer)

    def summary(self):
        return self.Generator.summary()

    def save_model(self):
        plot_model(self.Generator.model, to_file='/mnt/d/OpenStylizer/data/Generator_Model.png')


class Discriminator(object):
    def __init__(self, width = 28, height= 28, channels = 1):
        self.W = width
        self.H = height
        self.C = channels
        self.CAPACITY = width*height*channels
        self.SHAPE = (width,height,channels)
        
        self.Discriminator = self.model()
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.Discriminator.compile(loss='mse', optimizer=self.OPTIMIZER, metrics=['accuracy'] )

        # self.save_model()
        self.summary()

    def model(self):
        input_layer = Input(self.SHAPE)

        up_layer_1 = Convolution2D(64, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)

        up_layer_2 = Convolution2D(64*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(up_layer_1)
        norm_layer_1 = InstanceNormalization()(up_layer_2)

        up_layer_3 = Convolution2D(64*4, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_layer_1)
        norm_layer_2 = InstanceNormalization()(up_layer_3)

        up_layer_4 = Convolution2D(64*8, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(norm_layer_2)
        norm_layer_3 =InstanceNormalization()(up_layer_4)

        output_layer = Convolution2D(1, kernel_size=4, strides=1, padding='same')(norm_layer_3)
        output_layer_1 = Flatten()(output_layer)
        output_layer_2 = Dense(1, activation='sigmoid')(output_layer_1)
        
        return Model(input_layer,output_layer_2)

    def summary(self):
        return self.Discriminator.summary()

    def save_model(self):
        plot_model(self.Discriminator.model, to_file='/mnt/d/OpenStylizer/data/Discriminator_Model.png')

class GAN(object):
    def __init__(self, model_inputs=[],model_outputs=[],lambda_cycle=10.0,lambda_id=1.0):
        self.OPTIMIZER = SGD(lr=2e-4,nesterov=True)
        # self.inputs are represented by an array of two Keras input classes instantiated in the training class and passed to the GAN class.
        self.inputs = model_inputs
        # Create a model with the input and output passed from the training class.
        self.outputs = model_outputs
        self.gan_model = Model(self.inputs,self.outputs)
        self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5)
        # The output array is six models, four generators, and two discriminators in an adversarial setup.
        self.gan_model.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            lambda_cycle, lambda_cycle,
                                            lambda_id, lambda_id ],
                            optimizer=self.OPTIMIZER)
        #self.save_model()
        self.summary()

    def model(self):
        model = Model()
        return model

    def summary(self):
        return self.gan_model.summary()

    def save_model(self):
        plot_model(self.gan_model.model, to_file='/mnt/d/OpenStylizer/data/GAN Model/GAN_Model.png')  

class Trainer:
    def __init__(self, height = 64, width = 64, epochs = 5, batch = 32, checkpoint = 50, train_data_path_A = '',train_data_path_B = '',test_data_path_A='',test_data_path_B='',lambda_cycle=10.0,lambda_id=1.0):
        self.EPOCHS = epochs
        self.BATCH = batch
        self.RESIZE_HEIGHT = height
        self.RESIZE_WIDTH = width
        self.CHECKPOINT = checkpoint
        # Load all of the data into its respective class variables
        self.X_train_A, self.H_A, self.W_A, self.C_A = self.load_data(train_data_path_A)
        self.X_train_B, self.H_B, self.W_B, self.C_B  = self.load_data(train_data_path_B)
        self.X_test_A, self.H_A_test, self.W_A_test, self.C_A_test = self.load_data(test_data_path_A)
        self.X_test_B, self.H_B_test, self.W_B_test, self.C_B_test  = self.load_data(test_data_path_B)
         
        '''We need the generators that go from A to B and from B to A. 
        The instantiation of these models is direct.'''
        self.generator_A_to_B = Generator(height=self.H_A, width=self.W_A, channels=self.C_A)
        self.generator_B_to_A = Generator(height=self.H_B, width=self.W_B, channels=self.C_B)
        
        #  following lines to instantiation in the class definition for training
        '''We need to make sure we have the original A and B images stored as the Input class from Keras. 
        Variables orig_A and orig_B are the input values shared among the next three components.''' 
        self.orig_A = Input(shape=(self.W_A, self.H_A, self.C_A))
        self.orig_B = Input(shape=(self.W_B, self.H_B, self.C_B))
        '''fake_A and fake_B are the generators that take us from one style to the other and produce 
        an image with the translated style. Hence, this is why we say they are fake.'''
        self.fake_B = self.generator_A_to_B.Generator(self.orig_A)
        self.fake_A = self.generator_B_to_A.Generator(self.orig_B)
        '''reconstructed_A and reconstructed_B take the fake A and B images and retranslate 
        them into the original image style.'''
        self.reconstructed_A = self.generator_B_to_A.Generator(self.fake_B)
        self.reconstructed_B = self.generator_A_to_B.Generator(self.fake_A)
        '''id_A and id_B are identity functions because they take in the original image and translate 
        back into the same style. Ideally, these functions would not apply any style changes to these images'''
        self.id_A = self.generator_B_to_A.Generator(self.orig_A)
        self.id_B = self.generator_A_to_B.Generator(self.orig_B)

        '''We need our discriminators that evaluate both A and B images. 
        We also need a validity discriminator that checks the fake_A and fake_B generators'''
        self.discriminator_A = Discriminator(height=self.H_A, width=self.W_A, channels=self.C_A)
        self.discriminator_B = Discriminator(height=self.H_B, width=self.W_B, channels=self.C_B)
        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False
        self.valid_A = self.discriminator_A.Discriminator(self.fake_A)
        self.valid_B = self.discriminator_B.Discriminator(self.fake_B)

        ''' we are able to simply pass all of the models to the GAN class 
        and it will construct our adversarial model'''
        model_inputs  = [self.orig_A,self.orig_B]
        model_outputs = [self.valid_A, self.valid_B,self.reconstructed_A,self.reconstructed_B,self.id_A, self.id_B]
        self.gan = GAN(model_inputs=model_inputs,model_outputs=model_outputs,lambda_cycle=lambda_cycle,lambda_id=lambda_id)
        
    # The load_data function expects a string that represents the path to the folder and it'll read every image with a certain file ending within that folder.
    def load_data(self,data_path,amount_of_data = 1.0):
        listOFFiles = self.grabListOfFiles(data_path,extension="jpg")
        X_train = np.array(self.grabArrayOfImages(listOFFiles))
        height, width, channels = np.shape(X_train[0])
        X_train = X_train[:int(amount_of_data*float(len(X_train)))]
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = np.expand_dims(X_train, axis=3)
        return X_train, height, width, channels

    def grabListOfFiles(self,startingDirectory,extension=".webp"):
        listOfFiles = []
        for file in os.listdir(startingDirectory):
            if file.endswith(extension):
                listOfFiles.append(os.path.join(startingDirectory, file))
        return listOfFiles

    def grabArrayOfImages(self,listOfFiles,gray=False):
        imageArr = []
        for f in listOfFiles:
            if gray:
                im = Image.open(f).convert("L")
            else:
                im = Image.open(f).convert("RGB")
            im = im.resize((self.RESIZE_WIDTH,self.RESIZE_HEIGHT))
            imData = np.asarray(im)
            imageArr.append(imData)
        return imageArr

    '''we need to collect our data for our batch generator differently and 
    we need to train each one of the discriminators we just developed (four in total)'''
    def train(self):
        for e in range(self.EPOCHS):
            b = 0
            X_train_A_temp = deepcopy(self.X_train_A)
            X_train_B_temp = deepcopy(self.X_train_B)
            '''Because the batch represents a single image, it isn't strictly required that each domain 
            contain the same number of images. Now, this means that our while statement needs to take 
            into account that there is one folder smaller than the other. The epoch will end when 
            there are no more images in the smaller array of images between A and B.'''
            while min(len(X_train_A_temp),len(X_train_B_temp))>self.BATCH:
                # Keep track of Batches
                b=b+1

                # Train Discriminator
                # Grab Real Images for this training batch
                '''we need to have an A and B version of our batches'''
                count_real_images = int(self.BATCH)
                starting_indexs = randint(0, (min(len(X_train_A_temp),len(X_train_B_temp))-count_real_images))
                real_images_raw_A = X_train_A_temp[ starting_indexs : (starting_indexs + count_real_images) ]
                real_images_raw_B = X_train_B_temp[ starting_indexs : (starting_indexs + count_real_images) ]

                # Delete the images used until we have none left
                X_train_A_temp = np.delete(X_train_A_temp,range(starting_indexs,(starting_indexs + count_real_images)),0)
                X_train_B_temp = np.delete(X_train_B_temp,range(starting_indexs,(starting_indexs + count_real_images)),0)
                batch_A = real_images_raw_A.reshape( count_real_images, self.W_A, self.H_A, self.C_A )
                batch_B = real_images_raw_B.reshape( count_real_images, self.W_B, self.H_B, self.C_B )

                self.discriminator_A.Discriminator.trainable = True
                self.discriminator_B.Discriminator.trainable = True
                x_batch_A = batch_A
                x_batch_B = batch_B
                y_batch_A = np.ones([count_real_images,1])
                y_batch_B = np.ones([count_real_images,1])
                # Now, train the discriminator with this batch of reals
                discriminator_loss_A_real = self.discriminator_A.Discriminator.train_on_batch(x_batch_A,y_batch_A)[0]
                discriminator_loss_B_real = self.discriminator_B.Discriminator.train_on_batch(x_batch_B,y_batch_B)[0]

                x_batch_B = self.generator_A_to_B.Generator.predict(batch_A)
                x_batch_A = self.generator_B_to_A.Generator.predict(batch_B)
                y_batch_A = np.zeros([self.BATCH,1])
                y_batch_B = np.zeros([self.BATCH,1])
                # Now, train the discriminator with this batch of fakes
                discriminator_loss_A_fake = self.discriminator_A.Discriminator.train_on_batch(x_batch_A,y_batch_A)[0]
                discriminator_loss_B_fake = self.discriminator_B.Discriminator.train_on_batch(x_batch_B,y_batch_B)[0]    

                self.discriminator_A.Discriminator.trainable = False
                self.discriminator_B.Discriminator.trainable = False

                discriminator_loss_A = 0.5*(discriminator_loss_A_real + discriminator_loss_A_fake)
                discriminator_loss_B = 0.5*(discriminator_loss_B_real + discriminator_loss_B_fake)
            
                # In practice, flipping the label when training the generator improves convergence
                '''we introduce label noise into the training process with the development of the 
                batches for training the individual discriminators'''
                if self.flipCoin(chance=0.9):
                    y_generated_labels = np.ones([self.BATCH,1])
                else:
                    y_generated_labels = np.zeros([self.BATCH,1])
                generator_loss = self.gan.gan_model.train_on_batch([x_batch_A, x_batch_B],
                                                        [y_generated_labels, y_generated_labels,
                                                        x_batch_A, x_batch_B,
                                                        x_batch_A, x_batch_B])    

                print ('Epoch: '+str(int(e))+' Batch: '+str(int(b))+', [Discriminator_A :: Loss: '+str(discriminator_loss_A)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                print ('Epoch: '+str(int(e))+' Batch: '+str(int(b))+', [Discriminator_B :: Loss: '+str(discriminator_loss_B)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                if b % self.CHECKPOINT == 0 :
                    label = str(e)+'_'+str(b)
                    self.plot_checkpoint(label)

            print ('Epoch: '+str(int(e))+', [Discriminator_A :: Loss: '+str(discriminator_loss_A)+'], [ Generator :: Loss: '+str(generator_loss)+']')
            print ('Epoch: '+str(int(e))+', [Discriminator_A :: Loss: '+str(discriminator_loss_B)+'], [ Generator :: Loss: '+str(generator_loss)+']')
                        
            #if e % self.CHECKPOINT == 0 :
                #self.plot_checkpoint(e)
        return
 
    def flipCoin(self,chance=0.5):
        return np.random.binomial(1, chance)

    def plot_checkpoint(self,b):
        orig_filename = "/mnt/d/OpenStylizer/data/batch_check_"+str(b)+"_original.png"

        image_A = self.X_test_A[5]
        image_A = np.reshape(image_A, [self.W_A_test,self.H_A_test,self.C_A_test])
        print("Image_A shape: " +str(np.shape(image_A)))
        fake_B = self.generator_A_to_B.Generator.predict(image_A.reshape(1, self.W_A, self.H_A, self.C_A ))
        fake_B = np.reshape(fake_B, [self.W_A_test,self.H_A_test,self.C_A_test])
        print("fake_B shape: " +str(np.shape(fake_B)))
        reconstructed_A = self.generator_B_to_A.Generator.predict(fake_B.reshape(1, self.W_A, self.H_A, self.C_A ))
        reconstructed_A = np.reshape(reconstructed_A, [self.W_A_test,self.H_A_test,self.C_A_test])
        print("reconstructed_A shape: " +str(np.shape(reconstructed_A)))
        # from IPython import embed; embed()

        checkpoint_images = np.array([image_A, fake_B, reconstructed_A])

        # Rescale images 0 - 1
        checkpoint_images = 0.5 * checkpoint_images + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axes = plt.subplots(1, 3)
        for i in range(3):
            image = checkpoint_images[i]
            image = np.reshape(image, [self.H_A_test,self.W_A_test,self.C_A_test])
            axes[i].imshow(image)
            axes[i].set_title(titles[i])
            axes[i].axis('off')
        fig.savefig("/mnt/d/OpenStylizer/data/batch_check_"+str(b)+".png")
        plt.close('all')
        return

# Command Line Argument Method
HEIGHT  = 64
WIDTH   = 64
CHANNEL = 3
EPOCHS = 2
#EPOCHS = 2
# our batch in this base is only a single image.
BATCH = 1
CHECKPOINT = 200


TRAIN_PATH_A = "/mnt/d/OpenStylizer/ukiyoe2photo/trainA"
TRAIN_PATH_B = "/mnt/d/OpenStylizer/ukiyoe2photo/trainB"
TEST_PATH_A = "/mnt/d/OpenStylizer/ukiyoe2photo/testA"
TEST_PATH_B = "/mnt/d/OpenStylizer/ukiyoe2photo/testB"

trainer = Trainer(height=HEIGHT,width=WIDTH,epochs =EPOCHS,\
                 batch=BATCH,\
                 checkpoint=CHECKPOINT,\
                 train_data_path_A=TRAIN_PATH_A,\
                 train_data_path_B=TRAIN_PATH_B,\
                 test_data_path_A=TEST_PATH_A,\
                 test_data_path_B=TEST_PATH_B,\
                 lambda_cycle=10.0,\
                 lambda_id=1.0)
trainer.train()