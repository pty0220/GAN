import os

from keras.layers import Layer, Input, Dropout, Conv3D, Activation, add, BatchNormalization, UpSampling3D, \
    Conv3DTranspose, Flatten, concatenate, MaxPooling3D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Model
from keras.engine.topology import Network
import cv2
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
import SimpleITK as sitk

import glob
import matplotlib.image as mpimage
import nibabel as nib
import numpy as np
import datetime
import time
import json
import csv
import sys


import keras.backend as K

import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class CycleGAN():
    def __init__(self):

        self.root_dir = 'F:/model/Medical-Image-Synthesis-multiview-Resnet_patch3D/'
        self.retrain_path = False#'20210105-121719-T1-CT_cropped_bias_P9_LR_0.0002_RL_9_DF_64_GF_32_RF_70'   #20201208-180043-T1-CT_cropped_bias_P9_LR_0.0002_RL_9_DF_64_GF_32_RF_70' previous good model
        self.retrain_epoch = False

        # Parse input arguments
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)  # Select GPU device±

        # data location
        self.volume_folder = 'T1-CT_cropped_bias_P9'  # 'T1-CT_cropped_bias_NS2' #'T1-CT_brain'

        self.patch = False
        self.augmentation = True
        self.padding_size = (140, 140, 140)
        # ======= Data ==========
        print('--- Caching data ---')

        # data = load_data_3D(subfolder=self.volume_folder)
        data = self.load_nii_3d(self.volume_folder, self.padding_size)

        self.patch_size = data["volume_size_A"][1]    # 64
        self.stride = data["volume_size_A"][1] //10  # 16
        self.patch_size_test = data["volume_size_A"][1]   # 64
        self.stride_test = data["volume_size_A"][1] // 10  # 16

        input_img = np.squeeze(data['trainA_images'][0])
        skull_m, mask = self.segmentation_mask(input_img, 250)
        self.brain_mask = mask

        if self.patch:
            data["volume_size_A"] = (self.patch_size, self.patch_size, self.patch_size)
            data["volume_size_B"] = (self.patch_size, self.patch_size, self.patch_size)
            data["volume_size_B"] = (self.patch_size, self.patch_size, self.patch_size)

            data["testA_images"] = self.extract_volume_patches(np.array(data["testA_images"]), self.patch_size,
                                                               self.stride)
            data["testB_images"] = self.extract_volume_patches(np.array(data["testB_images"]), self.patch_size,
                                                               self.stride)
            print('patch_size', self.patch_size)
            print('stride', self.stride)

        self.channels_A = data["nr_of_channels_A"]
        self.vol_shape_A = data["volume_size_A"] + (self.channels_A,)

        self.channels_B = data["nr_of_channels_B"]
        self.vol_shape_B = data["volume_size_B"] + (self.channels_B,)

        print('volume A shape: ', self.vol_shape_A)
        print('volume B shape: ', self.vol_shape_B)

        # if self.fixedsize:
        self.input_shape_A = self.vol_shape_A
        self.input_shape_B = self.vol_shape_B
        # else:
        #     self.input_shape_A = (None, None, None) + (self.channels_A,)
        #     self.input_shape_B = (None, None, None) + (self.channels_B,)
        #     print('Using unspecified input size')

        self.A_train = data["trainA_images"]
        self.B_train = data["trainB_images"]
        self.A_test = data["testA_images"]
        self.B_test = data["testB_images"]
        self.A_test_data = data["testA_imageDatas"]
        self.B_test_data = data["testB_imageDatas"]
        self.train_num = self.A_train.shape[0]
        self.train_idx = 0
        del data

        # ===== Model parameters ======
        # Training parameters
        self.lambda_AB = 10.0  # Cyclic loss weight A_2_B
        self.lambda_peak_diff = 5.0
        self.lambda_DSC_FWHM = 1.0
        self.lambda_adversarial = 1.0  # Weight for loss from discriminator guess on synthetic volumes
        self.lambda_gdl = 1.0
        # self.lambda_identity = 1.0
        self.learning_rate_D = 2e-4  # 2.4072578979390963e-05#0.00013173546847239113
        self.learning_rate_G = 2e-4  # 2.4072578979390963e-05#0.00013173546847239113
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.synthetic_pool_size = 50  # Size of volume pools used for training the discriminators
        self.beta_1 = 0.5  # Adam parameter
        self.beta_2 = 0.999  # Adam parameter
        self.batch_size = 1  # Number of volumes per batch
        self.epochs = 500#400  # choose multiples of 20 since the models are saved each 20th epoch

        self.save_models = True  # Save or not the generator and discriminator models
        self.save_models_interval = 1  # Number of epoch between saves of generator and discriminator models
        self.save_training_vol = True  # Save or not example training results or only tmp.png
        self.save_training_vol_interval = 1  # Number of epoch between saves of intermediate training results
        self.tmp_vol_update_frequency = 3  # Number of batches between updates of tmp.png
        self.tmp_img_z_A = self.vol_shape_A[2] // 2
        self.tmp_img_z_B = self.vol_shape_B[2] // 2

        # Architecture parameters
        self.use_instance_normalization = False  # Use instance normalization or batch normalization
        self.use_dropout = True  # Dropout in residual blocks
        self.use_bias = True  # Use bias
        self.use_linear_decay = True  # Linear decay of learning rate, for both discriminators and generators
        self.decay_epoch = 1000 #201 # The epoch where the linear decay of the learning rates start
        self.use_patchgan = True  # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_resize_convolution = False  # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.discriminator_sigmoid = True
        self.generator_residual_blocks = 9
        self.discriminator_layers = 5
        self.stride_2_layers = 3
        self.base_discirminator_filters = 64
        self.base_generator_filters = 32

        # Tweaks
        self.REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # ===== Architecture =====
        # Normalization
        if self.use_instance_normalization:
            self.normalization = InstanceNormalization
        else:
            self.normalization = BatchNormalization

        # Optimizers
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # Build discriminators
        D_B = self.build_discriminator(self.input_shape_B)

        # Define discriminator models
        volume_B = Input(shape=self.input_shape_B)
        guess_B = D_B(volume_B)
        self.D_B = Model(inputs=volume_B, outputs=guess_B, name='D_B_model')

        # Compile discriminator models
        loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic volumes
        self.D_B.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)

        # Use containers to make a static copy of discriminators, used when training the generators
        self.D_B_static = Network(inputs=volume_B, outputs=guess_B, name='D_B_static_model')

        # Do note update discriminator weights during generator training
        self.D_B_static.trainable = False

        # Build generators
        self.G_A2B = self.build_generator(self.input_shape_A, self.input_shape_B, name='G_A2B_model')

        from contextlib import redirect_stdout

        # Define full CycleGAN model, used for training the generators
        real_A = Input(shape=self.input_shape_A, name='real_A')
        #  real_B = Input(shape=self.input_shape_A, name='real_B')

        synthetic_B = self.G_A2B(real_A)
        #   identity_B = self.G_A2B(real_B)
        dB_guess_synthetic = self.D_B_static(synthetic_B)

        # Compile full CycleGAN model
        # model_outputs = [synthetic_B, dB_guess_synthetic, synthetic_B]
        # model_outputs = [synthetic_B, dB_guess_synthetic, synthetic_B, synthetic_B]
        model_outputs = [synthetic_B, dB_guess_synthetic,  synthetic_B, synthetic_B]  # ,   identity_B]
        # compile_losses = [self.mpd, self.lse, self.gdl]
        # compile_losses = [self.mpd, self.lse, self.gdl ,self.threshold_mae]
        compile_losses = [self.mpd, self.lse, self.gdl, self.peak_diff]  # ,self.mpd]

        #  compile_weights = [self.lambda_AB, self.lambda_adversarial, self.lambda_gdl]
        compile_weights = [self.lambda_AB, self.lambda_adversarial, self.lambda_gdl, self.lambda_peak_diff]  # , self.lambda_identity]

        self.G_model = Model(inputs=[real_A],  # real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)

        with open('generator.txt', 'w') as f:
            with redirect_stdout(f):
                self.G_A2B.summary()

        with open('discriminator.txt', 'w') as f:
            with redirect_stdout(f):
                D_B.summary()

        # ===== Folders and configuration =====
        if True:
            # Calculate discriminator receptive field
            nDiscFiltsStride2 = np.clip(self.stride_2_layers, 0, self.discriminator_layers - 1)
            nDiscFiltsStride1 = self.discriminator_layers - nDiscFiltsStride2

            receptField = int((1 + 3 * nDiscFiltsStride1) * 2 ** nDiscFiltsStride2 + 2 ** (nDiscFiltsStride2 + 1) - 2)

            # Generate tag
            self.tag = '_LR_{}_RL_{}_DF_{}_GF_{}_RF_{}'.format(self.learning_rate_D, self.generator_residual_blocks,
                                                               self.base_discirminator_filters,
                                                               self.base_generator_filters, receptField)
        else:
            self.tag = ''

        # if args.extra_tag:
        #     self.tag = self.tag + '_' + args.extra_tag

        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '-' + self.volume_folder + self.tag

        if self.retrain_path:
            self.date_time = self.retrain_path

            # Output folder for run data and volumes
        self.out_dir = os.path.join(self.root_dir, 'runs', self.date_time)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if self.save_training_vol:
            self.out_dir_volumes = os.path.join(self.out_dir, 'training_volumes')
            if not os.path.exists(self.out_dir_volumes):
                os.makedirs(self.out_dir_volumes)

        # Output folder for saved models
        if self.save_models:
            self.out_dir_models = os.path.join(self.out_dir, 'models')
            if not os.path.exists(self.out_dir_models):
                os.makedirs(self.out_dir_models)

        self.write_metadata_to_JSON()

        # Don't pre-allocate GPU memory; allocate as needed
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # K.tensorflow_backend.set_session(tf.Session(config=config))

        # ======= Initialize training ==========
        sys.stdout.flush()
        # plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)

        if self.retrain_epoch:
            self.retrain()
        else:
            self.train(epochs=self.epochs, batch_size=self.batch_size)

    # ===============================================================================
    # Architecture functions

    # Discriminator layers
    def ck(self, x, k, use_normalization, use_bias, stride):
        x = Conv3D(filters=k, kernel_size=4, strides=stride, padding='same', use_bias=use_bias)(x)
        if use_normalization:
            x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    # First generator layer
    def c7Ak(self, x, k):
        x = Conv3D(filters=k, kernel_size=7, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # Downsampling
    def dk(self, x, k):  # Should have reflection padding
        x = Conv3D(filters=k, kernel_size=3, strides=2, padding='same', use_bias=self.use_bias)(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # Residual block
    def Rk(self, x0):
        k = int(x0.shape[-1])

        # First layer
        x = ReflectionPadding3D((1, 1, 1))(x0)
        x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)

        if self.use_dropout:
            x = Dropout(0.5)(x)

        # Second layer
        x = ReflectionPadding3D((1, 1, 1))(x)
        x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        # Merge
        x = add([x, x0])

        return x

    # Upsampling
    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = UpSampling3D(size=(2, 2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding3D((1, 1, 1))(x)
            x = Conv3D(filters=k, kernel_size=3, strides=1, padding='valid', use_bias=self.use_bias)(x)
        else:
            x = Conv3DTranspose(filters=k, kernel_size=3, strides=2, padding='same', use_bias=self.use_bias)(
                x)  # this matches fractionally stided with stride 1/2
        x = self.normalization(axis=4, center=True, epsilon=1e-5)(x, training=True)
        x = Activation('relu')(x)
        return x

    # ===============================================================================
    # Models

    # PacthGAN
    def build_discriminator(self, vol_shape, name=None):
        # Input
        input_vol = Input(shape=vol_shape)

        # Layers 1-4
        for i in range(self.discriminator_layers - 1):
            stride = int(i < self.stride_2_layers) + 1
            if i == 0:
                x = self.ck(input_vol, self.base_discirminator_filters, False, True,
                            stride)  # Instance normalization is not used for this layer)
            else:
                x = self.ck(x, (2 ** (i - 1)) * self.base_discirminator_filters, True, self.use_bias, stride)

        # Layer 5: Output
        if self.use_patchgan:
            x = Conv3D(filters=1, kernel_size=4, strides=1, padding='same', use_bias=True)(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)

        if self.discriminator_sigmoid:
            x = Activation('sigmoid')(x)

        return Model(inputs=input_vol, outputs=x, name=name)

    # ResNet
    def build_generator(self, vol_shape_in, vol_shape_out, name=None):
        # Layer 1: Input
        input_vol = Input(shape=vol_shape_in)
        x = ReflectionPadding3D((3, 3, 3))(input_vol)
        x = self.c7Ak(x, self.base_generator_filters)

        # Layer 2-3: Downsampling
        x = self.dk(x, 2 * self.base_generator_filters)
        x = self.dk(x, 4 * self.base_generator_filters)

        # Layers 4-12: Residual blocks
        for _ in range(4, 4 + self.generator_residual_blocks):
            x = self.Rk(x)

        # Layer 13:14: Upsampling
        x = self.uk(x, 2 * self.base_generator_filters)
        x = self.uk(x, self.base_generator_filters)

        # Layer 15: Output
        x = ReflectionPadding3D((3, 3, 3))(x)
        x = Conv3D(filters=vol_shape_out[-1], kernel_size=7, strides=1, padding='valid', use_bias=True)(x)
        x = Activation('tanh')(x)

        return Model(inputs=input_vol, outputs=x, name=name)


    # ===============================================================================
    # Training

    # load prior model if exist
    def retrain(self):

        # load generator A to B
        path_to_weights = os.path.join(self.root_dir, 'runs', self.retrain_path, 'models',
                                       'G_A2B_model_epoch_{}.hdf5'.format(self.retrain_epoch))
        self.G_A2B.load_weights(path_to_weights)

        # load discriminator B
        path_to_weights = os.path.join(self.root_dir, 'runs', self.retrain_path, 'models',
                                       'D_B_model_epoch_{}.hdf5'.format(self.retrain_epoch))
        self.D_B.load_weights(path_to_weights)

        self.train(self.epochs, self.batch_size)

    # start training
    def train(self, epochs, batch_size=1):

        def run_training_batch():

            # ======= Discriminator training ======
            # Generate batch of synthetic volumes
            synthetic_volumes_B = self.G_A2B.predict(real_volumes_A)
            synthetic_volumes_B = synthetic_pool_B.query(synthetic_volumes_B)

            # Train discriminators on batch
            D_B_loss = []
            for _ in range(self.discriminator_iterations):
                D_B_loss_real = self.D_B.train_on_batch(x=real_volumes_B, y=ones)
                D_B_loss_synthetic = self.D_B.train_on_batch(x=synthetic_volumes_B, y=zeros)
                D_B_loss.append(D_B_loss_real + D_B_loss_synthetic)

            # ======= Generator training ==========
            # Reconstructed volumes need to match originals, discriminators need to predict ones
            target_data = [real_volumes_B, ones, real_volumes_B, real_volumes_B]  # , real_volumes_B]

            # Train generators on batch
            G_loss = []
            for _ in range(self.generator_iterations):
                G_loss.append(self.G_model.train_on_batch(
                    x=[real_volumes_A], y=target_data))

            # =====================================

            # Update learning rates
            if self.use_linear_decay and epoch >= self.decay_epoch:
                self.update_lr(self.D_B, decay_D, loop_index, epoch)
                self.update_lr(self.G_model, decay_G, loop_index, epoch)

            # Store training losses
            D_B_losses.append(D_B_loss[-1])

            G_AB_supervised_loss = G_loss[-1][1]
            G_AB_adversarial_loss = G_loss[-1][2]
            G_AB_gd_loss = G_loss[-1][3]
            #      G_AB_identity_loss = G_loss[-1][4]

            G_AB_supervised_losses.append(G_AB_supervised_loss)
            G_AB_adversarial_losses.append(G_AB_adversarial_loss)
            G_AB_gd_losses.append(G_AB_gd_loss)
            #       G_AB_identity_losses.append(G_AB_identity_loss)

            G_losses.append(G_loss[-1][0])

            # Print training status
            print('\n')
            print('Epoch ---------------------', epoch, '/', epochs)
            print('Loop index ----------------', loop_index + 1, '/', nr_vol_per_epoch)
            if self.discriminator_iterations > 1:
                print('  Discriminator losses:')
                for i in range(self.discriminator_iterations):
                    print('D_B_loss', D_B_loss[i])
            if self.generator_iterations > 1:
                print('  Generator losses:')
                for i in range(self.generator_iterations):
                    print('G_loss', G_loss[i])
            print('  Summary:')
            print('D_lr:', K.get_value(self.D_B.optimizer.lr))
            print('G_lr', K.get_value(self.G_model.optimizer.lr))
            print('D_B_loss: ', D_B_loss[-1])
            print('G_loss: ', G_loss[-1][0])
            print('G_AB_supervised_loss: ', G_AB_supervised_loss)
            self.print_ETA(start_time, epoch, nr_vol_per_epoch, loop_index)
            sys.stdout.flush()

            if loop_index % self.tmp_vol_update_frequency * self.batch_size == 0:
                # Save temporary images continously
                self.save_tmp_images(real_volumes_A[0], synthetic_volumes_B[0], real_volumes_B[0])

        # ======================================================================
        # Begin training
        # ======================================================================
        if self.save_training_vol:

            if not os.path.exists(os.path.join(self.out_dir_volumes, 'train_A')):
                os.makedirs(os.path.join(self.out_dir_volumes, 'train_A'))
                os.makedirs(os.path.join(self.out_dir_volumes, 'test_A'))

        D_B_losses = []

        G_AB_adversarial_losses = []
        G_AB_supervised_losses = []
        G_AB_gd_losses = []
        #  G_AB_identity_losses = []

        G_losses = []

        # volume pools used to update the discriminators
        synthetic_pool_B = volumePool(self.synthetic_pool_size)

        # Labels used for discriminator training
        # if self.fixedsize:
        label_shape = (self.batch_size,) + self.D_B.output_shape[1:]
        # else:
        #     label_shape = (batch_size,) + self.D_B.compute_output_shape((1,) + self.vol_shape_A)[1:]

        ones = np.ones(shape=label_shape) * self.REAL_LABEL
        zeros = ones * 0

        # Start stopwatch for ETAs
        start_time = time.time()
        timer_started = False

        if self.retrain_epoch:
            start = self.retrain_epoch + 1
        else:
            start = 1

        for epoch in range(start, epochs + 1):

            A_train = self.A_train
            B_train = self.B_train
            # A_train = self.A_train[self.train_idx][np.newaxis]
            # B_train = self.B_train[self.train_idx][np.newaxis]

            self.train_idx = self.train_idx + 1
            if self.train_idx == self.train_num:
                self.train_idx = 0

            # if opt['augmentation'] and epoch >= self.train_num:

            if self.augmentation:
                A_train, B_train = self.data_augmentation3D( A_train, B_train, epoch)

            if self.patch:
                # patch_size = data["volume_size_A"][1] // 4
                # strides = data["volume_size_A"][1] // 8

                A_train = self.extract_volume_patches(A_train, self.patch_size, self.stride)
                B_train = self.extract_volume_patches(B_train, self.patch_size, self.stride)

            # Linear learning rate decay
            if self.use_linear_decay:
                decay_D, decay_G = self.get_lr_linear_decay_rate(A_train)

            nr_train_vol = A_train.shape[0]
            nr_vol_per_epoch = int(np.ceil(nr_train_vol / self.batch_size) * self.batch_size)

            random_order = np.concatenate((np.random.permutation(nr_train_vol),
                                           np.random.randint(nr_train_vol, size=nr_vol_per_epoch - nr_train_vol)))

            # Train on volume batch
            for loop_index in range(0, nr_vol_per_epoch, self.batch_size):
                indices = random_order[loop_index:loop_index + self.batch_size]

                real_volumes_A = A_train[indices]
                real_volumes_B = B_train[indices]

                # Train on volume batch
                run_training_batch()

                # Start timer after first (slow) iteration has finished
                if not timer_started:
                    start_time = time.time()
                    timer_started = True

            # Save training volumes
            # if self.save_training_vol and epoch % self.save_training_vol_interval == 0:
            print('\n', '\n', '-------------------------Saving volumes for epoch', epoch, '-------------------------',
                  '\n', '\n')
            self.save_epoch_volumes(epoch)

            # Save model
            # if self.save_models and epoch % self.save_models_interval == 0:
            self.save_model(self.D_B, epoch)
            self.save_model(self.G_A2B, epoch)

            # Save training history
            training_history = {
                'DB_losses': D_B_losses,
                'G_AB_adversarial_losses': G_AB_adversarial_losses,
                'G_AB_supervised_losses': G_AB_supervised_losses,
                'G_losses': G_losses}
            self.write_loss_data_to_file(training_history)

    # ===============================================================================
    # Loss functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
        return loss

    def peak_diff(self, y_true, y_pred):
        y_true = tf.multiply(y_true, self.brain_mask)
        y_pred = tf.multiply(y_pred, self.brain_mask)
        loss = tf.abs(tf.keras.backend.max(y_pred) - tf.keras.backend.max(y_true))
        return loss

    def DSC_FWHM(self, y_true, y_pred):
        y_true = tf.multiply(y_true, self.brain_mask)
        y_pred = tf.multiply(y_pred, self.brain_mask)

        true_FWHM = tf.cast(y_true>tf.keras.backend.max(y_true)*0.5, tf.float32)
        pred_FWHM = tf.cast(y_pred>tf.keras.backend.max(y_pred)*0.5, tf.float32)

        overlab = tf.math.count_nonzero(tf.multiply(true_FWHM, pred_FWHM))

        true_FWHM = tf.math.count_nonzero(true_FWHM)
        pred_FWHM = tf.math.count_nonzero(pred_FWHM)

        DSC = overlab*2/(true_FWHM+pred_FWHM)
        loss = 1-DSC

        return loss


    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    # Mean p-norm error
    def mpd(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true) ** 1.5)
        return loss

    # Mean p-norm error
    def mae(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def gdl(self, y_true, y_pred):
        alpha = 2

        """
         Calculates the sum of GDL losses between the predicted and ground truth frames.
         This is the 3d version.
         @param gen_frames: The predicted frames at each scale.
         @param gt_frames: The ground truth frames at each scale
         @param alpha: The power to which each gradient term is raised.
         @return: The GDL loss for 3d. Dong
        """

        # calculate the loss for each scale
        scale_losses = []

        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        pos = tf.constant(np.identity(1), dtype=tf.float32)
        neg = -1 * pos

        baseFilter = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]# 2x1x1x1
        filter_x = tf.expand_dims(baseFilter, 1)  # [-1, 1] # 2x1x1x1x1
        filter_y = tf.expand_dims(baseFilter, 0)  # [-1, 1] # 1x2x1x1x1
        filter_z = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1] # 1x2x1x1
        filter_z = tf.expand_dims(filter_z, 0)  # [-1, 1] #1x1x2x1x1
        strides = [1, 1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv3d(y_pred, filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv3d(y_pred, filter_y, strides, padding=padding))
        gen_dz = tf.abs(tf.nn.conv3d(y_pred, filter_z, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv3d(y_true, filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv3d(y_true, filter_y, strides, padding=padding))
        gt_dz = tf.abs(tf.nn.conv3d(y_true, filter_z, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)
        grad_diff_z = tf.abs(gt_dz - gen_dz)

        loss = tf.reduce_mean((grad_diff_x ** alpha + grad_diff_y ** alpha + grad_diff_z ** alpha))

        # condense into one tensor and avg
        return loss

    # ===============================================================================
    # Learning rates

    def get_lr_linear_decay_rate(self, A_train):
        # Calculate decay rates
        # max_nr_volumes = max(len(self.A_train), len(self.B_train))

        nr_train_vol = A_train.shape[0]
        nr_batches_per_epoch = int(np.ceil(nr_train_vol / self.batch_size))

        updates_per_epoch = nr_batches_per_epoch
        nr_decay_updates = (self.epochs - self.decay_epoch + 1) * updates_per_epoch
        decay_D = self.learning_rate_D / nr_decay_updates
        decay_G = self.learning_rate_G / nr_decay_updates

        return decay_D, decay_G

    def update_lr(self, model, decay, loop_index, epoch):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

        if loop_index == 0:
            lr_path = '{}/test_A/epoch{}_lr_{}.npy'.format(self.out_dir_volumes, epoch, loop_index)
            f = open(lr_path, mode='wt', encoding='utf-8')
            f.write(str(new_lr))
            f.close()

    # ===============================================================================
    # Augmentation
    def data_augmentation3D(self, A_train, B_train, epoch):

        def get_slice_orientation(orientation, index, volume):

            if orientation == 'axial':  # axial
                slice = volume[index, :, :]
            if orientation == 'sagittal':  # sagittal
                slice = volume[:, :, index]
            if orientation == 'coronal':  # coronal
                slice = volume[:, index, :]

            return slice

        source = A_train
        target = B_train

        augmented_source_list = []
        augmented_target_list = []

        #     augmented_source_list.extend(list(source))
        #     augmented_target_list.extend(list(target))
        aug_num = 1
        for aug in range(aug_num):

            random = np.random.randint(1, 4)

            self.orientation = 'axial'
            if random == 1:
                self.orientation = 'axial'
            if random == 2:
                self.orientation = 'coronal'
            if random == 3:
                self.orientation = 'sagittal'

            print(self.orientation)
            # subject num
            for i in range(source.shape[0]):
                # prepare iterator

                # create image data augmentation generator
                datagen = ImageDataGenerator(rotation_range=4.5, zoom_range=0.3, shear_range=0.04, horizontal_flip=True)
                # shear_range=0.03
                if self.orientation == 'sagittal':  # data should not be flipped when sagittal
                    datagen = ImageDataGenerator(rotation_range=4.5, zoom_range=0.3, shear_range=0.04)

                source_volume = source[i, :, :, :, :]
                target_volume = target[i, :, :, :, :]

                augmented_source = np.full((source.shape[1], source.shape[1], source.shape[1]), -1.0)
                augmented_target = np.full((source.shape[1], source.shape[1], source.shape[1]), -1.0)

                for j in range(source.shape[1]):

                    source_slice = get_slice_orientation(self.orientation, j, source_volume)
                    target_slice = get_slice_orientation(self.orientation, j, target_volume)

                    # source_slice = source_slice.reshape((1,) + source_slice.shape)
                    # target_slice = target_slice.reshape((1,) + target_slice.shape)

                    source_slice = source_slice[np.newaxis, :, :, :]  # , np.newaxis]
                    target_slice = target_slice[np.newaxis, :, :, :]  # , np.newaxis]

                    itA = datagen.flow(source_slice, batch_size=1, seed=epoch)
                    itB = datagen.flow(target_slice, batch_size=1, seed=epoch)

                    # generate batch of images
                    batchA = itA.next()
                    batchB = itB.next()

                    # augmented_source.append(batchA[0, :, :, :])
                    # augmented_target.append(batchB[0, :, :, :])

                    imageA = batchA[0, :, :, 0]
                    imageB = batchB[0, :, :, 0]

                    # if epoch == 1 or epoch % 20 == 0 :
                    # save_image(imageA, os.path.join(self.root_dir, 'augmented_images', str(epoch), 'A'),
                    #           str(j) + '.png')
                    # save_image(imageB, os.path.join(self.root_dir, 'augmented_images', str(epoch), 'B'),
                    #           str(j) + '.png')
                    #
                    # aug_idx = aug_idx + 1

                    if self.orientation == 'axial':  # axial
                        augmented_source[j, :, :] = imageA
                        augmented_target[j, :, :] = imageB

                    if self.orientation == 'sagittal':  # sagittal
                        augmented_source[:, :, j] = imageA
                        augmented_target[:, :, j] = imageB

                    if self.orientation == 'coronal':  # coronal
                        augmented_source[:, j, :] = imageA
                        augmented_target[:, j, :] = imageB

                augmented_source_list.append(augmented_source[:, :, :, np.newaxis])
                augmented_target_list.append(augmented_target[:, :, :, np.newaxis])

        return (np.array(augmented_source_list), np.array(augmented_target_list))

    # ===============================================================================
    # Patch extraction & reconstruction
    def extract_volume_patches(self, img_arr, size=64, stride=16):

        if size % stride != 0:
            raise ValueError("size % stride must be equal 0")

        patches_list = []
        if len(img_arr.shape) == 5:
            for i in range(img_arr.shape[0]):
                temp = self.get_3Dpatches(img_arr[i, :, :, :, 0], size, stride)
                patches_list.extend(temp)
            np.stack(patches_list)

        if len(img_arr.shape) == 4:
            patches_list = self.get_3Dpatches(img_arr[:, :, :, 0], size, stride)

        return np.array(patches_list)

    def get_3Dpatches(self, img_arr, size=64, stride=16):

        if size % stride != 0:
            raise ValueError("size % stride must be equal 0")

        patches_list = []
        overlapping = 0
        if stride != size:
            overlapping = (size // stride) - 1

        i_max = img_arr.shape[0] // stride - overlapping
        j_max = img_arr.shape[1] // stride - overlapping
        k_max = img_arr.shape[2] // stride - overlapping
        # print(i_max)

        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    # print(i*stride, i*stride+size, j*stride, j*stride+size ,k*stride, k*stride+size)

                    patches_list.append(
                        img_arr[
                        i * stride: i * stride + size,
                        j * stride: j * stride + size,
                        k * stride: k * stride + size
                        ]
                    )

        patches = np.stack(patches_list)
        return patches[:, :, :, :, np.newaxis]

    def reconstruct_from_patches(self, img_arr, org_img_size, stride=None):
        """[summary]

        Args:
            img_arr (numpy.ndarray): [description]
            org_img_size (tuple): [description]
            stride ([type], optional): [description]. Defaults to None.
            size ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            numpy.ndarray: [description]
        """
        # check parameters
        # stride = self.stride_test
        if type(org_img_size) is not tuple:
            raise ValueError("org_image_size must be a tuple")

        if img_arr.ndim == 3:
            img_arr = np.expand_dims(img_arr, axis=0)

        # if size is None:
        size = img_arr.shape[1]
        print(size)
        if stride is None:
            stride = size

        i_max = (org_img_size[0] // stride) + 1 - (size // stride)
        j_max = (org_img_size[1] // stride) + 1 - (size // stride)
        k_max = (org_img_size[2] // stride) + 1 - (size // stride)

        kk = 0
        #     print(total_nm_images)
        #     for img_count in range(total_nm_images):
        img_bg = np.zeros(
            (org_img_size[0], org_img_size[1], org_img_size[2]), dtype=img_arr[0].dtype
        )
        div_bg = np.zeros(
            (org_img_size[0], org_img_size[1], org_img_size[2]), dtype=img_arr[0].dtype
        )

        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    img_bg[i * stride: i * stride + size, j * stride: j * stride + size,
                    k * stride: k * stride + size] += img_arr[kk, :, :, :]
                    div_bg[i * stride: i * stride + size, j * stride: j * stride + size,
                    k * stride: k * stride + size] = np.add(np.ones(img_arr[kk, :, :, :].shape),
                                                            div_bg[i * stride: i * stride + size,
                                                            j * stride: j * stride + size,
                                                            k * stride: k * stride + size], )
                    kk += 1

        recon = img_bg / div_bg

        return recon

    # ===============================================================================
    # create nii
    def slices2nii(self, slices, refer_img, name):

        volume = slices

        # print(volume.shape)

        new_img = sitk.GetImageFromArray(volume)
        new_img.SetSpacing(refer_img.GetSpacing())
        new_img.SetDirection(refer_img.GetDirection())
        new_img.SetOrigin(refer_img.GetOrigin())

        # directory = os.path.join(path)
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filename = name + '.nii.gz'
        sitk.WriteImage(new_img, name)

        return volume

    # ===============================================================================
    # Image and model output

    def save_tmp_images(self, real_volume_A, synthetic_volume_B, real_volume_B):
        try:

            # real_image_A = real_volume_A[:,:,self.tmp_img_z_A,:].transpose((2,0,1))
            # synthetic_image_B = synthetic_volume_B[:,:,self.tmp_img_z_B,:].transpose((2,0,1))
            # real_image_B = real_volume_B[:,:,self.tmp_img_z_B,:].transpose((2,0,1))

            real_image_A = real_volume_A[self.tmp_img_z_A, :, :, :]  # .transpose((2, 0, 1))
            synthetic_image_B = synthetic_volume_B[self.tmp_img_z_B, :, :, :]  # .transpose((2, 0, 1))
            real_image_B = real_volume_B[self.tmp_img_z_B, :, :, :]  # .transpose((2, 0, 1))

            length_A = self.vol_shape_A[0] * self.channels_A
            length_B = self.vol_shape_B[0] * self.channels_B
            width = self.vol_shape_A[1]
            length_max = np.maximum(length_A, length_B)

            real_images_A = np.zeros((length_max, width))
            real_images_A[:length_A, :] = np.reshape(real_image_A, (length_A, width))

            synthetic_images_B = np.zeros((length_max, width))
            synthetic_images_B[:length_B, :] = np.reshape(synthetic_image_B, (length_B, width))

            real_images_B = np.zeros((length_max, width))
            real_images_B[:length_B, :] = np.reshape(real_image_B, (length_B, width))

            save_path = '{}/tmp_A2B.png'.format(self.out_dir)
            self.join_and_save((real_images_A, synthetic_images_B, real_images_B), save_path)

        except:  # Ignore if file is open
            print('???')
            pass

    def join_and_save(self, images, save_path):
        # Join images
        image = np.hstack(images)

        # Save images
        mpimage.imsave(save_path, image, vmin=-1, vmax=1, cmap='gray')

    def save_epoch_volumes(self, epoch, num_saved_volumes=1):
        # Save test volumes
        # nr_test_vol = self.A_test.shape[0]
        #
        # rand_ind = np.random.randint(nr_test_vol)

        real_volume_A = self.A_test
        real_volume_B = self.B_test

        #  synthetic_volume_B = self.G_A2B.predict(real_volume_A)

        synthetic_volume_B = []
        for i in range(real_volume_A.shape[0]):
            synthetic_volume_B.extend(self.G_A2B.predict(real_volume_A[i][np.newaxis, :, :, :, :]))

        synthetic_volume_B = np.array(synthetic_volume_B)
        recon_real_volume_B = self.reconstruct_from_patches(real_volume_B[:, :, :, :, 0], (140, 140, 140), self.stride)
        recon_synthetic_volume_B = self.reconstruct_from_patches(synthetic_volume_B[:, :, :, :, 0], (140, 140, 140),
                                                                 self.stride)

        refer_data = self.A_test_data[0]

        # save_path_realA = '{}/test_A/epoch{}_realA.nii.gz'.format(self.out_dir_volumes, epoch)
        save_path_realB = '{}/test_A/epoch{}_realB.nii.gz'.format(self.out_dir_volumes, epoch)
        save_path_syntheticB = '{}/test_A/epoch{}_syntheticB.nii.gz'.format(self.out_dir_volumes, epoch)

        if epoch == 1 or epoch == self.retrain_epoch + 1:
            self.real_volume = self.slices2nii(recon_real_volume_B[:, :, :, np.newaxis], refer_data, save_path_realB)
        syn_volume = self.slices2nii(recon_synthetic_volume_B[:, :, :, np.newaxis], refer_data, save_path_syntheticB)



        # real_volume = self.real_volume[  height_s:  height_s +  height , :, :]
        # syn_volume = syn_volume[  height_s:  height_s +  height , :, :]

        # new_img = sitk.GetImageFromArray( syn_volume)
        # new_img.SetSpacing( refer_data.GetSpacing())
        # new_img.SetDirection( refer_data.GetDirection())
        # new_img.SetOrigin( refer_data.GetOrigin())

        # sitk.WriteImage(new_img, '{}/test_A/epoch{}_syntheticB_cutted.nii.gz'.format(self.out_dir_volumes, epoch))

        MAE = np.mean(np.abs(self.real_volume - syn_volume))
        np_path = '{}/test_A/epoch{}_mae.txt'.format(self.out_dir_volumes, epoch)

        # np.savetxt(np_path , MAE,  fmt='%1.3f', delimiter=',')

        f = open(np_path, mode='wt', encoding='utf-8')
        f.write(str(MAE))
        f.close()

    def save_model(self, model, epoch):
        weights_path = '{}/{}_epoch_{}.hdf5'.format(self.out_dir_models, model.name, epoch)
        model.save_weights(weights_path)

        model_path = '{}/{}_epoch_{}.json'.format(self.out_dir_models, model.name, epoch)
        model_json_string = model.to_json()
        with open(model_path, 'w') as outfile:
            outfile.write(model_json_string)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    #########################################################33
    # load
    def load_nii_3d(self, subfolder, image_size):

        dataset_path = os.path.join('data', subfolder)
        if not os.path.isdir(dataset_path):
            sys.exit(' Dataset ' + subfolder + ' does not exist')

        # volume paths
        trainA_path = os.path.join(dataset_path, 'trainA')
        trainB_path = os.path.join(dataset_path, 'trainB')
        testA_path = os.path.join(dataset_path, 'testA')
        testB_path = os.path.join(dataset_path, 'testB')

        # volume file names
        trainA_volume_names = sorted(glob.glob(os.path.join(trainA_path, '*.nii')))
        trainB_volume_names = sorted(glob.glob(os.path.join(trainB_path, '*.nii')))
        testA_volume_names = sorted(glob.glob(os.path.join(testA_path, '*.nii')))
        testB_volume_names = sorted(glob.glob(os.path.join(testB_path, '*.nii')))

        def split(lists):
            t = 0
            for x in lists:
                if t%2 == 0:
                    names = os.path.basename(x)
                t = t+1
            return names

        trainA_volume_names = [os.path.basename(x) for x in trainA_volume_names]
        trainB_volume_names = [os.path.basename(x) for x in trainB_volume_names]
        testA_volume_names = [os.path.basename(x) for x in testA_volume_names]
        testB_volume_names = [os.path.basename(x) for x in testB_volume_names]

        del trainA_volume_names[1:len(trainA_volume_names):2]
        del trainB_volume_names[1:len(trainB_volume_names):2]
        # del testA_volume_names[1:len(trainA_volume_names):2]
        # del testB_volume_names[1:len(trainA_volume_names):2]


        # Examine one volume to get size and number of channels
        vol_test_A = nib.load(os.path.join(trainA_path, trainA_volume_names[0]))
        vol_test_B = nib.load(os.path.join(trainB_path, trainB_volume_names[0]))

        if len(vol_test_A.shape) == 3:
            volume_size_A = vol_test_A.shape
            nr_of_channels_A = 1
        else:
            volume_size_A = vol_test_A.shape[0:-1]
            nr_of_channels_A = vol_test_A.shape[-1]

        if len(vol_test_B.shape) == 3:
            volume_size_B = vol_test_B.shape
            nr_of_channels_B = 1
        else:
            volume_size_B = vol_test_B.shape[0:-1]
            nr_of_channels_B = vol_test_B.shape[-1]

        testA, testB, testA_imageDatas, testB_imageDatas = self.create_dataset(testA_path, testA_volume_names,
                                                                               testB_path, testB_volume_names,
                                                                               image_size)
        trainA, trainB, trainA_imageDatas, trainB_imageDatas = self.create_dataset(trainA_path, trainA_volume_names,
                                                                                   trainB_path, trainB_volume_names,
                                                                                   image_size)

        return {
            "image_size": (trainA.shape[1], trainA.shape[2], trainA.shape[3]),

            "volume_size_A": (trainA.shape[1], trainA.shape[2], trainA.shape[3]),
            "volume_size_B": (trainB.shape[1], trainB.shape[2], trainB.shape[3]),
            "nr_of_channels": nr_of_channels_A,
            "nr_of_channels_A": nr_of_channels_A,
            "nr_of_channels_B": nr_of_channels_B,
            "trainA_images": trainA,
            "trainB_images": trainB,
            "testA_images": testA,
            "testB_images": testB,
            "trainA_image_names": trainA_volume_names,
            "trainB_image_names": trainB_volume_names,
            "testA_image_names": testA_volume_names,
            "testB_image_names": testB_volume_names,
            "testA_imageDatas": testA_imageDatas,
            "testB_imageDatas": testB_imageDatas
        }

    def create_dataset(self, A_path, A_nii_list, B_path, B_nii_list, image_size):
        source_image_datas = []
        target_image_datas = []

        All_source_slices = []
        All_target_slices = []

        for idx_ in range(len(A_nii_list)):

            A_dicom_name = A_nii_list[idx_]
            source_volume, source_slices_num, source_image_data = self.read_nii(A_path, A_dicom_name, True)

            B_dicom_name = B_nii_list[idx_]
            target_volume, target_slices_num, target_image_data = self.read_nii(B_path, B_dicom_name, False)

            #
            # save original image metadata
            #
            source_image_datas.append(source_image_data)
            target_image_datas.append(target_image_data)

            All_source_slices.append(source_volume[:, :, :, np.newaxis])
            All_target_slices.append(target_volume[:, :, :, np.newaxis])


        #
        #     #
        #     # segmented using MR mask (threshold - 200)
        #     #
        #     source_segmented_volume, target_segmented_volume = self.create_segmented_volume(source_volume,
        #                                                                                     target_volume)
        #     print('Denoised')
        #
        #     #
        #     # clip source image range[0,2500]
        #     #
        #     source_clipped_volume = np.where(source_segmented_volume <= 0, 0, source_segmented_volume)
        #     source_clipped_volume = np.where(source_clipped_volume >= 2500, 2500, source_clipped_volume)
        #
        #     #
        #     # clip target image range[-1000HU, 3000HU]
        #     #
        #     target_clipped_volume = np.where(target_segmented_volume <= -1000, -1000, target_segmented_volume)
        #     target_clipped_volume = np.where(target_clipped_volume >= 3000, 3000, target_clipped_volume)
        #     print('Intensity Clipped')
        #
        #     #
        #     # padding to 256 x 256 x 256
        #     #
        #     width = image_size[0]
        #     height = image_size[1]
        #     depth = image_size[2]
        #
        #     padded_source_volume = np.full((height, width, depth), 0)
        #     padded_target_volume = np.full((height, width, depth), -1000)
        #
        #     if source_clipped_volume.shape[0] < image_size[0] or source_clipped_volume.shape[1] < image_size[1] or \
        #             source_clipped_volume.shape[2] < image_size[2]:
        #         # create new image of desired size
        #
        #         org_width = source_clipped_volume.shape[0]
        #         org_height = source_clipped_volume.shape[1]
        #         org_depth = source_clipped_volume.shape[2]
        #
        #         # compute center offset
        #         source_xx = (width - org_width) // 2
        #         source_yy = (height - org_height) // 2
        #         source_zz = (depth - org_depth) // 2
        #
        #         # copy img image into center of result image
        #         padded_source_volume[source_xx:source_xx + org_width, source_yy:source_yy + org_height,
        #         source_zz:source_zz + org_depth] = source_clipped_volume
        #         padded_target_volume[source_xx:source_xx + org_width, source_yy:source_yy + org_height,
        #         source_zz:source_zz + org_depth] = target_clipped_volume
        #
        #         # copy img image into upper part of result image
        #         # padded_source_slice[0:org_height, source_xx:source_xx + org_width] = source_slice
        #         # padded_target_slice[0:org_height, source_xx:source_xx + org_width] = target_slice
        #
        #     All_source_slices.append(np.array(padded_source_volume[:, :, :, np.newaxis]))
        #     All_target_slices.append(np.array(padded_target_volume[:, :, :, np.newaxis]))
        #
        #     print('Size Padding', np.array(padded_source_volume).shape)
        #
        # All_source_slices = np.array(All_source_slices)
        # All_target_slices = np.array(All_target_slices)
        #
        # print('shape', All_source_slices.shape, All_target_slices.shape)
        #
        # a = np.min(All_source_slices)
        # b = np.max(All_source_slices)
        # c = np.min(All_target_slices)
        # d = np.max(All_target_slices)
        #
        # print('source min/max', a, b, 'target min/max', c, d)
        #
        # #
        # # the intensity is scaled range of [-1, 1]
        # #
        # All_source_slices = All_source_slices - np.min(All_source_slices)
        # All_source_slices = All_source_slices / (np.max(All_source_slices) - np.min(All_source_slices))
        # All_source_slices = All_source_slices * 2 - 1
        #
        # All_target_slices = All_target_slices - np.min(All_target_slices)
        # All_target_slices = All_target_slices / (np.max(All_target_slices) - np.min(All_target_slices))
        # All_target_slices = All_target_slices * 2 - 1

        All_source_slices = np.array(All_source_slices)
        All_target_slices = np.array(All_target_slices)

        return_val = (All_source_slices, All_target_slices, source_image_datas, target_image_datas)

        a = np.min(All_source_slices)
        b = np.max(All_source_slices)
        c = np.min(All_target_slices)
        d = np.max(All_target_slices)

        print('source normalized min/max', a, b, 'target normalized min/max', c, d)

        return return_val


    def segmentation_mask(self, arr, threshold):

        arr = arr*3000
        ret, img_th = cv2.threshold(arr, threshold, 1, 0)

        # remain largest region
        label_im, nb_labels = ndimage.label(img_th)
        sizes = ndimage.sum(img_th, label_im, range(nb_labels + 1))
        mask = sizes == np.max(sizes)
        label_img = mask[label_im]

        # seg_arr = np.multiply(arr, label_img)

        if threshold == -100:
            kernel = np.ones((3, 3), np.uint8)

            open_volume = cv2.morphologyEx(np.array(label_img).astype(np.uint8), cv2.MORPH_OPEN, kernel)

            label_im, nb_labels = ndimage.label(open_volume)
            sizes = ndimage.sum(img_th, label_im, range(nb_labels + 1))
            mask = sizes == np.max(sizes)
            label_img = mask[label_im]
            open_volume = np.array(label_img).astype(np.uint8)

            closed_volume = cv2.morphologyEx(open_volume, cv2.MORPH_CLOSE, kernel)
            # closed_volume = ndimage.binary_erosion(closed_volume)
            #         temp =  closed_volume
            temp = np.zeros_like(label_img)

            for idx in range(closed_volume.shape[0]):
                closed_slice = closed_volume[idx, :, :]
                mask_slice = ndimage.morphology.binary_fill_holes(closed_slice)
                temp[idx, :, :] = mask_slice
            #   mpimg.imsave(os.path.join('E:/2020/KHM_synt/seg', str(threshold)+'_'+str(idx)+'.png'), mask_slice, vmin=-1, vmax=1, cmap='gray')
            temp = ndimage.binary_erosion(temp)

        if threshold == 250:
            kernel = np.ones((3, 3), np.uint8)
            open_volume = cv2.morphologyEx(np.array(label_img).astype(np.uint8), cv2.MORPH_OPEN, kernel)

            label_im, nb_labels = ndimage.label(open_volume)
            sizes = ndimage.sum(img_th, label_im, range(nb_labels + 1))
            mask = sizes == np.max(sizes)
            label_img = mask[label_im]
            open_volume = np.array(label_img).astype(np.uint8)

            closed_volume = cv2.morphologyEx(open_volume, cv2.MORPH_CLOSE, kernel)
            temp = closed_volume

        skull_volume = np.array(temp)

        # extract brain region only
        mask = np.copy(skull_volume)

        mask[:10, :, :] = 1
        mask[-10:, :, :] = 1
        mask[:, :, :10] = 1
        mask[:, :, -10:] = 1
        mask = mask - 1
        mask = -mask
        brain_seg = np.zeros_like(mask)

        for i in range(140):
            ret, makers = cv2.connectedComponents(mask[:, i, :])
            brain_seg[:, i, :] = makers

        brain_seg[brain_seg <= 1] = 0
        brain_seg[brain_seg > 0] = 1

        return skull_volume, brain_seg



    def create_segmented_volume(self, source_volume, target_volume):

        # for idx in range(source_volume.shape[1]):
        #     source_slice = source_volume[:, idx, :]
        #     target_slice = target_volume[:, idx, :]
        #
        #     save_image(source_slice, os.path.join(opt['root_dir'], 'original_images', 'A'),
        #                'slice_' + str(idx) + '.png')
        #     save_image(target_slice, os.path.join(opt['root_dir'], 'original_images', 'B'),
        #                'slice_' + str(idx) + '.png')

        ret, img_th = cv2.threshold(source_volume, 100, 1, 0)

        # Get largest continuous image
        label_im, nb_labels = ndimage.label(img_th)
        sizes = ndimage.sum(img_th, label_im, range(nb_labels + 1))
        mask = sizes == np.max(sizes)
        binary_img = mask[label_im]

        # binary_img  = ndimage.binary_dilation(binary_img )
        # closed_volume = ndimage.binary_closing(binary_img).astype(np.int)

        kernel = np.ones((10, 10), np.uint8)

        # close holes & erosion
        closed_volume = cv2.morphologyEx(np.array(binary_img).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        closed_volume = ndimage.binary_erosion(closed_volume)

        # binary fill holes
        temp = np.zeros_like(source_volume)
        for idx in range(closed_volume.shape[0]):
            closed_slice = closed_volume[idx, :, :]
            mask_slice = ndimage.morphology.binary_fill_holes(closed_slice)
            temp[idx, :, :] = mask_slice
        temp = ndimage.binary_dilation(temp)
        mask_volume = np.array(temp)

        # fill background
        source_denoised_volume = np.where(mask_volume, source_volume, np.zeros_like(source_volume))
        target_denoised_volume = np.where(mask_volume, target_volume,
                                          np.ones_like(target_volume) * -1000)  # np.zeros_like(target_volume)

        # check denoised image
        # for idx in range(mask_volume.shape[0]):
        #     source_slice = source_denoised_volume[idx, :, :]
        #     target_slice = target_denoised_volume[idx, :, :]
        #     mask_slice = mask_volume[idx, :, :]
        #     save_image(mask_slice, os.path.join(opt['root_dir'], 'mask'), 'slice_' + str(idx) + '.png')
        #
        #     save_image(source_slice, os.path.join(opt['root_dir'], 'denoised_images', 'A'),
        #                'slice_' + str(idx) + '.png')
        #     save_image(target_slice, os.path.join(opt['root_dir'], 'denoised_images', 'B'),
        #                'slice_' + str(idx) + '.png')

        return (source_denoised_volume, target_denoised_volume)

    def read_nii(self, path, file_name, source=None):

        inputImageFileName = os.path.join(path, file_name)

        print(file_name)

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NiftiImageIO")
        reader.SetFileName(inputImageFileName)
        image = reader.Execute()

        numpyImage = sitk.GetArrayFromImage(image)
        slices = image.GetDepth()

        return (numpyImage, slices, image)

    # ===============================================================================
    # Other output

    def print_ETA(self, start_time, epoch, nr_vol_per_epoch, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * nr_vol_per_epoch + loop_index) / self.batch_size
        iterations_total = self.epochs * nr_vol_per_epoch / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Elapsed time', passed_time_string, ': ETA in', eta_string)

    def write_loss_data_to_file(self, history):
        keys = sorted(history.keys())
        with open(self.root_dir + '/runs/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def write_metadata_to_JSON(self):
        # Save metadata
        metadata = {
            'vol shape_A: height,width,depth,channels': self.vol_shape_A,
            'vol shape_B: height,width,depth,channels': self.vol_shape_B,
            'input shape_A: height,width,channels': self.input_shape_A,
            'input shape_B: height,width,channels': self.input_shape_B,
            'batch size': self.batch_size,
            'save training vol interval': self.save_training_vol_interval,
            'normalization function': str(self.normalization),
            'lambda_AB': self.lambda_AB,
            'lambda_adversarial': self.lambda_adversarial,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of A train examples': len(self.A_train),
            'number of B train examples': len(self.B_train),
            'number of A test examples': len(self.A_test),
            'number of B test examples': len(self.B_test),
            'discriminator sigmoid': self.discriminator_sigmoid,
            'resize convolution': self.use_resize_convolution,
            'tag': self.tag,
            'volume_folder': self.volume_folder
        }

        with open('{}/metadata.json'.format(self.out_dir), 'w') as outfile:
            json.dump(metadata, outfile, sort_keys=True)


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        size_increase = [0, 2 * self.padding[0], 2 * self.padding[1], 2 * self.padding[2], 0]
        output_shape = list(s)

        for i in range(len(s)):
            if output_shape[i] == None:
                continue
            output_shape[i] += size_increase[i]

        return tuple(output_shape)

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [d_pad, d_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class volumePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_vols = 0
            self.volumes = []

    def query(self, volumes):
        if self.pool_size == 0:
            return volumes
        return_volumes = []
        for volume in volumes:
            if len(volume.shape) == 4:
                volume = volume[np.newaxis, :, :, :, :]

            if self.num_vols < self.pool_size:  # fill up the volume pool
                self.num_vols = self.num_vols + 1
                if len(self.volumes) == 0:
                    self.volumes = volume
                else:
                    self.volumes = np.vstack((self.volumes, volume))

                if len(return_volumes) == 0:
                    return_volumes = volume
                else:
                    return_volumes = np.vstack((return_volumes, volume))

            else:  # 50% chance that we replace an old synthetic volume
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.volumes[random_id, :, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :, :]
                    self.volumes[random_id, :, :, :, :] = volume[0, :, :, :, :]
                    if len(return_volumes) == 0:
                        return_volumes = tmp
                    else:
                        return_volumes = np.vstack((return_volumes, tmp))
                else:
                    if len(return_volumes) == 0:
                        return_volumes = volume
                    else:
                        return_volumes = np.vstack((return_volumes, volume))

        return return_volumes


if __name__ == '__main__':
    CycleGAN()