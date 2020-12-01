import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Input,ZeroPadding2D,Add,Dense,Activation,BatchNormalization,MaxPool2D,UpSampling2D,PReLU,Conv2DTranspose
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import json
AUTOTUNE=tf.data.experimental.AUTOTUNE


def create_model(img_size=(16,16,3),heatmap_size=(128,128,19)):
    input_heatmap = Input(shape=heatmap_size)
    input_img = Input(shape=img_size)
    
    output_coarse = generator(input_img)
    bottleneck = bottleneck_block
    output_HG = create_hourglass_network(input_heatmap, 1, 128, heatmap_size, bottleneck)
    output_img = Add()([output_coarse, output_HG])

    model = Model(inputs=[input_heatmap,input_img], outputs=[output_coarse,output_img])
    model.compile(optimizer=Adam(5e-4), loss=[mean_squared_error,mean_squared_error], metrics=["accuracy"])

    return model

def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization()(model)
        
    model = Add()([gen, model])
    return model

def generator(gen_input):

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(gen_input)
    model = PReLU()(model)

    for index in range(12):
        model= res_block_gen(model, 3, 64, 1)
    
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same")(model)
    
    for index in range(3):
        model = res_block_gen(model, 3, 64, 1)
        
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same")(model)
    
    for index in range(3):
        model = res_block_gen(model, 3, 64, 1)
    
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    model = Conv2DTranspose(filters = 64, kernel_size = 4, strides = 2, padding = "same")(model)
    
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Activation('relu')(model)
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Activation('relu')(model)
    model = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = "same")(model)

    model = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = "same")(model)

    return model

def create_hourglass_network(input, num_stacks, num_channels, inres, bottleneck):

    head_next_stage = input

    for i in range(num_stacks):
        head_next_stage = hourglass_module(head_next_stage, num_channels, bottleneck, i)
    output = Conv2D(3, kernel_size=(1, 1), padding='same')(head_next_stage)
    
    return output

def hourglass_module(bottom, num_channels, bottleneck, hgid):
    left_features = create_left_half_blocks(bottom, bottleneck, hgid, num_channels)

    rf1 = create_right_half_blocks(left_features, bottleneck, hgid, num_channels)

    head_next_stage = create_heads(bottom, rf1, hgid, num_channels)

    return head_next_stage

def bottleneck_block(bottom, num_out_channels, block_name='a'):
    if K.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same')(bottom)

    _x = Conv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same')(bottom)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same')(_x)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same')(_x)
    _x = BatchNormalization()(_x)
    _x = Add()([_skip, _x])

    return _x

def create_front_module(input, num_channels, bottleneck):

    _x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1_x1')(
        input)
    _x = BatchNormalization()(_x)

    _x = bottleneck(_x, num_channels // 2)
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)

    _x = bottleneck(_x, num_channels // 2)
    _x = bottleneck(_x, num_channels)

    return _x


def create_left_half_blocks(bottom, bottleneck, hglayer, num_channels):

    hgname = 'hg' + str(hglayer)

    f1 = bottleneck(bottom, num_channels)
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

    f2 = bottleneck(_x, num_channels)
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

    f4 = bottleneck(_x, num_channels)
    _x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

    f8 = bottleneck(_x, num_channels)

    return (f1, f2, f4, f8)


def connect_left_to_right(left, right, bottleneck, num_channels):

    _xleft = bottleneck(left, num_channels)
    _xright = UpSampling2D()(right)
    add = Add()([_xleft, _xright])
    out = bottleneck(add, num_channels)
    return out


def bottom_layer(lf8, bottleneck, hgid, num_channels):
    
    lf8_connect = bottleneck(lf8, num_channels)

    _x = bottleneck(lf8, num_channels)
    _x = bottleneck(_x, num_channels)
    _x = bottleneck(_x, num_channels)

    rf8 = Add()([_x, lf8_connect])

    return rf8


def create_right_half_blocks(leftfeatures, bottleneck, hglayer, num_channels):
    lf1, lf2, lf4, lf8 = leftfeatures

    rf8 = bottom_layer(lf8, bottleneck, hglayer, num_channels)

    rf4 = connect_left_to_right(lf4, rf8, bottleneck, num_channels)

    rf2 = connect_left_to_right(lf2, rf4, bottleneck, num_channels)

    rf1 = connect_left_to_right(lf1, rf2, bottleneck, num_channels)

    return rf1


def create_heads(prelayerfeatures, rf1, hgid, num_channels):
    head = Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same')(rf1)
    head = BatchNormalization()(head)

    head = Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same')(head)
    return head

def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 1.
    return rlt.astype(np.float32)

def calc_psnr(img1, img2):

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calc_ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

if __name__ == '__main__':
  
    print('Test Start!')

    option = json.loads(open('option.json').read())
    if option['use_gpu']:
        os.environ["CUDA_VISIBLE_DEVICES"] = option['gpu_no']
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


    lr_size = (option['LR_size'], option['LR_size'], 3)
    shape_size = (option['HR_size'], option['HR_size'], option['shape_priors_channel'])
    model = create_model(lr_size, shape_size)
    model.load_weights(option['pretrained_model_path'])

    lr_path = option['LR_image_path']
    hr_path = option['HR_image_path']
    sr_path = option['SR_image_path']
    shape_path = option['shape_priors_path']

    dataset = dict()

    for root,_,files in os.walk(lr_path):
        for file in files:
            data =dict()
            img = plt.imread(os.path.join(root, file))
            data['lr'] = img
            data['name'] = file.split('.')[0]
            dataset[file.split('.')[0]] = data
    for root,_,files in os.walk(hr_path):
        for file in files:
            img = plt.imread(os.path.join(root, file))
            dataset[file.split('.')[0]]['hr'] = img
    for root,_,files in os.walk(shape_path):
        for file in files:
            shape = np.load(os.path.join(root, file))
            dataset[file.split('.')[0]]['shape'] = shape

    print(str(len(dataset)) + ' files to test!')
        
    output = []
    for sample in dataset.values():
        data = dict()
        data['file_name'] = sample['name']
        lr = np.array(sample['lr']).reshape(1,option['LR_size'],option['LR_size'],3)
        hr = np.array(sample['hr'])
        if option['rgb_range'] == 1:
            lr*=255
            hr*=255
        shape = sample['shape'].reshape(1,option['HR_size'],option['HR_size'],option['shape_priors_channel'])
        coarse, refine = model.predict([shape,lr])
        sr = refine.reshape(option['HR_size'],option['HR_size'],3)
        cv2.imwrite(sr_path+'/'+sample['name']+'.png', np.array(sr)[:, :, ::-1])
    
        psnr = calc_psnr(rgb2ycbcr(np.array(hr)/255.), rgb2ycbcr(np.array(sr)/255.))
        ssim = calc_ssim(rgb2ycbcr(np.array(hr)/255.), rgb2ycbcr(np.array(sr)/255.))
        data['psnr'] = psnr
        data['ssim'] = ssim
        output.append(data)
    
    output = json.dumps(output, indent=4)
    open(sr_path+'/result.json','w').write(output)

    print('Finish! Results saved in '+ sr_path+'/result.json')