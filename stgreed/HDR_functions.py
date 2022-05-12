import numpy as np
import scipy


def fread(fid, nelements, dtype):
     if dtype is np.str:
         dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
     else:
         dt = dtype

     data_array = np.fromfile(fid, dt, nelements)
     data_array.shape = (nelements, 1)

     return data_array


def hdr_yuv_read(file_object,frame_num,height,width):
    file_object.seek(frame_num*height*width*3)
    y1 = fread(file_object,height*width,np.uint16)
    u1 = fread(file_object,height*width//4,np.uint16)
    v1 = fread(file_object,height*width//4,np.uint16)
    y = np.reshape(y1,(height,width))
    u = np.reshape(u1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)
    v = np.reshape(v1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)
    return y,u,v

def yuv_read(filename,frame_num,height,width):
    file_object = open(filename)
    file_object.seek(frame_num*height*width*1.5)
    y1 = fread(file_object,height*width,np.uint8)
    u1 = fread(file_object,height*width//4,np.uint8)
    v1 = fread(file_object,height*width//4,np.uint8)
    y = np.reshape(y1,(height,width))
    u = np.reshape(u1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)
    v = np.reshape(v1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)
    return y,u,v


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


def local_exp(image,par,patch_size):
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)

    if image.max()>1:
        image=image/image.max()
    
    avg_window = gen_gauss_window(patch_size//2, 7.0/6.0)
    mu_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode='constant')
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode='constant')
    y = np.exp(par*(image - mu_image))
    return y
    

def local_m_exp(image,par,patch_size = 31):
    if image.max()>1:
        image=image/image.max()
    maxY = scipy.ndimage.maximum_filter(image,size=(patch_size,patch_size))
    minY = scipy.ndimage.minimum_filter(image,size=(patch_size,patch_size))
    image = -4+(image-minY)* 8/(1e-3+maxY-minY)
    Y_transform =  np.exp(np.abs(image)**par)-1
    Y_transform[image<0] = -Y_transform[image<0]
    return Y_transform

def global_m_exp(Y,delta):
    if Y.max()>1:
        image=Y/Y.max()
    Y = -4+(Y-np.amin(Y))* 8/(1e-3+np.amax(Y)-np.amin(Y))
    Y_transform =  np.exp(np.abs(Y)**delta)-1
    Y_transform[Y<0] = -Y_transform[Y<0]
    return Y_transform


def global_exp(image,par):

    assert len(np.shape(image)) == 2
    if image.max()>1:
        image=image/image.max()
    avg = np.average(image)
    y = np.exp(par*(image-avg))
    return y
