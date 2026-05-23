
import numpy as np
import cv2
from scipy import ndimage

def Id(x):
    return x
##
## differential forward and backward operators
##

# gradient
def D(x):

    if x.ndim == 1:
        grad = np.concatenate((x[1:] - x[:-1], [0]))/2.

    elif x.ndim == 2:
        sz = x.shape
        Dx_im = np.concatenate((  x[:,1:] - x[:,:-1] , np.zeros((sz[0],1)) ), axis=1)/ 2.
        Dy_im = np.concatenate((  x[1:,:] - x[:-1,:] , np.zeros((1,sz[1])) ), axis=0)/ 2.
        
        grad = np.array([Dx_im,Dy_im])
    return grad

def Dt(x):
    if x.ndim == 1:
        div = - np.concatenate(([x[0]], x[1:-1] - x[:-2], [-x[-2]])) /2.

    elif x.ndim == 3:
        x1 = x[0]
        x2 = x[1]
        div = - np.concatenate((x1[:,[0]], x1[:,1:-1] - x1[:,:-2], -x1[:,[-2]]), axis=1) /2. \
              - np.concatenate((x2[[0],:], x2[1:-1,:] - x2[:-2,:], -x2[[-2],:]), axis=0) /2.
    return div

# laplacian
def L(x):
    if x.ndim == 1:
        ker = np.array([1, -2, 1])
        #lap = np.convolve(x,ker,'same')
        lap = ndimage.convolve1d(x,ker,mode='nearest')
    elif x.ndim == 2:
        ker = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])    # V4
        #ker = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])    # V8
        lap = ndimage.convolve(x,ker,mode='nearest')
    return lap

def Lt(x):
    if x.ndim == 1:
        ker = np.array([1, -2, 1])
        #lap = np.correlate(x,ker,'same')
        lap = ndimage.correlate1d(x,ker,mode='nearest')
    elif x.ndim == 2:
        ker = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])    # V4
        #ker = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])    # V8
        lap = ndimage.correlate(x,ker,mode='nearest')
    return lap



## bluring operator
def generatePSF(dim,blurtype,kernelSize):
    # compute kernel
    if blurtype == 'none':
        h = np.array([1.])

    elif blurtype == 'gaussian':
        std = kernelSize/6
        x = np.linspace(-(kernelSize-1)/2, (kernelSize-1)/2, kernelSize)
        arg = -x**2/(2*std**2)
        h = np.exp(arg)

    elif blurtype == 'uniform':
        h = np.ones(kernelSize)
    
    # kernel normalization
    h = h/sum(h)

    # return kernel
    if dim == '1D':
        ker = h
    elif dim == '2D':
        ker = np.tensordot(h,h, axes=0)
    
    return ker 

def A(x,psf):
    if x.ndim == 1:
        b = np.convolve(x,psf,'same')
    elif x.ndim == 2:
        b = ndimage.convolve(x,psf,mode='nearest')

    return b

def At(x,psf):
    if x.ndim == 1:
        b = np.correlate(x,psf,'same')
    elif x.ndim == 2:
        b = ndimage.correlate(x,psf,mode='nearest')

    return b


##  Lipschitz constant of operators
def opNorm(op,opt,dim):
    def T(x):
        return opt(op(x))

    if dim == '1D':  
        xn = np.random.standard_normal((64))
    elif dim == '2D':
        xn = np.random.standard_normal((64,64))

    xnn = xn

    n = np.zeros((100,),float)
    n[1] = 1
    tol  = 1e-4
    rhon = n[1]+2*tol

    k = 1
    while abs(n[k]-rhon)/n[k] >= tol:
        xn  = T(xnn)
        xnn = T(xn)

        rhon   = n[k]
        n[k+1] = np.sum(xnn**2)/np.sum(xn**2)
   
        k = k+1

    N = n[k-1] + 1e-16
    return 1.01* N**(.25)              # sqrt(L) gives |||T|||=|||D'D||| ie |||D|||^2