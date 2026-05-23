import numpy as np
from matplotlib import pyplot as plt
from operators import *
import cv2
from skimage.metrics import structural_similarity as ssim

def mse(image1, image2):
    """
    Calculate the Mean Squared Error (MSE) between two images.
    
    Args:
        image1 (numpy.ndarray): First image.
        image2 (numpy.ndarray): Second image.
        
    Returns:
        float: MSE value.
    """
    return np.mean((image1 - image2) ** 2)

def snr(image, reference):
    """
    Calculate the Signal to Noise Ratio (SNR) between an image and a reference.
    
    Args:
        image (numpy.ndarray): Image whose SNR is calculated.
        reference (numpy.ndarray): Reference image.
        
    Returns:
        float: SNR value in dB.
    """
    mse_value = mse(image, reference)
    power_signal = np.sum(reference ** 2)
    snr_value = 10 * np.log10(power_signal / mse_value)
    return snr_value

def ssim_score(image, reference, data_range=256):
    """
    Calculate the Structural SIMilarity (SSIM) between an image and a reference.
    
    Args:
        image (numpy.ndarray): Image whose SSIM is calculated.
        reference (numpy.ndarray): Reference image.
        data_range (float or None, optional): Range of possible values for the data. If None, the data range is determined by the data type of the image. Default is None.
        
    Returns:
        float: SSIM value.
    """
    return ssim(image, reference, data_range=data_range, multichannel=True)

def TV(image, coeffLambda, mu, Niter):
    """
    Perform Total Variation (TV) denoising on an image.
    
    Args:
        coeffLambda (float): Coefficient for the regularization term.
        mu (float): Regularization parameter.
        image (str): Path to the input image.
        
    Returns:
        numpy.ndarray: Denoised image.
    """
    I = cv2.imread(image, 0)
    N = 9  # dimension
    h = generatePSF('2D', 'gaussian', N)
                
    def H(x):
        return A(x, h)

    def Ht(x):
        return At(x, h)

    bruit = np.random.standard_normal(I.shape)
    z = H(I) + bruit

    def Norme_1(x):
        return np.sum(np.abs(x))

    def Norme_2(x):
        return np.sqrt(np.dot(np.transpose(x), x))

    def Gradient_1(x, z, yk, mu):
        return 2 * Ht(H(x) - z) - mu * Dt(yk - D(x))

    def Gradient_2(y, xkplus1, mu):
        return mu * (y - D(xkplus1))

    def prox_g(y, coeff, gamma):
        xGamma = np.ones(y.shape) * gamma * coeff
        return np.maximum(np.minimum(np.zeros(y.shape), y + xGamma), y - xGamma)


    # Step definitions
    a = 2 / 4

    LDf2 = mu
    gamma2 = a / (LDf2)

    LDf1 = 2 * opNorm(H, Ht, '2D') ** 2 + mu * opNorm(D, Dt, '2D') ** 2
    gamma1 = a / LDf1

    # Initial values (pseudo-random)
    y0 = np.ones((2, I.shape[0], I.shape[1]))
    x0 = np.ones(I.shape) * 20

    X = [x0]
    Y = [y0]

    for k in range(Niter):
        # Calculate xk+1
        X.append(X[k] - gamma1 * Gradient_1(X[k], z, Y[k], mu))

        # Calculate yk+1
        Y.append(Y[k] - gamma2 * Gradient_2(Y[k], X[k + 1], mu))
        Y[k + 1] = prox_g(Y[k + 1], coeffLambda, gamma2)

    Iapprox = X[-1]
    return [I, Iapprox, z]

def DisplayImages(initial_image, denoised_image,  blurred_image):
    """
    Display three images in subplots: initial image, blurred and noisy image, denoised image.

    Args:
        initial_image (numpy.ndarray): Initial image.
        blurred_image (numpy.ndarray): Blurred and noisy image.
        denoised_image (numpy.ndarray): Denoised image.

    Returns:
        None
    """
    plt.figure(figsize=(10, 4))

    # Plot initial image
    plt.subplot(131)
    plt.imshow(initial_image, cmap='gray')
    plt.title("Initial Image")
    plt.axis('off')

    # Plot blurred and noisy image
    plt.subplot(132)
    plt.imshow(blurred_image, cmap='gray')
    plt.title("Blurred and Noisy Image")
    plt.axis('off')

    # Plot denoised image
    plt.subplot(133)
    plt.imshow(denoised_image, cmap='gray')
    plt.title("Denoised Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def AnalyseErrorlambda(N, mu, pas, image):
    """
    Analyze the error for different values of lambda (λ).

    Args:
        N (int): Number of lambda values to analyze.
        mu (float): Regularization parameter for TV denoising.
        pas (float): Step size for lambda values.
        image (str): Path to the input image.

    Returns:
        tuple: Tuple containing lists of lambda values (K) and corresponding errors (MSE, SNR, SSIM).
    """
    # Generate lambda values
    K = [pas * k for k in range(1, N + 1)]
    E1 = []  # List to store MSE values
    E2 = []  # List to store SNR values
    E3 = []  # List to store SSIM values
    for k in range(len(K)):
        try:
            # Perform TV denoising and calculate errors
            I, Iapprox,Ih = TV(K[k], mu, image)
            E1.append(mse(I, Iapprox))
            E2.append(snr(Iapprox, I))
            E3.append(ssim_score(Iapprox, I))
        except:
            # If an error occurs during TV denoising, remove lambda value from the list
            K.remove(0.5 * (k + 1))
    return (K, E1, E2, E3)

def AfficheErrorLambda(K, MSE, SNR, SSIM):
    """
    Display the error values for different lambda (λ) values.

    Args:
        K (list): List of lambda values.
        MSE (list): List of Mean Squared Error (MSE) values.
        SNR (list): List of Signal to Noise Ratio (SNR) values.
        SSIM (list): List of Structural SIMilarity (SSIM) values.

    Returns:
        None
    """
    # Plot the error values
    plt.figure()
    plt.subplot(131)
    plt.plot(K, MSE)
    plt.title("MSE")
    plt.xlabel('λ')

    plt.subplot(132)
    plt.plot(K, SNR)
    plt.title("SNR")
    plt.xlabel('λ')

    plt.subplot(133)
    plt.plot(K, SSIM)
    plt.title("SSIM")
    plt.xlabel('λ')
    plt.show()

def AnalyzeErrorMu(N, lambda_val, mu_range, image):
    """
    Analyze the error for different values of mu.

    Args:
        N (int): Number of mu values to analyze.
        lambda_val (float): Regularization parameter lambda (λ) for TV denoising.
        mu_range (tuple): Range of mu values to analyze, specified as (mu_min, mu_max).
        image (str): Path to the input image.

    Returns:
        tuple: Tuple containing lists of mu values (Mu) and corresponding errors (MSE, SNR, SSIM).
    """
    # Generate mu values
    mu_values = np.linspace(mu_range[0], mu_range[1], N)
    E1 = []  # List to store MSE values
    E2 = []  # List to store SNR values
    E3 = []  # List to store SSIM values
    for mu in mu_values:
        try:
            # Perform TV denoising and calculate errors
            I, Iapprox, Ih = TV(lambda_val, mu, image)
            E1.append(mse(I, Iapprox))
            E2.append(snr(Iapprox, I))
            E3.append(ssim_score(Iapprox, I))
        except:
            pass
    return (mu_values, E1, E2, E3)

def DisplayErrorMu(Mu, MSE, SNR, SSIM):
    """
    Display the error values for different mu values.

    Args:
        Mu (list): List of mu values.
        MSE (list): List of Mean Squared Error (MSE) values.
        SNR (list): List of Signal to Noise Ratio (SNR) values.
        SSIM (list): List of Structural SIMilarity (SSIM) values.

    Returns:
        None
    """
    # Plot the error values
    plt.figure()
    plt.subplot(131)
    plt.plot(Mu, MSE)
    plt.title("MSE")
    plt.xlabel('μ')

    plt.subplot(132)
    plt.plot(Mu, SNR)
    plt.title("SNR")
    plt.xlabel('μ')

    plt.subplot(133)
    plt.plot(Mu, SSIM)
    plt.title("SSIM")
    plt.xlabel('μ')
    plt.show()
