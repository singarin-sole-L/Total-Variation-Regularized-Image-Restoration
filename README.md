# Image Deblurring and Denoising

This Python script is designed to perform image deblurring and denoising using a total variation (TV) regularization method. It takes a blurred and noisy input image as input and produces a denoised output image using an iterative optimization algorithm. In order to test the accuracy of the program, the script takes as an argument an image without blurring. He applies a Gaussian blur and tries to deflect this image.

## Overview

The script implements a total variation (TV) regularization-based optimization algorithm for image deblurring and denoising. It utilizes gradient descent with proximal operators to minimize a cost function composed of a data fidelity term and a TV regularization term.

## Requirements

- Python 3.x
- numpy
- matplotlib
- OpenCV (cv2)
- scikit-image

You can install the required dependencies using pip:

```bash
pip install numpy matplotlib opencv-python scikit-image
```

## Usage
Run the script `deblurring_denoising.py` from the command line with the following arguments for example:
```bash
python deblurring_denoising.py --input_image data/leopard.png --output_directory results --Lambda 0.1 --mu 0.01 --iterations 100
```
--input_image: Path to the input image (blurred and noisy).
--output_directory: Directory where the results will be saved.
--lambda: Regularization parameter controlling the balance between data fidelity and TV regularization (float value).
--mu: Parameter controlling the step size for gradient descent (float value).
--iterations: Number of iterations for the optimization algorithm (integer value).

## Output
The script generates the following outputs:

1.Deblurred and Denoised Image: The final denoised output image.
![example](https://github.com/Lxvxo/Total-variation-optimization-for-image-deflection/assets/113984090/cf98a995-f925-4196-ae06-b2341eccf9f6)

2.MSE, SNR, and SSIM Plots: Plots showing the Mean Squared Error (MSE), Signal-to-Noise Ratio (SNR), and Structural Similarity Index (SSIM) over iterations
For plot the noise, you have to recup the funtions in `functions.py` and launched yourself the program with your settings.
Example : 
![l1](https://github.com/Lxvxo/Total-variation-optimization-for-image-deflection/assets/113984090/aa6c3726-6ae6-47bb-8d0a-d58a8d586454)

## Report 
The report correctly explains the mathematical resolution by the ADMM method (Alternating Directions Algorithm) for image deflection. It is written in French and contains various illustrations to help understand the influence of different parameters on the accuracy of the total variation model.
