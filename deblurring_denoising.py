import argparse

from functions import *


def main():
   
    parser = argparse.ArgumentParser(description="Deblur and denoise an image using total variation regularization.")
    parser.add_argument("--input_image", type=str, help="Path to the input image.", required=True)
    parser.add_argument("--output_directory", type=str, help="Directory to save the results.", required=True)
    parser.add_argument("--Lambda", type=float, help="Value of lambda for TV regularization.", required=True)
    parser.add_argument("--mu", type=float, help="Value of mu for TV regularization.", required=True)
    parser.add_argument("--iterations", type=int, help="Number of iterations for the optimization algorithm.", required=True)
    args = parser.parse_args()
    
    
    I,Iapprox, If = TV(args.input_image, args.Lambda, args.mu, args.iterations)
    DisplayImages(I,Iapprox,If)
    if args.output_directory : 
        cv2.imwrite(args.output_directory + "/deblurred_denoised_image.png", Iapprox)

if __name__ == "__main__":
    main()
