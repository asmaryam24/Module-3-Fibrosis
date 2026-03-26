"""
Module 3: Count black and white pixels, compute the percentage of white pixels
in .jpg images, write results to CSV, and optionally interpolate a point.
"""
 
import sys
from time import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from termcolor import colored
import time


# Configuration

# choosing 6 images to analyze
FILENAMES = [
    r"images/MASK_SK658 Slobe ch010118.jpg",
    r"images/MASK_SK658 Slobe ch010113.jpg",        
    r"images/MASK_SK658 Slobe ch010098.jpg",
    r"images/MASK_SK658 Llobe ch010022.jpg",
    r"images/MASK_SK658 Llobe ch010168.jpg",
    r"images/MASK_SK658 Slobe ch010156.jpg"
]

# Load depths from CSV
depth_df = pd.read_csv("Filenames and Depths for Students.csv")
depth_dict = dict(zip(depth_df["Filenames"], depth_df["Depth from lung surface (in micrometers) where image was acquired"])) # Create a list of depths corresponding to the FILENAMES
DEPTHS = [depth_dict[fn] for fn in FILENAMES] # Create a list of depths corresponding to the FILENAMES
 
OUTPUT_CSV = "Percent_White_Pixels.csv" # Output CSV filename
THRESHOLD = 127 # Threshold for binarization (0-255); adjust as needed based on image characteristics
 
# Helper functions
def load_image(filepath: str) -> np.ndarray: # type: (str) -> np.ndarray
    """Load a grayscale image; exit with a clear message if not found."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) # Load as grayscale
    if img is None: # Check if the image was loaded successfully
        print(colored(f"ERROR: Could not load image: {filepath}", "red")) # Print error message in red
        sys.exit(1) # Exit the program with a non-zero status to indicate an error
    return img # Return the loaded image as a NumPy array
 
 
def count_pixels(img: np.ndarray, threshold: int = THRESHOLD) -> tuple[int, int]: # type: (np.ndarray, int) -> tuple[int, int]
    """Return (white_count, black_count) for a grayscale image."""
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY) # Binarize the image using the specified threshold
    white = int(np.sum(binary == 255)) # Count white pixels (255 in binary image)
    black = int(np.sum(binary == 0)) # Count black pixels (0 in binary image)
    return white, black # Return the counts of white and black pixels as integers
 
 
def white_percentage(white: int, black: int) -> float: # type: (int, int) -> float
    """Return white pixels as a percentage of total pixels."""
    total = white + black # Calculate total pixels
    return 100.0 * white / total if total > 0 else 0.0 # Return percentage of white pixels; handle division by zero
 
 
def analyze_images(filenames: list[str]) -> tuple[list[int], list[int], list[float]]: # type: (list[str]) -> tuple[list[int], list[int], list[float]]
    """Load and analyze all images; return pixel counts and percentages."""
    white_counts, black_counts, white_percents = [], [], [] # Initialize lists to store results
    for filepath in filenames: # Loop through each file path in the list of filenames
        img = load_image(filepath)
        white, black = count_pixels(img)
        white_counts.append(white)
        black_counts.append(black)
        white_percents.append(white_percentage(white, black))
    return white_counts, black_counts, white_percents # Return the lists of white counts, black counts, and white percentages for all images


# Reporting
 
def print_pixel_counts(filenames, white_counts, black_counts): # type: (list[str], list[int], list[int]) -> None
    print(colored("\nCounts of pixels by color in each image", "yellow")) # Print header in yellow
    for i, filepath in enumerate(filenames): # Loop through each file path and corresponding pixel counts
        print(colored(f"  Image {i} — White: {white_counts[i]:,}  |  Black: {black_counts[i]:,}", "white")) # Print pixel counts for each image in white
 
 
def print_white_percents(filenames, depths, white_percents): # type: (list[str], list[int], list[float]) -> None
    print(colored("\nPercent white pixels per image", "yellow")) # Print header in yellow
    for i, filepath in enumerate(filenames): # Loop through each file path and corresponding white pixel percentage
        print(colored(f"  {filepath}", "red")) # Print the filename in red
        print(f"    {white_percents[i]:.2f}% white  |  Depth: {depths[i]:,} µm\n") # Print the percentage of white pixels and corresponding depth for each image
 
 
def save_csv(filenames, depths, white_percents, output_path: str = OUTPUT_CSV): # type: (list[str], list[int], list[float], str) -> None
    df = pd.DataFrame({ # Create a DataFrame to store results
        "Filename":      filenames,
        "Depth (µm)":   depths,
        "White (%)":    [round(p, 4) for p in white_percents],
    })
    df.to_csv(output_path, index=False) # Save the DataFrame to a CSV file without the index
    print(colored(f"Results saved to '{output_path}'.", "green")) # Print confirmation message in green

if __name__ == "__main__": # Run the analysis and reporting when the script is executed
    white_counts, black_counts, white_percents = analyze_images(FILENAMES) # Analyze the images and get pixel counts and percentages
    print_pixel_counts(FILENAMES, white_counts, black_counts) # Print the pixel counts for each image
    print_white_percents(FILENAMES, DEPTHS, white_percents) # Print the percentage of white pixels and corresponding depths for each image
    save_csv(FILENAMES, DEPTHS, white_percents) # Save the results to a CSV file

if __name__ == "__main__": # Run the analysis and reporting when the script is executed, and also measure runtime
    start = time.time() # Record the start time

    white_counts, black_counts, white_percents = analyze_images(FILENAMES) # Analyze the images and get pixel counts and percentages
    print_pixel_counts(FILENAMES, white_counts, black_counts) # Print the pixel counts for each image
    print_white_percents(FILENAMES, DEPTHS, white_percents) # Print the percentage of white pixels and corresponding depths for each image
    save_csv(FILENAMES, DEPTHS, white_percents) # Save the results to a CSV file

    end = time.time() # Record the end time
    print(f"\nTotal runtime: {end - start:.4f} seconds") # Print the total runtime of the analysis and reporting in seconds


if __name__ == "__main__":
    start = time.time()

    # Run analysis
    white_counts, black_counts, white_percents = analyze_images(FILENAMES)
    print_pixel_counts(FILENAMES, white_counts, black_counts)
    print_white_percents(FILENAMES, DEPTHS, white_percents)
    save_csv(FILENAMES, DEPTHS, white_percents)

    end = time.time()
    print(f"\nTotal runtime: {end - start:.4f} seconds")

    # Interpolation Section

    interpolate_depth = float(input(colored(
        "Enter the depth at which you want to interpolate a point (in microns): ",
        "yellow"
    )))

    # Sort for interpolation
    x = np.array(DEPTHS)
    y = np.array(white_percents)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    interp_fn = interp1d(x_sorted, y_sorted, kind='cubic', fill_value="extrapolate")
    interpolated_y = float(interp_fn(interpolate_depth))

    print(colored(
        f"The interpolated point is at depth {interpolate_depth} µm "
        f"with estimated white pixel percentage {interpolated_y:.4f}%.",
        "green"
    ))

    # Plotting
    depths_i = list(x_sorted) + [interpolate_depth]
    white_percents_i = list(y_sorted) + [interpolated_y]

    fig, axs = plt.subplots(2, 1, figsize=(7, 10))

    axs[0].plot(x_sorted, y_sorted, marker='o', color='blue')
    axs[0].set_title('Depth vs % White Pixels')
    axs[0].grid(True)

    axs[1].plot(x_sorted, y_sorted, marker='o', color='blue')
    axs[1].scatter(interpolate_depth, interpolated_y, color='red', s=120)
    axs[1].set_title('Depth vs % White Pixels (with interpolation)')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    