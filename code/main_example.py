'''Module 3: count black and white pixels and compute the percentage of white pixels in a .jpg image and extrapolate points'''

from termcolor import colored
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

# Load the images you want to analyze

filenames = [
    r"../images/MASK_SK658 Llobe ch010039.jpg",
    r"../images/MASK_SK658 Slobe ch010066.jpg",
    r"../images/MASK_SK658 Slobe ch010147.jpg",
    r"../images/MASK_SK658 Slobe ch010110.jpg",
    r"../images/MASK_SK658 Slobe ch010130.jpg",
    r"../images/MASK_SK658 Slobe ch010114.jpg",
]
'''

filenames = [
    r"../images/MASK_SK658 Slobe ch010118.jpg",
    r"../images/MASK_SK658 Slobe ch010113.jpg",        
    r"../images/MASK_SK658 Slobe ch010098.jpg",
    r"../images/MASK_SK658 Llobe ch010022.jpg",
    r"../images/MASK_SK658 Slobe ch010089.jpg",
    r"../images/MASK_SK658 Slobe ch010156.jpg"
]


# Load depths from CSV
depth_df = pd.read_csv("Filenames and Depths for Students.csv")
depth_dict = dict(zip(depth_df["Filenames"], depth_df["Depth from lung surface (in micrometers) where image was acquired"]))
DEPTHS = [depth_dict[fn] for fn in FILENAMES]
 
OUTPUT_CSV = "Percent_White_Pixels.csv"
THRESHOLD = 127
 
OUTPUT_CSV = "Percent_White_Pixels.csv"
THRESHOLD = 127
 
# Helper functions
 
def load_image(filepath: str) -> np.ndarray:
    """Load a grayscale image; exit with a clear message if not found."""
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(colored(f"ERROR: Could not load image: {filepath}", "red"))
        sys.exit(1)
    return img
 
 
def count_pixels(img: np.ndarray, threshold: int = THRESHOLD) -> tuple[int, int]:
    """Return (white_count, black_count) for a grayscale image."""
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    white = int(np.sum(binary == 255))
    black = int(np.sum(binary == 0))
    return white, black
 
 
def white_percentage(white: int, black: int) -> float:
    """Return white pixels as a percentage of total pixels."""
    total = white + black
    return 100.0 * white / total if total > 0 else 0.0
 
 
def analyze_images(filenames: list[str]) -> tuple[list[int], list[int], list[float]]:
    """Load and analyze all images; return pixel counts and percentages."""
    white_counts, black_counts, white_percents = [], [], []
    for filepath in filenames:
        img = load_image(filepath)
        white, black = count_pixels(img)
        white_counts.append(white)
        black_counts.append(black)
        white_percents.append(white_percentage(white, black))
    return white_counts, black_counts, white_percents
 
# Reporting
 
def print_pixel_counts(filenames, white_counts, black_counts):
    print(colored("\nCounts of pixels by color in each image", "yellow"))
    for i, filepath in enumerate(filenames):
        print(colored(f"  Image {i} — White: {white_counts[i]:,}  |  Black: {black_counts[i]:,}", "white"))
 
 
def print_white_percents(filenames, depths, white_percents):
    print(colored("\nPercent white pixels per image", "yellow"))
    for i, filepath in enumerate(filenames):
        print(colored(f"  {filepath}", "red"))
        print(f"    {white_percents[i]:.2f}% white  |  Depth: {depths[i]:,} µm\n")
 
 
def save_csv(filenames, depths, white_percents, output_path: str = OUTPUT_CSV):
    df = pd.DataFrame({
        "Filename":      filenames,
        "Depth (µm)":   depths,
        "White (%)":    [round(p, 4) for p in white_percents],
    })
    df.to_csv(output_path, index=False)
    print(colored(f"Results saved to '{output_path}'.", "green"))

if __name__ == "__main__":
    white_counts, black_counts, white_percents = analyze_images(FILENAMES)
    print_pixel_counts(FILENAMES, white_counts, black_counts)
    print_white_percents(FILENAMES, DEPTHS, white_percents)
    save_csv(FILENAMES, DEPTHS, white_percents)

##############
# LECTURE 2: UNCOMMENT BELOW

# # Interpolate a point: given a depth, find the corresponding white pixel percentage

# interpolate_depth = float(input(colored(
#     "Enter the depth at which you want to interpolate a point (in microns): ", "yellow")))

# x = depths
# y = white_percents

# # You can also use 'quadratic', 'cubic', etc.
# i = interp1d(x, y, kind='linear')
# interpolate_point = i(interpolate_depth)
# print(colored(
#     f'The interpolated point is at the x-coordinate {interpolate_depth} and y-coordinate {interpolate_point}.', "green"))

# depths_i = depths[:]
# depths_i.append(interpolate_depth)
# white_percents_i = white_percents[:]
# white_percents_i.append(interpolate_point)


# # make two plots: one that doesn't contain the interpolated point, just the data calculated from your images, and one that also contains the interpolated point (shown in red)
# fig, axs = plt.subplots(2, 1)

# axs[0].scatter(depths, white_percents, marker='o', linestyle='-', color='blue')
# axs[0].set_title('Plot of depth of image vs percentage white pixels')
# axs[0].set_xlabel('depth of image (in microns)')
# axs[0].set_ylabel('white pixels as a percentage of total pixels')
# axs[0].grid(True)


# axs[1].scatter(depths_i, white_percents_i, marker='o',
#                linestyle='-', color='blue')
# axs[1].set_title(
#     'Plot of depth of image vs percentage white pixels with interpolated point (in red)')
# axs[1].set_xlabel('depth of image (in microns)')
# axs[1].set_ylabel('white pixels as a percentage of total pixels')
# axs[1].grid(True)
# axs[1].scatter(depths_i[len(depths_i)-1], white_percents_i[len(white_percents_i)-1],
#                color='red', s=100, label='Highlighted point')


# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()
