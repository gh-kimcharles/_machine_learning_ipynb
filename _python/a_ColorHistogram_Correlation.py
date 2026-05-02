# Import libraries for correlation
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset path
dataset_dir = "crab-images-dataset/test"

# Extract and compute histogograms for red, green, and blue (RGB) channels
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Compute histograms for Red, Green, Blue channels
    hist_b = cv2.calcHist([image], [0], None, [bins[0]], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [bins[1]], [0, 256]).flatten()
    hist_r = cv2.calcHist([image], [2], None, [bins[2]], [0, 256]).flatten()
    
    # Normalize histograms
    hist_r /= (hist_r.sum() + 1e-6)
    hist_g /= (hist_g.sum() + 1e-6)
    hist_b /= (hist_b.sum() + 1e-6)
    
    return np.concatenate([hist_r, hist_g, hist_b])

# Dataset Preparation: Store species and histogram features in the data[] for dataframe
data = []

for species in os.listdir(dataset_dir):  
    species_path = os.path.join(dataset_dir, species)
    if os.path.isdir(species_path):  # Ensure it's a directory
        for image_file in os.listdir(species_path):
            if image_file.endswith(('.jpg', '.png')):  # Only process images
                image_path = os.path.join(species_path, image_file)
                hist = extract_color_histogram(image_path)
                
                # Store species label and histogram features
                data.append([species] + hist.tolist())

# Convert to dataframe
columns = ["Species"] + [f"Bin_{i}" for i in range(len(hist))]
df = pd.DataFrame(data, columns=columns)

# Display rows
print(df.head())

# Compute mean color histogram for each species
df_avg = df.groupby("Species").mean()

# Display average histograms per species
print(df_avg)

# Select two species for color histogram comparison
def cmpr_color_hist_species(target, cmpr_species):
    hist1 = df[df["Species"] == target].iloc[:, 1:].mean()
    hist2 = df[df["Species"] == cmpr_species].iloc[:, 1:].mean()
    
    # Plot histograms using line graph
    plt.figure(figsize=(15, 5))
    plt.plot(hist1, color='red', label=target)
    plt.plot(hist2, color='blue', label=cmpr_species)
    plt.title(f"Color Histogram Comparison: {target} vs {cmpr_species}")
    plt.legend()

    # Save the figure
    filename = f"plot_{target}_vs_{cmpr_species}.png"
    plt.savefig(filename)
    
    plt.show()

# Compare Dungeness Crab to other species
target = "dungeness_crab"
species_list = ["fiddler_crab", "hermit_crab", "king_crab", "rock_crab"]

for species in species_list:
    cmpr_color_hist_species(target, species)

# Compare Fiddler Crab to other species
target =  "fiddler_crab"
species_list = ["dungeness_crab", "hermit_crab", "king_crab", "rock_crab"]

for species in species_list:
    cmpr_color_hist_species(target, species)

# Compare Hermit Crab to other species
target =  "hermit_crab"
species_list = ["dungeness_crab", "fiddler_crab", "king_crab", "rock_crab"]

for species in species_list:
    cmpr_color_hist_species(target, species)

# Compare King Crab to other species
target =  "king_crab"
species_list = ["dungeness_crab", "fiddler_crab", "hermit_crab", "rock_crab"]

for species in species_list:
    cmpr_color_hist_species(target, species)

# Compare Rock Crab to other species
target =  "rock_crab"
species_list = ["dungeness_crab", "fiddler_crab", "hermit_crab", "king_crab"]

for species in species_list:
    cmpr_color_hist_species(target, species)

# Display correlation matrix between crab species
corr_matrix = df_avg.corr()

# Plot heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Color Histogram Correlation Between Crab Species")

# Save the figure
plt.savefig("plot_color_hist_heatmap.png")
plt.show()

# Display species-based correlation matrix
corr_matrix = df_avg.T.corr()

# Plot heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Species-Based Color Histogram Correlation")

# Save the figure
plt.savefig("plot_color_hist_heatmap_species-based.png")
plt.show()