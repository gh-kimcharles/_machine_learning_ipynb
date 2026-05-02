import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Load dataset path
dataset_path = "crab-images-dataset/test"

# Dataset Preparation: Loop through images in each species folder and compute contours, hu moments, 
# area, perimeter, and circularity (target shape features) stored in data[] for dataframe
data = []

for species in os.listdir(dataset_path):
    species_path = os.path.join(dataset_path, species)
    
    if not os.path.isdir(species_path):  
        continue # Skip
        
    for img_name in os.listdir(species_path):
        img_path = os.path.join(species_path, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Skipping {img_name} in {species}, image could not be read.")
            continue  # Skip invalid images

        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
        if np.count_nonzero(thresh) == 0:
            print(f"Skipping {img_name} in {species}, blank image after thresholding.")
            continue  # Skip

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            hu_moments = cv2.HuMoments(cv2.moments(cnt)).flatten()[:5]  # Keep only Hu_0 - Hu_4
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            data.append([species, img_name] + list(hu_moments) + [area, perimeter, circularity])

# Convert to dataframe
columns = ["Species", "Image_Name"] + [f"Hu_{i}" for i in range(5)] + ["Area", "Perimeter", "Circularity"]
df = pd.DataFrame(data, columns=columns)

# Display the first few rows
print(df.head())

# Select only numerical columns for normalization
features = ["Hu_0", "Hu_1", "Hu_2", "Hu_3", "Hu_4", "Area", "Perimeter", "Circularity"]
scaler = MinMaxScaler()

df[features] = scaler.fit_transform(df[features])

print(df.head())

# Select two species for shape feature comparison
def cmpr_shape_feature_species(target, cmpr_species):

    species_to_compare = [target, cmpr_species]
    df_filtered = df[df["Species"].isin(species_to_compare)]
    
    # Compute mean values for each species
    df_mean = df_filtered.groupby("Species").mean(numeric_only=True).reset_index()
    
    # Plot Bar Charts for Shape Features
    features = ["Hu_0", "Hu_1", "Hu_2", "Hu_3", "Hu_4", "Area", "Perimeter", "Circularity"]
    df_melted = df_mean.melt(id_vars=["Species"], value_vars=features, var_name="Feature", value_name="Value")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Feature", y="Value", hue="Species", palette="Set2")
    plt.xticks(rotation=45)
    plt.title(f"Shape Feature Comparison: {target} vs {cmpr_species}")
    plt.legend(title="Species")
    
    # Save the figure
    filename = f"plot_{target}_vs_{cmpr_species}.png"
    plt.savefig(filename)
    plt.show()

# Compare Dungeness Crab to other species
target = "dungeness_crab"
species_list = ["fiddler_crab", "hermit_crab", "king_crab", "rock_crab"]

for species in species_list:
    cmpr_shape_feature_species(target, species)

# Compare Fiddler Crab to other species
target =  "fiddler_crab"
species_list = ["dungeness_crab", "hermit_crab", "king_crab", "rock_crab"]

for species in species_list:
    cmpr_shape_feature_species(target, species)

# Compare Hermit Crab to other species
target =  "hermit_crab"
species_list = ["dungeness_crab", "fiddler_crab", "king_crab", "rock_crab"]

for species in species_list:
    cmpr_shape_feature_species(target, species)

# Compare King Crab to other species
target =  "king_crab"
species_list = ["dungeness_crab", "fiddler_crab", "hermit_crab", "rock_crab"]

for species in species_list:
    cmpr_shape_feature_species(target, species)

# Compare Rock Crab to other species
target =  "rock_crab"
species_list = ["dungeness_crab", "fiddler_crab", "hermit_crab", "king_crab"]

for species in species_list:
    cmpr_shape_feature_species(target, species)

# Display shape feature correlation matrix
corr_matrix = df.drop(columns=["Species", "Image_Name"]).corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Shape Features")

# Save the figure
plt.savefig("plot_shape_feature_heatmap.png")
plt.show()