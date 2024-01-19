import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import re

def get_group_and_exp_size(codedict_path):
    """
    Get the group size (number of recordings) and the number of expressions in the codedict directory.

    Parameters:
    - codedict_path (str): The path to the codedict directory.

    Returns:
    - tuple: A tuple containing group size (int) and number of expressions (int).
    """

    # Calculate group size (number of folders)
    group_size = len(os.listdir(codedict_path))

    # Find the subfolder with the most expressions
    subfolders = [f.path for f in os.scandir(codedict_path) if f.is_dir()]
    expressions_folder = max(subfolders, key=lambda x: len(os.listdir(x)))

    # Calculate the number of expressions in the identified subfolder
    expressions = len(os.listdir(expressions_folder))

    return group_size, expressions

def get_params(codedict_path, exp_size=50, pose_size=6, shape_size=100, add_path=""):
    """
    Extract expression, pose, and shape parameters from codedict files.

    Parameters:
    - codedict_path (str): The path to the codedict directory.
    - exp_size (int): The size of the expression parameter array.
    - pose_size (int): The size of the pose parameter array.
    - shape_size (int): The size of the shape parameter array.

    Returns:
    - list: A list containing expression, pose, and shape parameter arrays.
    """

    # Get the Recording and Expressions numbers
    group_size, expressions = get_group_and_exp_size(codedict_path)
    print(f"Anzahl Aufnahmen: {group_size}")
    print(f"Anzahl Max Expressionen: {expressions}")

    # Initialize arrays to store parameters
    exp_params_left = np.zeros((expressions,group_size,exp_size))
    exp_params_right = np.zeros((expressions,group_size,exp_size))
    pose_params_left = np.zeros((expressions,group_size,pose_size))
    pose_params_right = np.zeros((expressions,group_size,pose_size))
    shape_params_left = np.zeros((expressions,group_size,shape_size))
    shape_params_right = np.zeros((expressions,group_size,shape_size))

    # Find the smallest number for ithe right index
    smallest_num = 100000

    for fp in os.listdir(codedict_path):
        pattern = r"_(\d+)$"
        match = re.search(pattern, fp)
        if match and pattern != "_7":
            num_rec = int(match.group(1))
            smallest_num = min(smallest_num, num_rec)
    
    print(smallest_num)

    # Extract parameters from codedict files
    for fp in os.listdir(codedict_path):
        pattern = r"_(\d+)$"
        match = re.search(pattern, fp)
        if match:
            num_rec = int(match.group(1))
            for fp_exp in os.listdir(os.path.join(codedict_path,fp)):
                pattern = r"_0?(\d+)$"
                match = re.search(pattern, fp_exp)
                if match:
                    num_exp = int(match.group(1))
                    codedict = torch.load(os.path.join(codedict_path,fp,fp_exp,fp_exp+f"{add_path}_codedict.pth"))
                    exp_params_left[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["left"]["exp"].cpu())[:exp_size]
                    exp_params_right[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["right"]["exp"].cpu())[:exp_size]
                    pose_params_left[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["left"]["pose"].cpu())[:pose_size]
                    pose_params_right[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["right"]["pose"].cpu())[:pose_size]
                    shape_params_left[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["left"]["shape"].cpu())[:shape_size]
                    shape_params_right[num_exp - 1, num_rec - smallest_num - 1] = np.array(codedict["right"]["shape"].cpu())[:shape_size]

    return [exp_params_left, exp_params_right, pose_params_left, pose_params_right, shape_params_left, shape_params_right]

def visualization_exp_shape(expression_indices, expressionnames, parameter_list):
    """
    Visualize expressions and shapes using t-SNE with interactive sliders.

    Parameters:
    - expression_indices (list): List of indices representing expressions.
    - expressionnames (list): List of expression names corresponding to expression_indices.
    - parameter_list (list): List of parameter arrays for expressions and shapes.

    Returns:
    None
    """
    # Define color mapping for expressions
    expression_colors = {
    "Neutral": "cyan",
    "Stirn runzeln": "orange",
    "Augen zu": "lightgreen",
    "Augen doll zu": "green",
    "Nase rümpfen": "purple",
    "Lächeln": "mistyrose",
    "Lächeln doll": "red",
    "Mund spitzen": "gray",
    "Wangen aufblasen":"yellow",
    "Zähne fletschen":"brown",
    "Mundwinkel hoch":"blue",
    "Mundwinkel runter": "lightblue",
    }

    # Extract parameters for expressions and shapes
    ind = expression_indices
    exp_params_extract = np.concatenate([parameter_list[0][ind], parameter_list[1][ind]], axis=-1)
    exp_params_combined = exp_params_extract.reshape(-1, exp_params_extract.shape[-1])

    shape_params_extract = np.concatenate([parameter_list[4][ind], parameter_list[5][ind]], axis=-1)
    shape_params_combined = shape_params_extract.reshape(-1, shape_params_extract.shape[-1])

    # Map expressions to colors
    colors = np.zeros(exp_params_combined.shape[0], dtype=object)
    for i in range(exp_params_combined.shape[0]):
        for j in range(len(ind)):
            for h in range(len(parameter_list[0][0])):
                if np.array_equal(exp_params_combined[i], exp_params_extract[j, h]):
                    colors[i] = expression_colors[expressionnames[j]]
                    break

    def update_plot(perplexity_1, perplexity_2):
        # Compute t-SNE
        tsne_1 = TSNE(n_components=2, perplexity=perplexity_1)
        X_tsne_1 = tsne_1.fit_transform(exp_params_combined)

        tsne_2 = TSNE(n_components=2, perplexity=perplexity_2)
        X_tsne_2 = tsne_2.fit_transform(shape_params_combined)

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))


        scatter_1 = axes[0].scatter(X_tsne_1[:, 0], X_tsne_1[:, 1], c=colors)
        axes[0].set_title(f'Expression TSNE with Perplexity={perplexity_1}')
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        legend_1 = axes[0].legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=expression_colors[expressionnames[i]], markersize=10)
            for i in range(len(expressionnames))
                ], labels=expressionnames)
        
        scatter_2 = axes[1].scatter(X_tsne_2[:, 0], X_tsne_2[:, 1], c=colors)
        axes[1].set_title(f'Shape TSNE with Perplexity={perplexity_2}')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

        plt.show()
        
    # Create sliders for perplexity values
    perplexity_slider_1 = widgets.IntSlider(value=5, min=1, max=50, step=1, description='Perplexity 1:')
    perplexity_slider_2 = widgets.IntSlider(value=5, min=1, max=50, step=1, description='Perplexity 2:')

    plot = widgets.interactive(update_plot, perplexity_1=perplexity_slider_1, perplexity_2=perplexity_slider_2)
    display(plot)