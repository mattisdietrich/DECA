# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import re
import cv2

def get_blendshapes(image_path):
    image = mp.Image.create_from_file(image_path)
    # STEP 2: Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    detection_result = detector.detect(image)

    blendshape_scores = [face_blendshapes_category.score for face_blendshapes_category in detection_result.face_blendshapes[0]]

    return blendshape_scores

def get_face_landmarks(image_path):
    # Initialisiere MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    # Lade das Bild
    image = cv2.imread(image_path)

    # Konvertiere das Bild in das RGB-Format (MediaPipe erwartet ein RGB-Bild)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Erkenne Gesichter und extrahiere Landmarken
    results = face_mesh.process(image_rgb)
    face_landmarks_list = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                face_landmarks_list.append((landmark.x, landmark.y, landmark.z))

    return face_landmarks_list


def blendshapes_for_dataset(base_folder, num_exp=9, num_bs=52, end = ""):
    entries = os.listdir(base_folder)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(base_folder, entry))]
    individuals = len(folders)
    bs_list = np.zeros((individuals, num_exp, num_bs))
    count = 0
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.lower().endswith((f'{end}.jpg', f'{end}.JPG')):
                    match = re.search(r'\d+', file_name[::-1])
                    exp_num = int(match.group()[::-1])
                    if exp_num == 11:
                        exp_num = 9
                    if not exp_num >= 10:
                        blendshapes = get_blendshapes(file_path)
                        bs_list[count, exp_num-1] = blendshapes
        count+=1
    
    return bs_list

def blendshapes_for_dataset_2fold(base_folder, num_exp=9, num_bs=52, end = ""):
    entries = os.listdir(base_folder)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(base_folder, entry))]
    individuals = len(folders)
    bs_list = np.zeros((individuals, num_exp, num_bs))
    count = 0
    for folder_name in os.listdir(base_folder):
        folder_path_start = os.path.join(base_folder, folder_name)
        for folder_num in os.listdir(folder_path_start):
            folder_path = os.path.join(folder_path_start, folder_num)
            match = re.search(r'\d+', folder_num[::-1])
            exp_num = int(match.group()[::-1])
            for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if file_name.lower().endswith((f'{end}.png', f'{end}.JPG')):
                        if exp_num == 11:
                            exp_num = 9
                        if not exp_num >= 10:
                            blendshapes = get_blendshapes(file_path)
                            bs_list[count, exp_num-1] = blendshapes
        count+=1
    
    return bs_list

def show_tSNE(blendshapes, expressionnames, position="upper right"):
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
    # Reshape bs_list zu (234, 52), wo 234 die Anzahl der Datenpunkte ist (26 * 9)
    bs_list_reshaped = blendshapes.reshape(-1, blendshapes.shape[-1])

    # Erstelle ein Label-Array für die Expression-Namen
    labels = np.array([expressionnames[i % len(expressionnames)] for i in range(bs_list_reshaped.shape[0])])

    # Wende t-SNE an
    tsne = TSNE(n_components=2, random_state=42)
    embedded_data = tsne.fit_transform(bs_list_reshaped)

    # Zähle die Anzahl der Punkte pro Expression
    point_counts = {label: np.sum(labels == label) for label in expressionnames}

    # Sortiere die Expressionen nach der Anzahl der Punkte
    sorted_expressionnames = sorted(expressionnames, key=lambda x: point_counts[x])

    # Plotte die Daten mit verschiedenen Farben für verschiedene Expression-Namen
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=name, markerfacecolor=expression_colors[name], markersize=10)
                    for name in sorted_expressionnames]

    plt.figure(figsize=(12, 8))
    for label in sorted_expressionnames:
        indices = np.where(labels == label)
        plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], c=expression_colors[label], s=50)

    plt.legend(handles=legend_elements, loc=position)
    plt.title('t-SNE Visualization of Blendshapes with Expression Colors')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

def diff_neutral(blendshapes, exp_num=9):
    # Finde die Indizes der "Neutral"-Expression
    neutral_bs_values = blendshapes[:, 0, :]

    # Initialisiere eine Liste für die Differenzen
    bs_list_diff_list = []

    # Iteriere über alle Spalten
    for j in range(exp_num):
        # Berechne die Differenz zu den "Neutral"-Werten
        bs_list_diff = blendshapes[:, j, :] - neutral_bs_values
        bs_list_diff_list.append(bs_list_diff)

    # Erstelle ein 3D-Array aus der Liste der Differenzen
    bs_list_diff_array = np.stack(bs_list_diff_list, axis=1)

    # Extrahiere die Spalten 1 bis 8 (Index 0 bis 7)
    blendshapes_diff = bs_list_diff_array[:, 1:exp_num, :]

    return blendshapes_diff