import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import os
import PIL
import cv2
import time

# def preprocess_style_image(image_path, target_size=(224, 224)):
#     # Charger l'image et la prétraiter pour le modèle VGG-19
#     img = load_img(image_path, target_size=target_size)
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = vgg19.preprocess_input(img)
#     return img
#
# # Image de style (chemin vers l'image)
# style_image_path = "style.jpg"
# style_image = preprocess_style_image(style_image_path)
#
#
# # Charger le modèle VGG-19 sans les couches de classification
# vgg = vgg19.VGG19(weights="imagenet", include_top=False)
#
# # Spécifier les couches utilisées pour capturer les caractéristiques de style
# style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
#
# # Créer un sous-modèle qui renvoie uniquement les sorties des couches de style
# style_outputs = [vgg.get_layer(name).output for name in style_layers]
# style_model = tf.keras.Model(inputs=vgg.input, outputs=style_outputs)
#
#
# # Fonction pour calculer une matrice de Gram
# def gram_matrix(tensor):
#     result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)  # Produit tensoriel
#     shape = tf.shape(tensor)
#     num_elements = tf.cast(shape[1] * shape[2], tf.float32)
#     return result / num_elements
#
# # Extraire les caractéristiques de style
# style_features = style_model(style_image)
#
# # Calculer les matrices de Gram pour chaque couche de style
# style_gram_matrices = [gram_matrix(feature) for feature in style_features]
#
#
# # Sauvegarder les poids des couches convolutives nécessaires
# weights_dir = "style_weights"
# os.makedirs(weights_dir, exist_ok=True)
#
# for layer in style_layers:
#     conv_layer = vgg.get_layer(layer)
#
#     weights, biases = conv_layer.get_weights()  # Contient les poids du filtre et les biais
#
#     # Sauvegarder les poids (matrice 4D)
#     np.save(os.path.join(weights_dir, f"{layer}_weights.npy"), weights)
#
#     # Sauvegarder les biais (vecteur 1D)
#     np.save(os.path.join(weights_dir, f"{layer}_biases.npy"), biases)
#
#
# # Charger les poids ou biais
# weights = np.load("path/to/block1_conv1_weights.npy")
# biases = np.load("path/to/block1_conv1_biases.npy")
#
# # Afficher les formes des matrices
# print(f"Shape of weights: {weights.shape}")
# print(f"Shape of biases: {biases.shape}")
#
# # Afficher un échantillon des données
# print("Weights sample:")
# print(weights[0])  # Exemple : premier filtre
#
# print("Biases sample:")
# print(biases[:5])  # Exemple : 5 premiers biais

"""
########################################################################################################################
                                    Charger les poids des couches convolutives
########################################################################################################################
"""
def load_weights(weight_paths, bias_paths):
    weights = []
    biases = []
    for weight_path, bias_path in zip(weight_paths, bias_paths):
        weights.append(np.load(weight_path))  # Charger les poids (4D)
        biases.append(np.load(bias_path))    # Charger les biais (1D)
    return weights, biases

# Exemples de chemins
weight_paths = [
    "style_weights/block1_conv1_weights.npy",
    "style_weights/block2_conv1_weights.npy",
    "style_weights/block3_conv1_weights.npy",
    "style_weights/block4_conv1_weights.npy",
    "style_weights/block5_conv1_weights.npy"
]
bias_paths = [
    "style_weights/block1_conv1_biases.npy",
    "style_weights/block2_conv1_biases.npy",
    "style_weights/block3_conv1_biases.npy",
    "style_weights/block4_conv1_biases.npy",
    "style_weights/block5_conv1_biases.npy"
]

# Charger les poids et les biais
weights, biases = load_weights(weight_paths, bias_paths)

# # Afficher les formes des matrices
# for j in range(5):
#     print(f"Shape of weights: {weights[j].shape}")
# for i in range(5):
#     print(f"Shape of biases: {biases[i].shape}")
#
# # Afficher un échantillon des données
# print("Weights sample:")
# print(weights[0])  # Exemple : premier filtre
#
# print("Biases sample:")
# print(biases[:5])  # Exemple : 5 premiers biais

"""
########################################################################################################################
                                        Prétraitement de l'image capturée
########################################################################################################################
"""
def preprocess_image(image, target_size=(224, 224)):
    # Redimensionner l'image
    resized_image = cv2.resize(image, target_size)

    # cv2.imshow("Reference 224 x 224", resized_image)
    # cv2.waitKey(0)
    # Normaliser dans l'intervalle [-1, 1]
    normalized_image = resized_image / 127.5 - 1.0

    # Retourner l'image comme un tableau NumPy avec des canaux séparés
    return normalized_image.astype(np.float32)


# Simuler une capture d'image (remplacez par flux caméra réel)
camera_image = cv2.imread("camera_image.jpg")  # Charger une image pour tester
input_image = preprocess_image(camera_image)

"""
########################################################################################################################
                                            Convolution 2D + Relu
########################################################################################################################
"""
#Convolution 2D
def conv2d(input_image, filter_weights, biases):
    # Dimensions des filtres et de l'entrée
    filter_height, filter_width, input_channels, num_filters = filter_weights.shape
    input_height, input_width, _ = input_image.shape

    # Dimensions de la sortie
    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1
    output_image = np.zeros((output_height, output_width, num_filters), dtype=np.float32)

    # Appliquer la convolution
    for f in range(num_filters):  # Pour chaque filtre
        for h in range(output_height):  # Pour chaque position en hauteur
            for w in range(output_width):  # Pour chaque position en largeur
                for c in range(input_channels):  # Pour chaque canal
                    # Produit matriciel pour une position
                    region = input_image[h:h + filter_height, w:w + filter_width, c]
                    output_image[h, w, f] += np.sum(region * filter_weights[:, :, c, f])
                # Ajouter le biais du filtre
                output_image[h, w, f] += biases[f]
    return output_image

# Fonction ReLU
def relu(input_image):
    return np.maximum(0, input_image)

# Max Pooling
def max_pooling(input_image, pool_size=(2, 2)):
    input_height, input_width, input_channels = input_image.shape
    pool_height, pool_width = pool_size

    # Dimensions de sortie
    output_height = input_height // pool_height
    output_width = input_width // pool_width
    output_image = np.zeros((output_height, output_width, input_channels), dtype=np.float32)

    # Appliquer le max pooling
    for c in range(input_channels):  # Pour chaque canal
        for h in range(output_height):  # Pour chaque région de pooling
            for w in range(output_width):
                region = input_image[h*pool_height:(h+1)*pool_height, w*pool_width:(w+1)*pool_width, c]
                output_image[h, w, c] = np.max(region)
    return output_image


"""
########################################################################################################################
                                                Pipeline convolutif
########################################################################################################################
"""
def apply_cnn_pipeline(input_image, weights, biases):
    output_image = input_image
    for i in range(len(weights)):
        start_time = time.time()
        print(f"Applying Layer {i + 1}... weights len: {len(weights)}... out_im size: {output_image.shape}")
        output_image = conv2d(output_image, weights[i], biases[i])  # Convolution
        output_image = relu(output_image)                          # Activation
        end_time = time.time()
        if i < len(weights) - 1:
            print(f"Applying Max Pooling after Block {i + 1}...")
            output_image = max_pooling(output_image)
        print(f"Layer {i + 1} took {end_time - start_time:.4f} seconds")
    return output_image

"""
########################################################################################################################
                                               Reconstruction de l'image
########################################################################################################################
"""
def postprocess_image(output_image):
    # Normaliser dans l'intervalle [0, 255]
    normalized_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    final_image = (normalized_image * 255).astype(np.uint8)
    return final_image


"""
########################################################################################################################
                                              Pipeline Complet
########################################################################################################################
"""
def style_transfer_pipeline(camera_image, weights, biases):
    # Étape 1 : Prétraitement
    input_image = preprocess_image(camera_image)

    # Étape 2 : Pipeline convolutif
    stylized_image = apply_cnn_pipeline(input_image, weights, biases)

    # Étape 3 : Reconstruction
    final_image = postprocess_image(stylized_image)

    return final_image

"""
########################################################################################################################
                                              Call function for final matrix
########################################################################################################################
"""
# # Appliquer le pipeline complet
# stylized_result = style_transfer_pipeline(camera_image, weights, biases)
#
# print(stylized_result.shape)
#
# # Sauvegarder les données dans un fichier .npy
# def save_data_npy(filename, data):
#     np.save(filename, data)
#     print(f"Data saved to {filename}")
#
# # Exemple d'utilisation
# save_data_npy("stylized_result.npy", stylized_result)

"""
########################################################################################################################
                                             Constructon de l'image finale
########################################################################################################################
"""
# Charger les données sauvegardées
def load_data_npy(filename):
    data = np.load(filename)
    print(f"Data loaded from {filename}. Shape: {data.shape}")
    return data

# Upsampling bilinéaire
def upsample(input_image, target_size=(224, 224)):
    return cv2.resize(input_image, target_size, interpolation=cv2.INTER_LINEAR)

# Réduction des canaux (convolution 1x1 simulée)
def conv1x1(input_image, reduce_weights, reduce_biases):
    height, width, input_channels = input_image.shape
    num_filters = reduce_weights.shape[-1]  # Généralement 3 pour RGB
    output_image = np.zeros((height, width, num_filters), dtype=np.float32)

    for f in range(num_filters):  # Pour chaque filtre
        for c in range(input_channels):  # Pour chaque canal
            output_image[:, :, f] += input_image[:, :, c] * reduce_weights[c, f]
        output_image[:, :, f] += reduce_biases[f]
    return output_image

# Normalisation de l'image
def normalize_image(input_image):
    normalized_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    return (normalized_image * 255).astype(np.uint8)

# Charger les données stylisées
stylized_result = load_data_npy("stylized_result.npy")  # Shape (10, 10, 512)

# Étape 1 : Upsampling
upsampled_result = upsample(stylized_result, target_size=(224, 224))

# Étape 2 : Réduction des canaux (512 → 3)
reduce_weights = np.random.rand(512, 3)  # Poids aléatoires pour tester
reduce_biases = np.random.rand(3)
rgb_result = conv1x1(upsampled_result, reduce_weights, reduce_biases)

# Étape 3 : Normalisation
final_image = normalize_image(rgb_result)

# Sauvegarder ou afficher l'image finale
cv2.imwrite("stylized_final_image.jpg", final_image)
cv2.imshow("Stylized Image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()