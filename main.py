from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

def kmeans_segmentation(image_path, num_clusters, output_path):
    # Abre a imagem
    image = Image.open(image_path)
    # Converte a imagem para um array numpy
    image_array = np.array(image)
    # Obtém a forma da imagem
    rows, cols, _ = image_array.shape
    # Redimensiona o array para que cada pixel seja uma amostra
    image_array_flat = image_array.reshape((-1, 3))
    
    # Aplica o algoritmo K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(image_array_flat)
    
    # Obtém os rótulos dos clusters para cada pixel
    labels = kmeans.labels_
    # Obtém os centróides dos clusters
    centroids = kmeans.cluster_centers_
    
    # Atualiza os valores dos pixels na imagem original com base nos centróides dos clusters
    segmented_image_array_flat = centroids[labels]
    # Redimensiona a imagem segmentada de volta para a forma original
    segmented_image_array = segmented_image_array_flat.reshape((rows, cols, 3))
    # Converte o array numpy de volta para uma imagem PIL
    segmented_image = Image.fromarray(segmented_image_array.astype('uint8'))

    # Salva a imagem segmentada
    segmented_image.save(output_path)


import os

images_dir = './images'
masks_dir = './masks'

images_path = os.listdir(images_dir)
print(images_path)

# Exemplo de uso:
for image_path in images_path:
    output_path = f"{masks_dir}/{image_path.split('.')[0]}_mask.{image_path.split('.')[1]}"
    num_clusters = 10  # Número de clusters para o algoritmo K-Means
    kmeans_segmentation(f'{images_dir}/{image_path}', num_clusters, output_path)
