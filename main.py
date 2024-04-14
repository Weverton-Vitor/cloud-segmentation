from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os
from sklearn.mixture import GaussianMixture



def kmeans_segmentation(image_path, num_clusters, output_path, convert_to_gray=False):
    # Abrir a imagem
    image = Image.open(image_path)
    
    # Converter para tons de cinza, se necessário
    if convert_to_gray:
        image = image.convert('L')

    # Converte a imagem para um array numpy
    image_array = np.array(image)

    # Obtém a forma da imagem
    rows, cols = 0, 0
    num_channels = 1
    if len(image_array.shape) > 2:
        rows, cols, _ = image_array.shape
        num_channels = image_array.shape[2]
    else:
        rows, cols = image_array.shape


    # Redimensiona o array para que cada pixel seja uma amostra
    image_array_flat = image_array.reshape((-1, num_channels))
    
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
    segmented_image_array = []
    if convert_to_gray:
        segmented_image_array = segmented_image_array_flat.reshape((rows, cols))
    else:
        segmented_image_array = segmented_image_array_flat.reshape((rows, cols, num_channels))

    # Converte o array numpy de volta para uma imagem PIL
    segmented_image = Image.fromarray(segmented_image_array.astype('uint8'))

    # Salva a imagem segmentada
    segmented_image.save(output_path)



def gmm_segmentation(image_path, num_clusters, output_path, convert_to_gray=False):
    # Abrir a imagem
    image = Image.open(image_path)
    
    # Converter para tons de cinza, se necessário
    if convert_to_gray:
        image = image.convert('L')

    # Converte a imagem para um array numpy
    image_array = np.array(image)

    # Obtém a forma da imagem
    rows, cols = 0, 0
    num_channels = 1
    if len(image_array.shape) > 2:
        rows, cols, _ = image_array.shape
        num_channels = image_array.shape[2]
    else:
        rows, cols = image_array.shape
    # Redimensiona o array para que cada pixel seja uma amostra
    image_array_flat = image_array.reshape((-1, num_channels))
    
    # Aplica o algoritmo GMM
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    gmm.fit(image_array_flat)
    
    # Obtém os rótulos dos clusters para cada pixel
    labels = gmm.predict(image_array_flat)
    # Obtém as médias dos componentes para os clusters
    means = gmm.means_
    
    # Atualiza os valores dos pixels na imagem original com base nas médias dos componentes dos clusters
    segmented_image_array_flat = means[labels]

    # Redimensiona a imagem segmentada de volta para a forma original
    segmented_image_array = []
    if convert_to_gray:
        segmented_image_array = segmented_image_array_flat.reshape((rows, cols))
    else:
        segmented_image_array = segmented_image_array_flat.reshape((rows, cols, num_channels))


    # Converte o array numpy de volta para uma imagem PIL
    segmented_image = Image.fromarray(segmented_image_array.astype('uint8'))
    

    # Salva a imagem segmentada
    segmented_image.save(output_path)

images_dir = './images'
masks_dir = './masks'

images_path = os.listdir(images_dir)

# Exemplo de uso:
for image_path in images_path:
    output_path = f"{masks_dir}/{image_path.split('.')[0]}_kmeans_mask.{image_path.split('.')[1]}"
    output_path_gray = f"{masks_dir}/{image_path.split('.')[0]}_kmeans_mask_gray.{image_path.split('.')[1]}"

    num_clusters = 6  # Número de clusters para o algoritmo K-Means

    kmeans_segmentation(f'{images_dir}/{image_path}', num_clusters, output_path, convert_to_gray=False)
    kmeans_segmentation(f'{images_dir}/{image_path}', num_clusters, output_path_gray, convert_to_gray=True)
    
    output_path = f"{masks_dir}/{image_path.split('.')[0]}_gmm_mask.{image_path.split('.')[1]}"
    output_path_gray = f"{masks_dir}/{image_path.split('.')[0]}_gmm_mask_gray.{image_path.split('.')[1]}"

    gmm_segmentation(f'{images_dir}/{image_path}', num_clusters, output_path, convert_to_gray=False)
    gmm_segmentation(f'{images_dir}/{image_path}', num_clusters, output_path_gray, convert_to_gray=True)
