from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['figure.figsize'] = [16, 9]

image = imread(
    os.path.join(
        '/kaggle/input/peaky-blinder.jpg'))

normalized_image = image[:, :, :]

normalized_image = (normalized_image - normalized_image.min()) / (normalized_image.max() - normalized_image.min())

# Explicar a necessidade da normalizacao: https://en.wikipedia.org/wiki/Normalization_(image_processing)

img = plt.imshow(normalized_image)
plt.axis('off')
plt.show()

red_component = normalized_image[:, :, 0]
green_component = normalized_image[:, :, 1]
blue_component = normalized_image[:, :, 2]

U0, S0, VT0 = np.linalg.svd(red_component, full_matrices=False)
U1, S1, VT1 = np.linalg.svd(green_component, full_matrices=False)
U2, S2, VT2 = np.linalg.svd(blue_component, full_matrices=False)

S0 = np.diag(S0)
S1 = np.diag(S1)
S2 = np.diag(S2)

for precision in range(5, 200, 20):
    # Construct approximate image, one time for each color (RGB)
    red_approximation = U0[:, :precision] @ S0[0:precision, :precision] @ VT0[:precision, :]
    green_approximation = U1[:, :precision] @ S1[0:precision, :precision] @ VT1[:precision, :]
    blue_approximation = U2[:, :precision] @ S2[0:precision, :precision] @ VT2[:precision, :]

    image_approximation = np.zeros((normalized_image.shape[0], normalized_image.shape[1], normalized_image.shape[2]))
    image_approximation[:, :, 0] = red_approximation
    image_approximation[:, :, 1] = green_approximation
    image_approximation[:, :, 2] = blue_approximation

    plt.imshow(image_approximation)
    plt.axis('off')

    plt.title("precision = {} eigenimages - Compression ratio: {:.2f}%".format(precision, (
                (1 - precision / image.shape[0]) * 100)))
    plt.savefig("precision_" + str(precision))
    plt.show()

plt.title('Original image - precision = ' + str(image.shape[0]) + " eigenimages - Compression ratio: 0%")
plt.axis('off')
plt.imshow(image)
plt.tight_layout()
plt.show()

S = S2
# Explicar o S2

print(np.diag(S)[0] / np.diag(S)[len(S) - 1])
# Comentar esse quociente

plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular values\' magnitudes')
plt.ylabel("magnitude")
plt.xlabel("singular value's position")
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)))
plt.title('Singular values: cumulative sum')
plt.ylabel("magnitude percentage")
plt.xlabel("singular value's position")
plt.show()
