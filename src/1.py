import numpy as np, random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from keras.datasets import mnist
from scipy import stats

red = '#e56b6f'
blue = '#22577a'

bar_width = 0.5
line_thickness = 2

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
set_seed(810109203)

# A
(_, _) , (test_images, _) = mnist.load_data()
test_images = test_images.reshape(test_images.shape[0] , -1)
test_images = test_images.astype('float32') / 255.0

# B
autoencoder = tf.keras.models.load_model('mnist_AE.h5')
reconstructed_images = autoencoder.predict(test_images)

# C
n = len(test_images)

choices_cnt = 4
rands = [np.random.randint(n) for _ in range(choices_cnt)]

cmap = matplotlib.colormaps.get_cmap('grey')
fig, axs = plt.subplots(2, choices_cnt)

for i in range(choices_cnt):
    rand_num= rands[i]
    axs[0, i].imshow(test_images.reshape(n, 28, 28)[rand_num], cmap=cmap)
    axs[1, i].imshow(reconstructed_images.reshape(n, 28, 28)[rand_num], cmap=cmap)

axs[0, 0].set_ylabel('Original')
axs[1, 0].set_ylabel('Reconstructed')
plt.show()

# D
mse = np.sum((test_images - reconstructed_images) ** 2, axis=1) / n

plt.hist(mse, density=True, alpha=0.7, label='MSE', color=red, edgecolor='black', bins=100)
plt.title('Mean Squared Error (MSE)')
plt.ylabel('Density')
plt.xlabel('MSE')
plt.grid(True)
plt.legend()
plt.show()

# E
mu = np.mean(mse)
var = np.var(mse)
sigma = var ** 0.5
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.plot(x, stats.norm.pdf(x, mu, sigma), label="X ~ N({:.2e}, {:.2e})".format(mu, sigma), lw=line_thickness, color=blue)

plt.hist(mse, density=True, alpha=0.7, label='MSE', color=red, edgecolor='black', bins=100)
plt.title('Mean Squared Error (MSE)')
plt.ylabel('Density')
plt.xlabel('MSE')
plt.grid(True)
plt.legend()
plt.show()

ks_statistic, p_value = stats.kstest(mse, cdf='norm', args=(mu, sigma))
print(p_value)
