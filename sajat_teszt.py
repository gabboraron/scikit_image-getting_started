import matplotlib.pyplot as plt
import numpy as np

from skimage import data
from skimage import data_dir
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io
import os
from skimage.transform import rescale, resize, downscale_local_mean


filename = os.path.join("images/test_image.jpg")
img = io.imread(filename)
#image = rescale(img, 1.0/4.0, anti_aliasing=True)
image = resize(img, (231,263))

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
                               sharex=True, sharey=True)

img0 = ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title("Image")
ax0.axis("off")
fig.colorbar(img0, ax=ax0)

image = np.squeeze(image) 
img1 = ax1.imshow(entropy(image, disk(5)), cmap='gray')
ax1.set_title("Entropy")
ax1.axis("off")
fig.colorbar(img1, ax=ax1)

fig.tight_layout()

plt.show()