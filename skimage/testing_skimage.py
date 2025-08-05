import skimage as ski
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

image = ski.data.camera()

#plt.imshow(image)
#plt.axis("off")
#plt.show()

cat = ski.data.chelsea()
plt.imshow(cat)
plt.axis("off")
plt.show()