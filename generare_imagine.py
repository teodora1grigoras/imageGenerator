import time
import keras_cv
import keras
import matplotlib.pyplot as plt

    # Construirea modelului
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=False)
print("22ok")

# Promptul pentru generarea imaginii dorite
prompt = "fuchsia pie in a forest"  

print("33ok")
# Generarea imaginii pe baza promptului dat
images = model.text_to_image(prompt, batch_size=1)

print("11ok")
# Afișarea imaginii generate
plt.imshow(images[0])
plt.axis("off")
plt.show()

print("ok")