import numpy as np


def encode_jpg(img_array, quality=95):
    if img_array.dtype == np.float32:
        img_array = np.uint8(img_array * 255.0)
    img = Image.fromarray(img_array)
    bytesio = io.BytesIO()
    img.save(bytesio, format='jpeg')
    jpg_image = Image.open(bytesio)
    jpg_image = np.asarray(img, dtype=np.uint8)

    if img_array.dtype == np.float32:
        jpg_image = (img / 255.0).astype(np.float32)
    return jpg_image

