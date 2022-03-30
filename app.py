import urllib
from flask import Flask, request, jsonify, make_response
import uuid
import boto3
import cv2
import numpy as np
from PIL import Image

s3 = boto3.client("s3")
app = Flask(__name__)


def rgb_fft(image):
    f_size = 25
    fft_images = []
    fft_images_log = []
    for i in range(3):
        rgb_fft = np.fft.fftshift(np.fft.fft2((image[:, :, i])))
        fft_images.append(rgb_fft)
        fft_images_log.append(np.log(abs(rgb_fft)))

    return fft_images, fft_images_log


def normalize_image(img):
    img = img / np.max(img)
    return (img*255).astype('uint8')

def write_background_images(images_log, images, names):
    for image, name in zip(images_log, names):
        image3 = cv2.merge((image, image, image))
        image_3_nor = normalize_image(image3)
        cv2.imwrite(f"{name}.png", image_3_nor)
        s3.upload_file(Bucket='fourierbucket', Key=f'{name}.png', Filename=f'{name}.png')

    for image, name in zip(images, names):
        np.save(name, image)


def get_mask_from_canvas(canvas_images):
    list_mask = []
    for image in canvas_images:
        list_mask.append(image[:, :, 3])

    return list_mask

def load_images(imageName):
    names = [f"bg_image_r_{imageName}", f"bg_image_g_{imageName}", f"bg_image_b_{imageName}"]
    images = []
    for name in names:
        images.append(np.load(f"{name}.npy"))
    return images


def apply_mask(input_image, mask):
    _, mask_thresh = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY)
    mask_bool = mask_thresh.astype('bool')
    input_image[mask_bool] = 1

    return input_image


def apply_mask_all(list_images, list_mask):
    final_result = []

    for (i, mask) in zip(list_images, list_mask):
        result = apply_mask(i, mask)
        final_result.append(result)
    return final_result


def inverse_fourier(image):
    final_image = []
    for c in image:
        channel = abs(np.fft.ifft2(c))
        final_image.append(channel)
    final_image_assebled = np.dstack([final_image[0].astype('int'), final_image[1].astype('int'), final_image[2].astype('int')])
    return final_image_assebled

@app.route("/fft", methods=["POST"])
def fft():
    imageName = request.form["name"]
    f = request.files['file']
    f.save(f'{imageName}.png')
    s3.upload_file(Bucket='fourierbucket', Key=f'{imageName}.png', Filename=f'{imageName}.png')
    original = Image.open(f"{imageName}.png")
    img = np.array(original)
    fft_images, fft_images_log = rgb_fft(img)
    names = [f"bg_image_r_{imageName}", f"bg_image_g_{imageName}", f"bg_image_b_{imageName}"]
    write_background_images(fft_images_log, fft_images, names)
    res = jsonify(image=imageName)
    res.headers["Access-Control-Allow-Origin"] = "*"
    return res


@app.route("/ifft", methods=["POST"])
def ifft():
    imageName = request.form['imageName']
    r = request.form['canvasR']
    g = request.form['canvasG']
    b = request.form['canvasB']
    response = urllib.request.urlopen(r)
    with open(f'{imageName}_mask_r.png', 'wb') as f:
        f.write(response.file.read())
    response = urllib.request.urlopen(g)
    with open(f'{imageName}_mask_g.png', 'wb') as f:
        f.write(response.file.read())
    response = urllib.request.urlopen(b)
    with open(f'{imageName}_mask_b.png', 'wb') as f:
        f.write(response.file.read())
    canvasImages = []
    canvasImages.append(cv2.imread(f'{imageName}_mask_r.png', -1))
    canvasImages.append(cv2.imread(f'{imageName}_mask_g.png', -1))
    canvasImages.append(cv2.imread(f'{imageName}_mask_b.png', -1))
    list_mask = get_mask_from_canvas(canvasImages)
    fft_images = load_images(imageName)
    result = apply_mask_all(fft_images, list_mask)
    transformed = inverse_fourier(result)
    transformed_clipped = np.clip(transformed, 0, 255)
    data = Image.fromarray(transformed_clipped.astype('uint8'))
    data.save(f"{imageName}_transformed.png")
    s3.upload_file(Bucket='fourierbucket', Key=f'{imageName}_transformed.png', Filename=f'{imageName}_transformed.png')
    res = jsonify(image=imageName)
    res.headers["Access-Control-Allow-Origin"] = "*"
    return res
