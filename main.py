import numpy as np
from matplotlib import pyplot as plt
import cv2
from warpgan import WarpGAN
import imageio


def transform_face(img):
    output = face_to_cartoon(img)
    
    # Prepare inputs
    x, y = 330, 70
    img_overlay_rgba = resize(output[: ,: ,:], 135)

    # Perform blending
    alpha_mask = img_overlay_rgba[:, :, 2]
    alpha_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(alpha_mask.shape[0],alpha_mask.shape[1]))
    img_result = img[:, :, :3].copy() / 255
    img_overlay = img_overlay_rgba[:, :, :3]
    overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask)

    return img_result


def face_to_cartoon(img):
    model_dir = "pretrained/warpgan_pretrained"
    num_styles = 1
    scale = 1.0
    aligned=False

    network = WarpGAN()
    network.load_model(model_dir)

    if not aligned:
        from align.detect_align import detect_align
        img = detect_align(img)

    img = (img - 127.5) / 128.0

    images = np.tile(img[None], [num_styles, 1, 1, 1])
    scales = scale * np.ones((num_styles))
    styles = np.zeros((num_styles, network.input_style.shape[1]))

    output = network.generate_BA(images, scales, 16, styles=styles)
    output = 0.5*output + 0.5

    return output[0]



def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


def resize(img, scale_percent):
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  return resized