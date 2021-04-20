import numpy as np
import cv2
from warpgan import WarpGAN
import imageio
from align.detect_align import detect_align
from matplotlib import pyplot as plt


def transform_face(img):
    face, alpha_mask = face_to_cartoon(img)
    
    img_result = img[:, :, :3].copy() / 255

    overlay_image_alpha(img_result, face, 0, 0, alpha_mask)

    # fig, axs = plt.subplots(2, 2)
    # axs[0,0].imshow(img)
    # axs[0,1].imshow(alpha_mask)
    # axs[1,0].imshow(face)
    # axs[1,1].imshow(img_result)
    # plt.show()

    return img_result.astype(np.float32)


def face_to_cartoon(img):
    model_dir = "./pretrained/warpgan_pretrained"
    num_styles = 1
    scale = 1.0
    aligned=False

    network = WarpGAN()
    network.load_model(model_dir)

    original_shape = (img.shape[1], img.shape[0])  

    if not aligned:
        img, tfm = detect_align(img)

    img2 = cv2.warpAffine(img, tfm, original_shape, flags=cv2.WARP_INVERSE_MAP)

    img = (img - 127.5) / 128.0

    images = np.tile(img[None], [num_styles, 1, 1, 1])
    scales = scale * np.ones((num_styles))
    styles = np.zeros((num_styles, network.input_style.shape[1]))

    output = network.generate_BA(images, scales, 16, styles=styles)
    output = 0.5*output + 0.5

    mask_size = (int(output[0].shape[0]), int(output[0].shape[1]))
    alpha_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, mask_size)
    alpha_mask = cv2.blur(alpha_mask.astype(np.float64),(25,25), borderType=cv2.BORDER_CONSTANT)

    face = cv2.warpAffine(output[0], tfm, original_shape, flags=cv2.WARP_INVERSE_MAP)
    alpha_mask = cv2.warpAffine(alpha_mask, tfm, original_shape, flags=cv2.WARP_INVERSE_MAP)

    return face, alpha_mask



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