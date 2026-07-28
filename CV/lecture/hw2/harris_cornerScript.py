import cv2
import numpy as np


def single_scale_harris(img, sigma, k=0.04):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    sum_sigma = sigma * 1.5
    sum_Ix2 = cv2.GaussianBlur(Ix2, (0, 0), sum_sigma)
    sum_Iy2 = cv2.GaussianBlur(Iy2, (0, 0), sum_sigma)
    sum_Ixy = cv2.GaussianBlur(Ixy, (0, 0), sum_sigma)

    det = sum_Ix2 * sum_Iy2 - sum_Ixy ** 2
    tr = sum_Ix2 + sum_Iy2
    r = det - k * (tr ** 2)
    return r


def multi_scale_harris(img, num_scales=5, scales_factor=1.5, k=0.04,
                       threshold_ratio=0.01, nms_win=5):
    sigmas = []
    responses = []
    sigma = 0.5
    for _ in range(num_scales):
        r = single_scale_harris(img, sigma=sigma, k=k)
        sigmas.append(sigma)
        responses.append(r)
        sigma *= scales_factor

    responses = np.stack(responses)
    threshold = np.max(np.max(responses, axis=1, keepdims=True), axis=2, keepdims=True) * threshold_ratio
    threshold_mask = (responses > threshold)
    responses = responses * threshold_mask.astype(np.uint8)
    r_max = np.max(responses, axis=0)
    scales = np.argmax(responses, axis=0)
    local_max = cv2.dilate(r_max, np.ones((nms_win, nms_win), np.uint8), iterations=1)
    nms_mask = (local_max == r_max)

    y_x_sigma = []
    ys, xs = np.where(nms_mask & (r_max > 0))
    for y, x in zip(ys, xs):
        y_x_sigma.append((int(y), int(x), sigmas[scales[y][x]]))

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for y, x, sigma in y_x_sigma:
        cv2.circle(img, (int(x), int(y)), int(sigma * 4), (0, 0, 255), 1)
    return img


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('data/img10.jpg'), cv2.COLOR_BGR2GRAY)
    result = multi_scale_harris(img)
    cv2.imwrite('harris/img10_harris.jpg', result)