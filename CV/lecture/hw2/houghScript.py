import cv2
from myHoughTransform import hough_transform
from myHoughLines import hough_lines
import numpy as np
import plotly.express as px


img = cv2.cvtColor(cv2.imread('data/img07.jpg'), cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=1), 40, 150)
height, width = img.shape[:2]
img_hough, rho_scale, theta_scale = hough_transform(edges)
indices = hough_lines(img_hough, 30)

fig = px.imshow(img_hough.T, labels=dict(x='rho', y='theta', color='count'),
                color_continuous_scale='Viridis', title='Hough Counters')
fig.show()


def get_hough_lines(rho, theta):
    eps = 1e-6
    m = -np.cos(theta) / (np.sin(theta) + eps)
    c = rho / (np.sin(theta) + eps)
    x_top = int(rho / (np.cos(theta) + eps))
    x_bottom = int(x_top - height * np.sin(theta) / (np.cos(theta) + eps))
    y_left = int(rho / (np.sin(theta) + eps))
    y_right = int(m * width + c)
    points = []
    if 0 <= x_top < width:
        points.append((x_top, 0))
    if 0 <= x_bottom < width:
        points.append((x_bottom, height))
    if 0 <= y_left < height:
        points.append((0, y_left))
    if 0 <= y_right < height:
        points.append((width, y_right))

    return points[:2]


# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=200, maxLineGap=200)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# for line in lines[:40]:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

for (i, j) in indices:
    point1, point2 = get_hough_lines(rho_scale[i], theta_scale[j])
    cv2.line(img, point1, point2, (0, 0, 255), 2)

cv2.imwrite('./data/img07_hough_line.jpg', img)
