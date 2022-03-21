import cv2
import numpy as np
from PIL import Image
# 레고 판 잡기

# Load image, convert to HSV, and color threshold
image = cv2.imread('img/lego_38.jpg')
blank_mask = np.zeros(image.shape, dtype=np.uint8)

original = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower = np.array([0, 0, 170])
upper = np.array([179, 36, 255])
mask = cv2.inRange(hsv, lower, upper)

# Morph open to remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and sort for largest contour
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# Obtain rotated bounding box and draw onto a blank mask
rect = cv2.minAreaRect(cnts)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(image, [box], 0, (36, 255, 12), 3)
cv2.fillPoly(blank_mask, [box], (255, 255, 255))

# Bitwise-and mask with input image
blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)
result = cv2.bitwise_and(original, original, mask=blank_mask)
result[blank_mask == 0] = (255, 255, 255) # Color background white

# Detect corners
corners = cv2.goodFeaturesToTrack(blank_mask, maxCorners=4, qualityLevel=0.5, minDistance=150)

for corner in corners:
    x, y = corner.ravel()
    a = int(x); b = int(y)
    cv2.circle(image, (a, b), 8, (155, 20, 255), -1)
    #print("({}, {})".format(x, y))
print(corners)
cv2.imwrite('img/result.png', result)

x1, y1 = corners[0].ravel()
x2, y2 = corners[1].ravel()
x3, y3 = corners[2].ravel()
x4, y4 = corners[3].ravel()
xs = [int(x1), int(x2), int(x3), int(x4)]
xs.sort()
ys = [int(y1), int(y2), int(y3), int(y4)]
ys.sort()

result = Image.open('img/lego_38.jpg')
cropping_area = (xs[0], ys[0], xs[3], ys[3])
result = result.crop(cropping_area)
result.show()
#result.save('img/lego_36.jpg')
print(result.size)
# 전체 템플릿 찾기

#frame = cv2.imread('img/lego_33.jpg')
frame = result
frame = cv2.resize(frame, dsize=(640, 480))
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

cv2.imshow('frame', frame)
cv2.waitKey(0)

class Color:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.order = 0

# finding blue
lower_blue = np.array([101, 70, 0]) # light blue
upper_blue = np.array([115,255,255]) # darker blue
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 300:
        blue_x, blue_y, blue_w, blue_h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(frame, (blue_x, blue_y),
                                   (blue_x+blue_w, blue_y+blue_h),
                                   (255,0,0), 2)
        cv2.putText(imageFrame, 'Blue', (blue_x, blue_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

lower_green = np.array([64, 151, 0])
upper_green = np.array([89, 255, 255])
green_name = (lower_green + upper_green)/2
green_mask = cv2.inRange(hsv, lower_green, upper_green)
contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 300:
        green_x, green_y, green_w, green_h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(frame, (green_x, green_y),
                                   (green_x+green_w, green_y+green_h),
                                   green_name, 2)
        cv2.putText(imageFrame, 'Green', (green_x, green_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_name)

lower_pink = np.array([162, 100, 0])
upper_pink = np.array([171, 222, 255])
pink_name = (lower_pink + upper_pink)/2
orange_mask = cv2.inRange(hsv, lower_pink, upper_pink)
contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 300:
        pink_x, pink_y, pink_w, pink_h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(frame, (pink_x, pink_y),
                                   (pink_x+pink_w, pink_y+pink_h),
                                   (160, 103, 122), 2)
        cv2.putText(imageFrame, 'Pink', (pink_x, pink_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 103, 122))

lower_lpink = np.array([149, 50, 0])
upper_lpink = np.array([163, 104, 255])
lpink_name = (lower_lpink + upper_lpink)/2
lpink_mask = cv2.inRange(hsv, lower_lpink, upper_lpink)
contours, hierarchy = cv2.findContours(lpink_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 300:
        lpink_x, lpink_y, lpink_w, lpink_h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(frame, (lpink_x, lpink_y),
                                   (lpink_x+lpink_w, lpink_y+lpink_h),
                                   pink_name, 2)
        cv2.putText(imageFrame, 'Light Pink', (lpink_x+10,lpink_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lpink_name)

lower_lpurple = np.array([137, 85, 0])
upper_lpurple = np.array([150,163,255])
lpurple_name = (lower_lpurple + upper_lpurple)/2
lpurple_mask = cv2.inRange(hsv, lower_lpurple, upper_lpurple)
contours, hierarchy = cv2.findContours(lpurple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 300:
        lpurple_x, lpurple_y, lpurple_w, lpurple_h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(frame, (lpurple_x, lpurple_y),
                                   (lpurple_x+lpurple_w, lpurple_y+lpurple_h),
                                   lpurple_name, 2)
        cv2.putText(imageFrame, 'Light Purple', (lpurple_x+20, lpurple_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lpurple_name)

lower_purple = np.array([117, 50, 0])
upper_purple = np.array([133, 255, 255])
purple_name = (lower_purple + upper_purple)/2
purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
contours, hierarchy = cv2.findContours(purple_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 300:
        purple_x, purple_y, purple_w, purple_h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(frame, (purple_x,purple_y),
                                   (purple_x+purple_w, purple_y+purple_h),
                                   purple_name, 2)
        cv2.putText(imageFrame, 'Purple', (purple_x,purple_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, purple_name)

lower_orange = np.array([0, 213, 0])
upper_orange = np.array([12, 256, 255])
orange_name = (lower_orange + upper_orange)/2
orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 300:
        orange_x, orange_y, orange_w, orange_h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(frame, (orange_x,orange_y),
                                   (orange_x+orange_w, orange_y+orange_h),
                                   orange_name, 2)
        cv2.putText(imageFrame, 'Orange', (orange_x,orange_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, orange_name)


cv2.imwrite('result_0.jpg', imageFrame)

colors = []
colors_x = []
try:
    color_1 = Color('purple', purple_x)
    colors.append(color_1)
    colors_x.append(color_1.value)
except NameError:
    print("보라색이 검출되지 않았습니다.")

try:
    color_2 = Color('green', green_x)
    colors.append(color_2)
    colors_x.append(color_2.value)
except NameError:
    print("초록색이 검출되지 않았습니다.")

try:
    color_3 = Color('lpurple', lpurple_x)
    colors.append(color_3)
    colors_x.append(color_3.value)
except NameError:
    print("밝은 보라색이 검출되지 않았습니다.")

try:
    color_4 = Color('orange', orange_x)
    colors.append(color_4)
    colors_x.append(color_4.value)
except NameError:
    print("주황색이 검출되지 않았습니다.")

try:
    color_5 = Color('pink', pink_x)
    colors.append(color_5)
    colors_x.append(color_5.value)
except NameError:
    print("분홍색이 검출되지 않았습니다.")

try:
    color_6 = Color('blue', blue_x)
    colors.append(color_6)
    colors_x.append(color_6.value)
except NameError:
    print("파란색이 검출되지 않았습니다.")

try:
    color_7 = Color('lpink', lpink_x)
    colors.append(color_7)
    colors_x.append(color_7.value)
except NameError:
    print("밝은 분홍색이 검출되지 않았습니다.")

sorted_colors = colors_x.sort()
order_list =[]
for i in range(len(colors_x)):
    for j in range(len(colors)):
        if colors_x[i] == colors[j].value:
            order_list.append(colors[j].name)
for i in range(len(order_list)):
    if (i+1) == len(order_list):
        print(order_list[i])
    else:
        print(order_list[i], end = ' -> ')

cv2.imshow('frame', frame)
cv2.waitKey(0)
