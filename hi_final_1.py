import cv2
import numpy as np

frame = cv2.imread('img/lego_18.jpg')
frame = cv2.resize(frame, dsize=(640, 480))
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
