import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# stacking images and creating contour: thx https://stackoverflow.com/questions/42420470/opencv-subplots-images-with-titles-and-space-around-borders
size = 4

source = np.zeros((size,size,3), dtype=np.uint8)
source[1,:] = [255,255,255]

canvas = np.zeros((size,size,3), dtype=np.uint8)
canvas[0,0] = [255,0,0]

print(source)
source_img = Image.fromarray(source)
canvas_img = Image.fromarray(canvas)
# img = img.resize((16,16))
source_img = np.uint8(source_img)
canvas_img = np.uint8(canvas_img)

# cv2.imshow("pr", source_img)
# cv2.waitKey()
#
# plt.subplot(1, 2, 1), plt.imshow(source_img, 'gray')
# plt.subplot(1, 2, 2), plt.imshow(canvas_img, 'gray')
# plt.show()
#
# prev_idx = 0
# for i in range(size):
#     canvas[prev_idx, 0] = [0,0,0]
#     canvas[i,0] = [255,0,0]
#     canvas_img = Image.fromarray(canvas)
#     canvas_img = np.uint8(canvas_img)
#     plt.subplot(1, 2, 1), plt.imshow(source_img, 'gray')
#     plt.subplot(1, 2, 2), plt.imshow(canvas_img, 'gray')
#     plt.show()
#     prev_idx = i


# img = cv2.imread("/Users/anmoluppal/Downloads/Lenna.png")

height, width, ch = source_img.shape
print(height, width, ch)
new_width, new_height = width+1 , height + 2# width + width//20, height + height//8
print(new_width, new_height)

# Crate a new canvas with new width and height.
source_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125
canvas_background = np.ones((new_height, new_width, ch), dtype=np.uint8) * 125

# New replace the center of canvas with original image
padding_top, padding_left = 1, 0# 60, 10

source_background[padding_top:padding_top + height, padding_left:padding_left + width] = source_img
text1 = "Source image"
text2 = "Canvas"
text_color_list = np.array([255, 0, 0])
text_color = (int(text_color_list[0]), int(text_color_list[1]), int(text_color_list[2]))
img1 = cv2.putText(source_background.copy(), text1, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, text_color)

prev_idx = 0
for i in range(size):
    canvas[prev_idx, 0] = [0,0,0]
    canvas[i,0] = [255,0,0]
    canvas_img = Image.fromarray(canvas)
    canvas_img = np.uint8(canvas_img)
    canvas_background[padding_top:padding_top + height, padding_left:padding_left + width] = canvas_img
    img2 = cv2.putText(canvas_background.copy(), text2, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, text_color)

    final = cv2.hconcat((img1, img2))
    print(final.shape)
    # shape[1] is the width, it seems it needs to go first when resizing.
    final = cv2.resize(final, (final.shape[1] * 30, final.shape[0] * 30), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite("./debug.png", final)
    final = cv2.putText(final.copy(), text2, (int(0.25 * width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, text_color)
    cv2.imshow("pr", final)
    cv2.waitKey(200)
    prev_idx = i

cv2.destroyAllWindows()
exit(4)
if padding_top + height < new_height and padding_left + width < new_width:
    source_background[padding_top:padding_top + height, padding_left:padding_left + width] = source_img
    canvas_background[padding_top:padding_top + height, padding_left:padding_left + width] = canvas_img
else:
    print("The Given padding exceeds the limits.")

text1 = "Source image"
text2 = "Canvas"
color_list = np.array([255, 0, 0])
color = (int(color_list[0]), int(color_list[1]), int(color_list[2]))
img1 = cv2.putText(source_background.copy(), text1, (int(0.25*width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, color)
img2 = cv2.putText(canvas_background.copy(), text2, (int(0.25*width), 30), cv2.FONT_HERSHEY_COMPLEX, 1, color)

final = cv2.hconcat((img1, img2))
print(final.shape)
# shape[1] is the width, it seems it needs to go first when resizing.
final = cv2.resize(final, (final.shape[1]*30, final.shape[0]*30), interpolation=cv2.INTER_NEAREST)
#cv2.imwrite("./debug.png", final)
cv2.imshow("pr", final)
cv2.waitKey()
cv2.destroyAllWindows()

exit(5)
source[1] = 1
print(source)

img = Image.fromarray(source)
img = img.resize((300,300))

cv2.imshow("prov", img)
cv2.waitKey()