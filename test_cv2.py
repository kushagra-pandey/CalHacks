import cv2
import numpy as np
import imutils
img = cv2.imread("hand_written_notes.jpg")
img2 = img.copy()
boundaries = [([200, 200, 200], [230,230,230])]
def autocanny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0-sigma)*v))
    upper = int(min(255, (1.0+sigma)*v))
    edged = cv2.Canny(image, lower, upper)
    return edged
#gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray, (5,5), 0)
#edged = autocanny(gray)
for (lower, upper) in boundaries:
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(img2, lower, upper)
output = cv2.bitwise_and(img2, img2, mask=mask)
cv2.imwrite("edged.png", output)
#cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#cnts = cnts[0] if imutils.is_cv2() else cnts[1]
#cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
#for c in cnts:
    #rect = cv2.minAreaRect(c)
    #box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    #box = np.int0(box)
    #cv2.drawContours(img2, [box], -1, (0,255,0),3)
#cv2.imwrite("bounding_boxes.png", img2)
