from four_point_transform import four_point_transform
import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import morphology, label
import pytesseract
from wrapper import *
from random import randint
#from keras.preprocessing.image import img_to_array
#from keras.models import load_model
#import pickle
#import tensorflow as tf

#sess = tf.Session()
#saver = tf.train.import_meta_graph('hcr_linear_acc_91_3.meta')
#saver.restore(sess,tf.train.latest_checkpoint('./'))


# load the model
#model = load_model('/users/kushagrapandey/Desktop/Github/CalHacks/calhacks.model')
#lb = pickle.loads(open('lb.pickle', "rb").read())

image = cv2.imread("sketchy_test.jpg")
cv2_im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
pil_im = Image.fromarray(cv2_im)

ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
def autocanny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0-sigma)*v))
    upper = int(min(255, (1.0+sigma)*v))
    edged = cv2.Canny(image, lower, upper)
    return edged

edged = autocanny(gray)
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx)==4:
        screenCnt = approx
        break
cv2.drawContours(image, [screenCnt], -1, (0,255,0),2)
warped = four_point_transform(orig, screenCnt.reshape(4,2)*ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset=10, method="gaussian")
warped = (warped>T).astype("uint8")*255
height, width = warped.shape
warped = warped[int(0.05*height): int(0.95*height), int(0.01*width): int(0.99*width)]
#gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(warped, (5,5), 0)
edged = autocanny(gray)
warped = cv2.cvtColor(warped,cv2.COLOR_GRAY2RGB)
#kernel = np.ones((1,1),np.uint8)
#edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
kernel = np.ones((3,3),np.uint8)
edged = cv2.dilate(edged, kernel, iterations=1)
#kernel = np.ones((2,2),np.uint8)
#edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
thresh_area = 50
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
pil_im = Image.fromarray(warped)

#model = load_model('calhacks.model')
#lb = pickle.loads(open('lb.pickle', "rb").read())
class Line():
    box=False
    def __init__(self, coordinates):
        self.coordinates = coordinates
class Box():
    box=True
    def __init__(self, coordinates):
        self.coordinates = coordinates
def boxes(orig):
    img = ImageOps.grayscale(orig)
    im = np.array(img)

    # Inner morphological gradient.
    im = morphology.grey_dilation(im, (3, 3)) - im

    # Binarize.
    mean, std = im.mean(), im.std()
    t = mean + std
    im[im < t] = 0
    im[im >= t] = 1

    # Connected components.
    lbl, numcc = label(im)
    # Size threshold.
    min_size = 200 # pixels
    box = []
    print("Number of connected components is: " + str(numcc))
    for i in range(1, numcc + 1):
        py, px = np.nonzero(lbl == i)
        if len(py) < min_size:
            im[lbl == i] = 0
            continue

        xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()
        # Four corners and centroid.
        if (ymax-ymin) * (xmax-xmin) < 200000:
            continue
        box.append([
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            (np.mean(px), np.mean(py))])

    print("finished running boxes!")
    return im.astype(np.uint8) * 255, box
max_area = 0


orig = pil_im
im, box = boxes(orig)
for b, centroid in box:
    area = (b[2][1] - b[0][1]) * (b[2][0] - b[0][0])
    if area>max_area:
        max_area = area
print("Max area: "+str(max_area))
img = Image.fromarray(im)
visual = img.convert('RGB')
draw = ImageDraw.Draw(visual)
for b, centroid in box:
    draw.line(b + [b[0]], fill='yellow')
    cx, cy = centroid
    draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill='red')
opencvImage = cv2.cvtColor(np.array(visual), cv2.COLOR_RGB2BGR)


cv2.imwrite("testImage.png", opencvImage)

cv2.imwrite("edged.png", edged)
def sorted(cnts, method="left-to-right"):
    reverse = False
    i=0
    if method == "right-to-left" or method=="bottom-to-top":
        i=1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes

elements = []
for c in cnts:
    rect = cv2.minAreaRect(c)
    min_box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    min_box = np.int0(min_box)
    min_x = min(p[0] for p in min_box)
    max_x = max(p[0] for p in min_box)
    min_y = min(p[1] for p in min_box)
    max_y = max(p[1] for p in min_box)
    flag = True
    if cv2.contourArea(c)>100:
        for b, centroid in box:
            if min_x <= b[1][0] and max_x <= b[1][0] and min_x >= b[0][0] and max_x>=b[0][0] and min_y <= b[2][1] and max_y<= b[2][1] and min_y>= b[0][1] and max_y>=b[0][1]:
                flag=False
        if flag:
            if (max_x-min_x)>450 and (max_y-min_y)<= 100:
                cv2.drawContours(warped, [min_box], -1, (0,255,0),3)
                elements.append(Line([min_x, max_x, min_y, max_y]))
                print("found")
    else:
        continue
for b, centroid in box:
    cv2.rectangle(warped, (b[0][0], b[0][1]), (b[1][0], b[2][1]), (0,255,0))
    elements.append(Box([b[0][0], b[1][0], b[0][1], b[3][1]]))
elements.sort(key=lambda element: element.coordinates[2])
toReturn = []
i=0
while i<len(elements):
    if not elements[i].box:
        img = warped[elements[i].coordinates[3] - 300 :elements[i].coordinates[3], 0:elements[i].coordinates[1]]
        pillow = Image.fromarray(img)
        text = pytesseract.image_to_string(pillow)
        for char in text:
            if not char.isalnum():
                text = text.replace(char,"")
        print(text)
        toReturn.append(["HEAD",text])
        i+=1
    else:
        j=i
        while elements[j].box and j!=len(elements)-1:
            j+=1
        if not elements[j].box:
            if i+2==j:
                min_y = max(elements[i].coordinates[3], elements[i+1].coordinates[3])
                img = warped[min_y:elements[j].coordinates[2]-300 , 0:warped.shape[:2][1]]
                cv2.imwrite("passed_in1.png", img)
                pillow = Image.fromarray(img)
                text = pytesseract.image_to_string(pillow)
                img1 = warped[elements[i].coordinates[2]: elements[i].coordinates[3],elements[i].coordinates[0]:elements[i].coordinates[1]]
                x = randint(1, 1000000)
                cv2.imwrite(str(x)+ ".jpg", img1)
                upload_file(str(x)+ ".jpg") #generate large number
                url1 = get_base_url()+get_id_from_title(str(x)+ ".jpg")
                img2 = warped[elements[i+1].coordinates[2]: elements[i+1].coordinates[3], elements[i+1].coordinates[0]:elements[i+1].coordinates[1]]
                x = randint(1, 1000000)
                cv2.imwrite(str(x)+ ".jpg", img2)
                upload_file(str(x)+ ".jpg") #generate large number
                url2 = get_base_url()+get_id_from_title(str(x)+ ".jpg")
                toReturn.append(["IMAGE", [url1, url2]])
                toReturn.append(["TEXT", text])
                print(text)
                i+=2
            elif i+1==j:
                img = warped[elements[i].coordinates[3]:elements[j].coordinates[2]-300 , 0:warped.shape[:2][1]]
                pillow = Image.fromarray(img)
                text = pytesseract.image_to_string(pillow)
                img1 = warped[elements[i].coordinates[2]: elements[i].coordinates[3],elements[i].coordinates[0]:elements[i].coordinates[1]]
                x = randint(1, 1000000)
                cv2.imwrite(str(x)+ ".jpg", img1)
                upload_file(str(x)+ ".jpg") #generate large number
                url = get_base_url()+get_id_from_title(str(x)+ ".jpg")
                toReturn.append(["IMAGE", [url]])
                toReturn.append(["TEXT", text])
                print(text)
                i+=1
        else:
            if i<j:
                min_y = max(elements[i].coordinates[3], elements[i+1].coordinates[3])
                img = warped[min_y:warped.shape[:2][0], 0:warped.shape[:2][1]]
                cv2.imwrite("passed_in.png", img)
                pillow = Image.fromarray(img)
                text = pytesseract.image_to_string(pillow)
                img1 = warped[elements[i].coordinates[2]: elements[i].coordinates[3], elements[i].coordinates[0]:elements[i].coordinates[1]]
                x = randint(1, 1000000)
                cv2.imwrite(str(x)+ ".jpg", img1)
                upload_file(str(x)+ ".jpg") #generate large number
                url1 = get_base_url()+get_id_from_title(str(x)+ ".jpg")
                img2 = warped[elements[i+1].coordinates[2]: elements[i+1].coordinates[3], elements[i+1].coordinates[0]:elements[i+1].coordinates[1]]
                x = randint(1, 1000000)
                cv2.imwrite(str(x)+ ".jpg", img2)
                upload_file(str(x)+ ".jpg") #generate large number
                url2 = get_base_url()+get_id_from_title(str(x)+ ".jpg")
                toReturn.append(["IMAGE", [url1, url2]])
                toReturn.append(["TEXT", text])
                print(text)
                i+=2
            else:
                img = warped[elements[i].coordinates[3]:warped.shape[:2][0], 0:warped.shape[:2][1]]
                cv2.imwrite("passed_in.png", img)
                pillow = Image.fromarray(img)
                text = pytesseract.image_to_string(pillow)
                img1 = warped[elements[i].coordinates[2]: elements[i].coordinates[3],elements[i].coordinates[0]:elements[i].coordinates[1]]
                x = randint(1, 1000000)
                cv2.imwrite(str(x)+ ".jpg", img1)
                upload_file(str(x)+ ".jpg") #generate large number
                url = get_base_url()+get_id_from_title(str(x)+ ".jpg")
                toReturn.append(["IMAGE", [url]])
                toReturn.append(["TEXT", text])
                print(text)
                i+=1
print(toReturn)
f = open("toReturn.txt", "w+")
f.write(str(toReturn))
f.close()
#cv2.imwrite("edged.png", edged)

cv2.imwrite("warped.png", warped)
#cv2.waitKey(0)
