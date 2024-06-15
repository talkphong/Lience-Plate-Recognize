import os
# from pathlib import Path
# import tensorflow as tf
from ultralytics import YOLO #YOLOv8
import cv2
import numpy as np
import imutils
from imutils import perspective
from skimage.filters import threshold_local
from keras.models import load_model

model = YOLO("YOLO-Weight/best.pt")

# model_detect_text = load_model("CNN-Weight/alexnet_model.h5")
model_detect_text = load_model("CNN-Weight/alexnet_model.h5")


def recognize(image):
    results = model(image, stream=True)
    for i in results:
        obj_pos = i.boxes
        for box in obj_pos:
            x1, y1, x2, y2 = box.xyxy[0]
            xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)
    
    cv2.rectangle(image, (xmin-2, ymin-2), (xmax+2, ymax+2), (0, 255, 0), 2)
    cv2.imshow('origin',image)
    coord = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    LpRegion = perspective.four_point_transform(image, coord)
    # cv2.imshow('before',LpRegion)
    # LpRegion = rotate_and_crop(LpRegion)
    # cv2.imshow('after',LpRegion)

    image = LpRegion.copy()
    cv2.imwrite(r'C:\Users\Admin\Desktop\Nhan dien bien so xe\YOLOv8_CNN_final\cut.jpg', image)
    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
    # adaptive threshold
    T = threshold_local(V, 35, offset=5, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
    thresh = imutils.resize(thresh, width=600)

    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")
    total_pixels = thresh.shape[0] * thresh.shape[1]
    lower = total_pixels // 90
    upper = total_pixels // 20
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)

    cnts, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    boundingBoxes = np.array(boundingBoxes)
    mean_w = np.mean(boundingBoxes[:, 2])
    mean_h = np.mean(boundingBoxes[:, 3])
    mean_y = np.mean(boundingBoxes[:,1])
    threshold_w = mean_w * 1.5
    threshold_h = mean_h * 1.5
    new_boundingBoxes = boundingBoxes[(boundingBoxes[:, 2] < threshold_w) & (boundingBoxes[:, 3] < threshold_h)]
    line1 = []
    line2 = []
    for box in new_boundingBoxes:
        x,y,w,h = box
        if y > mean_y * 1.2:
            line2.append(box)
        else:
            line1.append(box)

    line1 = sorted(line1, key=lambda box: box[0])
    line2 = sorted(line2, key=lambda box: box[0])
    boundingBoxes = line1+line2

    img_with_boxes = imutils.resize(image.copy(), width=600)
    image = imutils.resize(image.copy(), width=600)
    for bbox in boundingBoxes:
        x, y, w, h = bbox
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    cv2.imshow('test',img_with_boxes)

    # Character Recognition

    chars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z'
    ]

    vehicle_plate = ""
    characters = []

    char = 1
    for rect in boundingBoxes:
        x, y, w, h = rect

        character = mask[y:y+h, x:x+w]
        character = cv2.bitwise_not(character)
        rows = character.shape[0]
        columns = character.shape[1]
        paddingY = (128 - rows) // 2 if rows < 128 else int(0.17 * rows)
        paddingX = (
            128 - columns) // 2 if columns < 128 else int(0.45 * columns)
        character = cv2.copyMakeBorder(character, paddingY, paddingY,
                                    paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

        character = cv2.cvtColor(character, cv2.COLOR_GRAY2RGB)
        character = cv2.resize(character, (227, 227))
        
        character = character.astype("float") / 255.0
        cv2.imshow('character'+str(char),character)
        char += 1
        characters.append(character)
    characters = np.array(characters)
    probs = model_detect_text.predict(characters)
    for prob in probs:
        idx = np.argsort(prob)[-1]
        vehicle_plate += chars[idx]
    return vehicle_plate

def format_string(string): #59L206377 to 59-L2 06377
    count = 1
    result = ''
    for char in string:
        result += char
        if count == 2:
            result += '-'
        if count == 4:
            result += ' '
        count += 1
    return result



def main():
    # Doc Bien So
    folder_path = 'data/test'
    # folder_path = 'data/train/xe may'
    files = os.listdir(folder_path)
    for file_name in files:
        img = cv2.imread(os.path.join(folder_path, file_name))
        string = recognize(img)
        string = format_string(string)
        print("Ket qua doc bien so xe:\n" + string)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()








if __name__ == "__main__":
    main()