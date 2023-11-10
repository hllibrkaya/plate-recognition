import cv2
import imutils
import numpy as np
import pytesseract

def preprocess_image(image_path, resize_width=600, resize_height=400):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (resize_width, resize_height))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)

    edged = cv2.Canny(gray, 30, 200)
    return img, gray, edged

def find_license_plate(edged):
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    return screenCnt

def main():
    image_path = 'example.jpg'

    img, gray, edged = preprocess_image(image_path)
    screenCnt = find_license_plate(edged)

    if screenCnt is None:
        print("No contour detected")
        return

    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape, np.uint8)
    plate = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    plate = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    crop = gray[topx:bottomx + 1, topy:bottomy + 1]

    text = pytesseract.image_to_string(crop, config='--psm 6')
    print("Plate Number:", text)

    cv2.imshow('Plate', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
