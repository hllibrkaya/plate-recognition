# Vehicle License Plate Recognition

This project involves a Python script for detecting license plates in images. The main steps of the project include:

1. **Image Preprocessing:** The image is prepared with resizing and conversion to grayscale.
2. **Edge Detection:** Canny edge detection is applied to identify the edges of the image.
3. **Contour Detection:** Contours are found on the edges.
4. **Isolating the Plate Region:** A mask is created to isolate the license plate region, and this mask is then applied to the image.
5. **Optical Character Recognition (OCR):** Tesseract OCR is used to extract text from the plate region, obtaining the license plate number.
6. **Displaying Results:** A window is created to visualize the algorithm's results, with the license plate number printed on the screen.

## Future Updates
This project will be updated with machine learning-based approaches. Please check for the latest features and improvements.

## Requirements

Ensure you have the following Python libraries installed:

- OpenCV
- imutils
- pytesseract
- numpy

You can install these libraries using the following command:

```bash
pip install opencv-python imutils pytesseract numpy


