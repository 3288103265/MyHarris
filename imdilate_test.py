import numpy as np
import cv2

src = np.random.randn(0, 255, (350, 350))
cv2.imshow("src", src)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dst = cv2.dilate(src, kernel)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()
