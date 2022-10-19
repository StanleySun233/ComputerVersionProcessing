import cv2

src = cv2.imread('./data/fig2.jpg')
cv2.imshow('origin', src)

dst = cv2.GaussianBlur(src, (15, 15), 0)
cv2.imshow("blur", dst)

cv2.waitKey(0)