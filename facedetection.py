import cv2 as cv


img=cv.imread('photos/fam1.jpeg')
cv.imshow('srk',img)                              #image open


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)          #grayscale an image
#cv.imshow('mark1gray',gray)


haarcascade=cv.CascadeClassifier('haarfrontface.xml')           #haarcascade classifier


face_cord = haarcascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

for(a,b,c,d) in face_cord:
    cv.rectangle(img,(a,b),(a+c,b+d),(0,250,250), thickness=2)
    
cv.imshow('DETECTION',img)

print(f'Number of person found = {len(face_cord)}')

cv.waitKey(0)