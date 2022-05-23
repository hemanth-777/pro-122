import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time
import cv2 as cv

X= np.load("image.npz")["arr_0"]
Y= pd.read_csv("labels.csv")["labels"]

print(pd.Series(Y).value_counts())

if(not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)
samples_perclass = 5
figure = plt.figure(figsize=(nclasses*2, (1+samples_perclass*2)))

iclass=0
for cls in classes:
  id= np.flatnonzero(Y==cls)
  id= np.random.choice(id, samples_perclass, replace=False)
  i=0
  for j in id:
      plt_j=i*nclasses+iclass+1
      p=plt.subplot(samples_perclass, nclasses, plt_j)
      p= sb.heatmap(np.reshape(X[j], (22,30)), cmap=plt.cm.Blues, xticklabels=False, yticklabels = False, cbar=False)
      i=i+1
  iclass= iclass+1

print("The total number of letters is ", len(X))
print("Pixel Size of each image: ", len(X[1]))

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.25, random_state=0)

Xtrainscale = Xtrain/255.0

Xtestscale= Xtest/255.0
classifier = LogisticRegression(solver="saga", multi_class="multinomial").fit(Xtrainscale, Ytrain)
Ypredict= classifier.predict(Xtestscale)
accuracy=accuracy_score(Ytest, Ypredict)
print("The accuracy of this model was", accuracy)

videocode = cv.VideoWriter_fourcc(*'XVID')
outputfile = cv.VideoWriter("video.avi",videocode,20.0,(640,480))
cap= cv.VideoCapture(0)
for i in range(0,60):
    ret,bg = cap.read()
while(True):

  try:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    height, width = gray.shape
    upper_left= (int(width/2)-56, int(height/2)-56)
    bottom_right= (int(width/2)+56, int(height/2)+56)
    cv.rectangle(gray, upper_left, bottom_right, (0,255,0), 2)

    roi= gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    im_pil= Image.fromarray(roi)

    print("E")
    image_bw = im_pil.convert('L')

    image_bw_resized = image_bw.resize((28,28))

    image_bw_resized_inverted= PIL.ImageOps.invert(image_bw_resized)
    pixel_filter=20

    min_pixel= np.percentile(image_bw_resized_inverted, pixel_filter)

    image_bw_resized_inverted_scaled= np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel= np.max(image_bw_resized_inverted)


    image_bw_resized_inverted_scaled= np.asarray(image_bw_resized_inverted_scaled)/max_pixel

    test_sample= np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
    prediction= classifier.predict(test_sample)
  
    print("Predicted letter is", prediction)

    cv.imshow("frame", gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

  except Exception as e:
    pass

cap.release()
cv.destroyAllWindows()