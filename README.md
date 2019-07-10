## FaceDetection[Mosaic]

[Youtube Link](https://www.youtube.com/watch?v=nl1kKhY4DQQ)

<img src = "https://user-images.githubusercontent.com/47768726/60770601-8f2e3280-a117-11e9-9e33-afdceefc1fc6.JPG" width = "70%" height = "60%"></img>
<img src = "https://user-images.githubusercontent.com/47768726/60770602-8f2e3280-a117-11e9-85c9-4cade923c42d.JPG" width = "70%" height = "60%"></img>


### import 라이브러리

```c
import cv2
import dlib
import numpy as np
```

```
먼저 이미지 처리를위한 opencv2와

얼굴인식을 하기위해 dlib을 사용합니다.

그리고 배열처리를 위해 numpy를 import했습니다.
```

### 이미지를 축소하거나 확대하는 함수

```
sizeIndex = 0.3

def ReSizeImage(frame):
    return cv2.resize(frame, None, fx=sizeIndex, fy=sizeIndex, interpolation= cv2.INTER_AREA)

cv2.resize를 통하여 이미지를 sizeIndex만큼 축소시키거나 확대시킵니다.

```

### 얼굴인식 전처리
```
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

먼저 얼굴을 찾아줄 detector를 dlib을 통해 정의하고 얼굴의 62개의 특징점을 찾아줄 dat를 받아와 predictor에 넣어줍니다.
```


### Main 함수
```
cap = cv2.VideoCapture('face.mp4')

동영상을 cap 변수에 받아옵니다
```

```
while True:
  # 프레임을 읽습니다.
  ret, frame = cap.read()
  if ret == True:
  
프레임과 ret(return)을 받아와 프레임이 존재한다면 메인함수를 처리합니다.
```
   
    # 이미지를 축소
    frame = ReSizeImage(frame)
    
    # 먼저 처리하기 이전에 이미지를 저장
    temp = frame
    origin = frame

    # 얼굴인식
    faces = detector(frame)

    # 여러 얼굴 중 대표 얼굴을 하나 지정
    face = faces[0]

    # 학습된 데이터를 이용하여 얼굴의 특징점을 찾음
    dlib_shape = predictor(frame, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # 찾은 얼굴의 4점 좌표를 계산합니다
    topLeft = np.min(shape_2d, axis=0)
    bottomRight = np.max(shape_2d, axis=0)

    topRight = np.array([bottomRight[0], topLeft[1]])
    bottomLeft = np.array([topLeft[0], bottomRight[1]])

    # original 이미지 출력
    cv2.imshow('original', frame)

    # 이미지의 높이 너비 채널의수를 받아옵니다.
    height, width, channel = frame.shape
    
   
    # 찾아낸 얼굴만 잘라 모자이크 처리하기위해 기하학적 변환인 warp변환을 통해 얼굴 이미지만 남도록 합니다.
    srcPoint = np.array([[(topLeft[0], topLeft[1]), (topRight[0], topRight[1]), (bottomRight[0], bottomRight[1]) , (bottomLeft[0], bottomLeft[1])]], dtype=np.float32)
    dstPoint = np.array([[(0,0),(width,0),(width,height),(0, height)]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    origin_befROI = cv2.getPerspectiveTransform(dstPoint, srcPoint)

    # 관심영역(얼굴 이미지)에 medianBlur 처리(55x55 필터에서 픽셀 중앙값으로 대체)를 통해 모자이크 처리합니다.
    ROI = cv2.warpPerspective(origin, matrix, (width, height))
    blur = cv2.medianBlur(ROI, 55)
    
    # 다시 이미지를 펼칩니다.
    # 펼친 이미지에는 모자이크된 얼굴만 있고 나머지는 검은색 배경입니다.
    unwarp = cv2.warpPerspective(blur, origin_befROI, (width, height))

    # original과 unwarp을 합치기위해 original얼굴부분을 검은색으로 처리하고
    # 두 이미지를 addweighted를 통해 합쳐줍니다.
    origin = cv2.rectangle(origin, pt1=(topLeft[0], topLeft[1]), pt2=(bottomRight[0]-1, bottomRight[1]-1), color = (0, 0, 0), thickness = -1)
    final = cv2.addWeighted(origin, 1.0, unwarp, 1.0, 0.0)

    # 마지막 모자이크된 이미지를 출력시킵니다.
    cv2.imshow('blur', final)

    # 검출 영역 표시
    # temp = cv2.rectangle(temp, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color = (0, 0, 255), thickness = 3, lineType=cv2.LINE_AA)

    # 특징점 표시
    #for s in shape_2d:
    #    cv2.circle(temp, center=tuple(s), radius = 1, color = (255,255,255), thickness = 2, lineType=cv2.LINE_AA)
    #cv2.imshow('FaceDetection', temp)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
     else:
        break

    cap.release()
    cv2.destroyAllWindows()
```
