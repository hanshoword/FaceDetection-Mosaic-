import cv2
import dlib
import numpy as np

sizeIndex = 0.3

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def ReSizeImage(frame):
    return cv2.resize(frame, None, fx=sizeIndex, fy=sizeIndex, interpolation= cv2.INTER_AREA)

cap = cv2.VideoCapture('face.mp4')

while True:
  # 프레임을 읽습니다.
  ret, frame = cap.read()
  if ret == True:
    frame = ReSizeImage(frame)

    temp = frame
    origin = frame

    # 얼굴인식
    faces = detector(frame)

    # 여러 얼굴 중 대표 얼굴을 하나 지정
    face = faces[0]

    # 학습된 데이터를 이용하여 얼굴의 특징점을 찾음
    dlib_shape = predictor(frame, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    topLeft = np.min(shape_2d, axis=0)
    bottomRight = np.max(shape_2d, axis=0)

    topRight = np.array([bottomRight[0], topLeft[1]])
    bottomLeft = np.array([topLeft[0], bottomRight[1]])


    cv2.imshow('original', frame)
   # cv2.circle(frame, center=tuple(topLeft), radius=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
   # cv2.circle(frame, center=tuple(topRight), radius=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
   # cv2.circle(frame, center=tuple(bottomLeft), radius=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
   # cv2.circle(frame, center=tuple(bottomRight), radius=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    height, width, channel = frame.shape

    srcPoint = np.array([[(topLeft[0], topLeft[1]), (topRight[0], topRight[1]), (bottomRight[0], bottomRight[1]) , (bottomLeft[0], bottomLeft[1])]], dtype=np.float32)
    dstPoint = np.array([[(0,0),(width,0),(width,height),(0, height)]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
    origin_befROI = cv2.getPerspectiveTransform(dstPoint, srcPoint)

    ROI = cv2.warpPerspective(origin, matrix, (width, height))
    blur = cv2.medianBlur(ROI, 55)
    unwarp = cv2.warpPerspective(blur, origin_befROI, (width, height))

    origin = cv2.rectangle(origin, pt1=(topLeft[0], topLeft[1]), pt2=(bottomRight[0]-1, bottomRight[1]-1), color = (0, 0, 0), thickness = -1)
    final = cv2.addWeighted(origin, 1.0, unwarp, 1.0, 0.0)

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