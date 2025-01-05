import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# 모델 파일 경로
MODEL_PATH = r'models_file\final_trained_model.keras'

# 모델 로드
if not os.path.exists(MODEL_PATH):
    print("모델 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

emotion_model = load_model(MODEL_PATH)

# 감정 레이블 (모델의 출력 순서에 따라 수정)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'sleepy', 'surprise']

# 얼굴 검출을 위한 OpenCV Haar Cascade 파일
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Haar Cascade 파일을 로드할 수 없습니다.")
    exit()

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠을 읽을 수 없습니다.")
        break

    # 그레이스케일로 변환 (얼굴 검출 용이)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face = gray_frame[y:y+h, x:x+w]
        
        # 모델 입력 크기에 맞게 조정 (48x48)
        face_resized = cv2.resize(face, (48, 48))
        
        # 회색조 이미지를 RGB로 변환 (3채널로 변환)
        face_color = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        
        # 정규화 (0 ~ 1 범위)
        face_normalized = face_color / 255.0
        
        # 배치 차원 추가 (모델 입력 형태로 변환)
        face_reshaped = np.expand_dims(face_normalized, axis=0)  # shape: (1, 48, 48, 3)
        
        # 입력 데이터 형태 확인 (디버깅용)
        # print("입력 데이터 형태:", face_reshaped.shape)
        # print("모델 기대 입력 형태:", emotion_model.input_shape)
        
        try:
            # 감정 예측
            predictions = emotion_model.predict(face_reshaped)
            emotion_index = np.argmax(predictions)
            
            # 레이블 매핑에 따라 감정 표시
            if emotion_index < len(emotion_labels):
                emotion = emotion_labels[emotion_index]
            else:
                emotion = "Unknown"  # 예외 처리: 범위 초과 시
        except Exception as e:
            emotion = "Error"
            print(f"예측 중 오류 발생: {e}")

        # 얼굴 영역 및 감정 레이블 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 결과 프레임 표시
    cv2.imshow('Emotion Detection', frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
