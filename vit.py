import cv2
import torch
import torchvision.transforms as transforms
from models.modules.patchEmbedding import PatchEmbedding
from models.modules.positionalEncoding import PositionalEncoding
from models.modules.transformer import TransformerBlock
from PIL import Image
import torch.nn as nn


# Vision Transformer 모델 정의
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim)
        self.positional_encoding = PositionalEncoding(self.patch_embedding.num_patches, dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, _, _, _ = x.shape

        # Patch Embedding + Positional Encoding
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.positional_encoding(x)

        # Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Classification Head
        cls_output = x[:, 0]  # CLS token만 사용
        return self.mlp_head(cls_output)


# 디바이스 설정 (GPU 사용 가능 시 GPU로)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 초기화 및 가중치 로드
MODEL_PATH = r'models_file\model2.pth'
image_size = 48
patch_size = 8
num_classes = 7  # 감정 클래스 수
dim = 128  # Transformer 차원
depth = 6  # Transformer 레이어 수
heads = 8  # Multi-head Attention
mlp_dim = 256  # MLP Hidden Layer 크기
dropout = 0.1

# 모델 생성
emotion_model = VisionTransformer(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout).to(device)
emotion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
emotion_model.eval()

# 감정 레이블 (모델의 출력 순서에 따라 수정)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Haar Cascade 파일 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Haar Cascade 파일을 로드할 수 없습니다.")
    exit()

# 이미지 전처리 (PyTorch 모델 입력 형식)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),  # 모델 입력 크기에 맞게 조정
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
])

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠이 시작되었습니다. ESC 키를 눌러 종료하세요.")

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
        # 얼굴 영역 추출 및 RGB 변환
        face = frame[y:y+h, x:x+w]
        face_tensor = transform(face).unsqueeze(0).to(device)  # 배치 차원 추가
        
        # 감정 예측
        with torch.no_grad():
            outputs = emotion_model(face_tensor)
            _, emotion_index = torch.max(outputs, 1)
            emotion = emotion_labels[emotion_index.item()]

        # 얼굴 영역 및 감정 레이블 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # 결과 프레임 표시
    cv2.imshow('Emotion Detection', frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        print("프로그램을 종료합니다.")
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
