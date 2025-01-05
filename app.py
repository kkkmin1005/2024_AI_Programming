from flask import Flask, Response, render_template, jsonify, request, redirect, url_for
import cv2
import torch
import torchvision.transforms as transforms
from models.modules.patchEmbedding import PatchEmbedding
from models.modules.positionalEncoding import PositionalEncoding
from models.modules.transformer import TransformerBlock
import torch.nn as nn

app = Flask(__name__)

# 웹캠 초기화
camera = cv2.VideoCapture(0)

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
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)

# 모델 설정 및 초기화
MODEL_PATH = r'models_file\model2.pth'
image_size = 48
patch_size = 8
num_classes = 7
dim = 128
depth = 6
heads = 8
mlp_dim = 256
dropout = 0.1

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
emotion_model = VisionTransformer(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout).to(device)
emotion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
emotion_model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 글로벌 변수
emotion_log = []
final_rating = 0

# 프레임 생성 및 감정 탐지 함수
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_tensor = transform(face).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = emotion_model(face_tensor)
                    _, emotion_index = torch.max(outputs, 1)
                    emotion = emotion_labels[emotion_index.item()]

                # 얼굴 영역 및 감정 표시
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # 프레임을 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_emotion')
def current_emotion():
    success, frame = camera.read()
    if not success:
        return jsonify({'emotion': 'error'})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    if len(faces) == 0:
        return jsonify({'emotion': 'neutral'})  # 얼굴이 감지되지 않으면 neutral 반환

    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]
    face_tensor = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = emotion_model(face_tensor)
        _, emotion_index = torch.max(outputs, 1)
        emotion = emotion_labels[emotion_index.item()]

    return jsonify({'emotion': emotion})

@app.route('/end_movie', methods=['POST'])
def end_movie():
    global emotion_log, final_rating
    data = request.json
    emotion_log = data.get('emotion_log', [])
    final_rating = float(data.get('rating', 0))  # 평점 저장
    return redirect(url_for('review_page'))


@app.route('/review')
def review_page():
    global emotion_log, final_rating
    movie_review = generate_movie_review(emotion_log)
    return render_template('review.html', rating=final_rating, review=movie_review)


def generate_movie_review(emotion_log):
    if not emotion_log:
        return "No significant emotional changes were observed during the movie."
    review = "The movie took you through an emotional journey, transitioning from "
    transitions = [f"{change['from']} to {change['to']}" for change in emotion_log]
    review += ", ".join(transitions) + "."
    return review


if __name__ == '__main__':
    app.run(debug=True)
