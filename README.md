## 표정 분류 모델을 이용한 영화 평점 측정 서비스

## 제안 배경
제작사 팬덤 등의 개입으로 인한 영화 평점의 신뢰도 하락으로 인하여 객관성있는 평점 측정 시스템을 도입하기 위함

## 목표
관객의 실시간 표정 분석으로 감정 변화 및 반응 평가  
특정 장면에서 감독의 의도와 관객의 감정이 얼마나 일치하는지 확인  

## 데이터 수집
AI hub - 한국인 감정인식을 위한 복합 영상

## 최종 모델
ViT(자체적으로 구현)
### 주요 구현 사항
- Transformer 구조
- Patch Embedding
- Multi-Head Self-Attention
### Parameters
![스크린샷 2024-12-15 161751](https://github.com/user-attachments/assets/7ef4dcb5-ded3-4621-9b4f-5dd0362b348b)
### Score
![스크린샷 2024-12-15 165056](https://github.com/user-attachments/assets/c4acbd5c-1dc5-4542-a091-e2efd3aea6b5)

## 실제 서비스

### Skills
![image](https://github.com/user-attachments/assets/b76df2a3-b6fe-4a70-8089-f8cfc178ba89)

### Workflow
![image](https://github.com/user-attachments/assets/45bcfe21-a17e-4471-a37e-f7686f2b613f)

### 실제 서비스 화면
![image](https://github.com/user-attachments/assets/68d877d3-6c50-4dc3-b28f-72060f629a91)
![image](https://github.com/user-attachments/assets/72a86cfa-f5bc-4800-a08f-61c5485189ae)
![image](https://github.com/user-attachments/assets/ecbe850a-0441-41af-a980-19c51beeeeee)

## 추가할 점
빛, 안경, 배경색 등 외적 요인이 모델에 미치는 영향 줄이기
-> 데이터셋 증강, face detection, style transfer 등 활용
