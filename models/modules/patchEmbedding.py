import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        # 패치 하나당 특성 수: patch_size * patch_size * C(RGB=3)
        # 여기서는 모델 입력 채널 수를 C라고 할 때
        # projection 레이어 초기화는 모델 코드에 따라 조정
        self.dim = dim
        # 프로젝션 레이어: 패치 벡터를 dim차원으로 매핑
        # 입력은 patch_size * patch_size * C 이므로, 실행 시점에 C 획득 필요
        # 이 경우 forward에서 C를 미리 알 수 없으므로, register_buffer나 build가 필요할 수 있음.
        # 예시를 위해 forward에서 C를 얻은 뒤 lazy init 하는 방식을 사용하거나,
        # 미리 C를 인자로 받아두세요.
        
        # 여기서는 일단 C=3 가정
        self.input_dim = patch_size * patch_size * 3
        self.projection = nn.Linear(self.input_dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "이미지 크기는 patch_size로 나누어 떨어져야 함."

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        # (B, C, H, W) -> (B, C, h_patches, patch_size, w_patches, patch_size)
        x = x.reshape(B, C, h_patches, self.patch_size, w_patches, self.patch_size)
        # 순서를 바꿔 (B, number_of_patches, patch_size*patch_size*C)
        # (B, C, h_patches, w_patches, patch_size, patch_size) -> (B, h_patches*w_patches, patch_size*patch_size*C)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h_patches, w_patches, patch_size, patch_size, C)
        x = x.reshape(B, h_patches * w_patches, self.patch_size * self.patch_size * C)

        # 이제 패치 벡터에 projection
        x = self.projection(x)  # (B, num_patches, dim)
        return x
