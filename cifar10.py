import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 論文仕様の定数 (Section 8.1) ---
M = 30           # 触手スロット数
D_ENV = 256      # 環境特徴量次元
D_V = 64         # ベクトル次元
NUM_CLASSES = 10
BATCH_SIZE = 64

class Tentacle(nn.Module):
    """論文 2.1.2 節に基づく知識モジュール"""
    def __init__(self, d_v):
        super().__init__()
        self.f_theta = nn.Sequential(
            nn.Linear(d_v, 128),
            nn.ReLU(),
            nn.Linear(128, d_v)
        )
        self.x_buffer = None

    def forward(self, x, mode="inference"):
        if mode == "inference":
            self.x_buffer = x.detach() # Section 4.1.2: Store announcement
            return self.f_theta(x)
        return self.f_theta(self.x_buffer)

class OctonetHead(nn.Module):
    """論文 2.1.1 節に基づく中央制御装置"""
    def __init__(self, m, d_env, d_v, num_classes):
        super().__init__()
        self.m = m
        self.d_v = d_v
        
        # Section 5.1: CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, d_env)
        )
        
        # s_t = [s_env, r, y_1...y_M] (Section 2.1.1)
        input_dim = d_env + 1 + (m * d_v)
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU()
        )
        
        # Output components (Section 6.3 Factorization)
        self.pi_ext = nn.Linear(512, num_classes) # a_ext
        self.pi_type = nn.Linear(512, m * 3)       # T_i: Null, Ann, Res
        self.pi_vec = nn.Linear(512, m * d_v)      # v_i (Result vector)
        self.v_theta = nn.Linear(512, 1)           # V_theta(s)

    def forward(self, img, r_prev, y_hats):
        s_env = self.encoder(img)
        y_flat = torch.cat(y_hats, dim=-1)
        s_t = torch.cat([s_env, r_prev, y_flat], dim=-1)
        
        h = self.shared_layers(s_t)
        
        a_ext = self.pi_ext(h)
        t_logits = self.pi_type(h).view(-1, self.m, 3)
        v_i = self.pi_vec(h).view(-1, self.m, self.d_v)
        val = self.v_theta(h)
        
        return a_ext, t_logits, v_i, val

class OctonetSystem:
    def __init__(self, device):
        self.device = device
        self.head = OctonetHead(M, D_ENV, D_V, NUM_CLASSES).to(device)
        self.tentacles = nn.ModuleList([Tentacle(D_V).to(device) for _ in range(M)])
        
        # 独立した最適化器 (Section 2.2 / 3.1)
        self.h_optimizer = optim.Adam(self.head.parameters(), lr=1e-4)
        self.t_optimizers = [optim.Adam(t.parameters(), lr=1e-3) for t in self.tentacles]

    def train_step(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)
        batch_size = images.size(0)
        
        # --- PHASE 1: Inference (Announcement) ---
        r_prev = torch.zeros(batch_size, 1).to(self.device)
        y_hats_in = [torch.zeros(batch_size, D_V).to(self.device) for _ in range(M)]
        
        a_ext, t_logits, v_i, v_s = self.head(images, r_prev, y_hats_in)
        t_types = torch.softmax(t_logits, dim=-1).argmax(dim=-1)
        
        # 各触手の推論応答を取得
        current_y_hats = []
        for i in range(M):
            # Announcement(1) または Result(2) の場合に推論
            y_hat = self.tentacles[i](v_i[:, i, :], mode="inference")
            current_y_hats.append(y_hat)

        # --- PHASE 2: Head Learning (Section 4.1 Phase B/C) ---
        # 外部報酬 z の決定
        with torch.no_grad():
            z = (a_ext.argmax(dim=1) == labels).float().unsqueeze(-1) * 2 - 1

        self.h_optimizer.zero_grad()
        
        # L_value (Section 4.1.1)
        loss_val = F.mse_loss(v_s, z)
        
        # L_policy (Section 4.1.2) - MCTSターゲットの代わりに外部ラベルで近似
        loss_policy = F.cross_entropy(a_ext, labels)
        
        # 統合損失 L_total
        l_total = loss_val + loss_policy
        l_total.backward()
        self.h_optimizer.step()

        # --- PHASE 3: Tentacle Imprinting (Section 3.1 Step 3) ---
        for i in range(M):
            # 論文 7.1: Result信号を受け取った触手のみ更新 (非破壊的)
            if (t_types[:, i] == 2).any():
                self.t_optimizers[i].zero_grad()
                # ターゲットはHeadが生成したベクトル v_i (detachなしでHeadの勾配を維持)
                target_v = v_i[:, i, :].detach()
                pred_y = self.tentacles[i](None, mode="learning")
                loss_t = F.mse_loss(pred_y, target_v)
                loss_t.backward()
                self.t_optimizers[i].step()

        return loss_policy.item(), z.mean().item()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    
    octonet = OctonetSystem(device)
    for epoch in range(50):
        for i, (imgs, lbls) in enumerate(train_loader):
            l_p, rew = octonet.train_step(imgs, lbls)
            if i % 100 == 0:
                print(f"Epoch {epoch} | Batch {i} | Policy Loss: {l_p:.4f} | Avg Reward: {rew:.2f}")
