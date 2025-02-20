import os
import torch
import matplotlib.pyplot as plt
from config import Config

def plot_concatenated_metrics(checkpoint_path):
    # グローバル checkpoint を読み込む
    checkpoint = torch.load(checkpoint_path)
    
    if not checkpoint:
        print("チェックポイント内にデータがありません。")
        return

    global_epochs = []   # 全フェーズのグローバル epoch インデックス
    train_losses = []    # 連結した train loss
    val_losses = []      # 連結した validation loss
    phase_boundaries = []  # 各フェーズの終了時のグローバル epoch 番号

    global_epoch = 0
    # キーはフェーズ番号（例: 1,2,3）である前提
    phases = sorted(checkpoint.keys(), key=lambda x: int(x))
    for phase in phases:
        phase_data = checkpoint[phase]
        metrics = phase_data.get('metrics', {})
        if not metrics:
            print(f"Phase {phase} の metrics が見つかりません。")
            continue
        # 各フェーズ内の epoch を数値順にソート
        phase_epochs = sorted(metrics.keys(), key=lambda x: int(x))
        for epoch in phase_epochs:
            global_epoch += 1
            global_epochs.append(global_epoch)
            train_losses.append(metrics[epoch]['train_loss'])
            val_losses.append(metrics[epoch]['val_loss'])
        phase_boundaries.append(global_epoch)
    
    # 連結したグラフの描画
    plt.figure(figsize=(12, 8))
    plt.plot(global_epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(global_epochs, val_losses, label="Validation Loss", marker="o")
    
    # フェーズの境界を示す垂直線（最終フェーズ以外）
    for boundary in phase_boundaries[:-1]:
        plt.axvline(x=boundary + 0.5, color='red', linestyle='--', 
                    label='Phase Transition' if boundary == phase_boundaries[0] else None)
    
    plt.xlabel("Global Epoch")
    plt.ylabel("Loss")
    plt.title("Concatenated Loss Progression Across Phases")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    checkpoint_path = os.path.join(Config.checkpoint_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        plot_concatenated_metrics(checkpoint_path)
    else:
        print(f"Checkpoint file not found at {checkpoint_path}")
