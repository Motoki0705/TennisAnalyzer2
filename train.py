import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy
from scene_dataset import SceneDataset
from SwinTransformerV2 import SwinTransformerV2
from config import Config
from tqdm import tqdm

# --- グローバル checkpoint 用の関数 ---
def load_global_checkpoint(checkpoint_dir):
    filename = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.exists(filename):
        return {}
    checkpoint = torch.load(filename)
    print(f"Loaded global checkpoint: {filename}")
    return checkpoint

def save_global_checkpoint(global_checkpoint, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filename = os.path.join(checkpoint_dir, "checkpoint.pth")
    torch.save(global_checkpoint, filename)
    print(f"Global checkpoint saved: {filename}")

# --- DataLoader の生成 ---
def get_dataloaders(annotated_frames_dir, seane_len, stride, input_size, train_transform, val_transform, batch_size, val_split=0.2):
    train_dataset = SceneDataset(
        annotated_frames_dir,
        seane_len=seane_len,
        stride=stride,
        input_size=input_size,
        transform=train_transform,
        split='train',
        val_ratio=val_split
    )
    val_dataset = SceneDataset(
        annotated_frames_dir,
        seane_len,
        stride,
        input_size,
        transform=val_transform,
        split='val',
        val_ratio=val_split
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, train_loader, val_loader

# --- クラス重みを計算するヘルパー関数 ---
def compute_class_weights(dataset, event_key, num_classes):
    counts = [0] * num_classes
    for i in range(len(dataset)):
        _, labels = dataset[i]
        label = labels[event_key]
        counts[label] += 1
    total = sum(counts)
    # 出現頻度の逆数を重みとする（ゼロ割防止のため微小値を加える）
    weights = [total / (count + 1e-6) for count in counts]
    return torch.tensor(weights, dtype=torch.float)

# --- モデル出力層の調整 ---
def setup_model(model, phase, device):
    event1_mapping = getattr(Config, 'event1_mapping', {"stroke": 0, "volley": 1, "serve": 2, "smash": 3})
    event2_mapping = getattr(Config, 'event2_mapping', {"fore": 0, "back": 1})
    event3_mapping = getattr(Config, 'event3_mapping', {"high": 0, "low": 1})
    
    num_event_1 = len(event1_mapping)
    num_event_2 = len(event2_mapping)
    num_event_3 = len(event3_mapping)
    
    if phase == 1:
        out_dim = num_event_1
    elif phase == 2:
        out_dim = num_event_1 + num_event_2
    elif phase == 3:
        out_dim = num_event_1 + num_event_2 + num_event_3
    else:
        raise ValueError("Phase must be 1, 2, or 3")
    
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, out_dim).to(device)
    
    return model, num_event_1, num_event_2, num_event_3

# --- エポック単位の学習 ---
def train_one_epoch(model, dataloader, criterions, optimizer, phase, num_event_1, num_event_2, num_event_3, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Training Phase {phase}", unit="batch", leave=False)
    
    # 各イベントの Accuracy を初期化（フェーズに応じて）
    accuracy_1 = Accuracy(task="multiclass", num_classes=num_event_1).to(device)
    if phase >= 2:
        accuracy_2 = Accuracy(task="multiclass", num_classes=num_event_2).to(device)
    if phase == 3:
        accuracy_3 = Accuracy(task="multiclass", num_classes=num_event_3).to(device)
    
    for inputs, labels in progress_bar:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        if phase == 1:
            target = labels['event_1'].to(device)
            loss = criterions['event1'](outputs, target)
            acc = accuracy_1(outputs, target)
        elif phase == 2:
            target1 = labels['event_1'].to(device)
            target2 = labels['event_2'].to(device)
            loss1 = criterions['event1'](outputs[:, :num_event_1], target1)
            loss2 = criterions['event2'](outputs[:, num_event_1:], target2)
            acc1 = accuracy_1(outputs[:, :num_event_1], target1)
            acc2 = accuracy_2(outputs[:, num_event_1:], target2)
            loss = loss1 + loss2
            acc = (acc1 + acc2) / 2
        elif phase == 3:
            target1 = labels['event_1'].to(device)
            target2 = labels['event_2'].to(device)
            target3 = labels['event_3'].to(device)
            loss1 = criterions['event1'](outputs[:, :num_event_1], target1)
            loss2 = criterions['event2'](outputs[:, num_event_1:num_event_1+num_event_2], target2)
            loss3 = criterions['event3'](outputs[:, num_event_1+num_event_2:], target3)
            acc1 = accuracy_1(outputs[:, :num_event_1], target1)
            acc2 = accuracy_2(outputs[:, num_event_1:num_event_1+num_event_2], target2)
            acc3 = accuracy_3(outputs[:, num_event_1+num_event_2:], target3)
            loss = loss1 + loss2 + loss3
            acc = (acc1 + acc2 + acc3) / 3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += acc.item()
        total_batches += 1
        progress_bar.set_postfix(loss=running_loss / total_batches, acc=running_acc / total_batches)
        
    print(f'model output: {outputs}')    
    return running_loss / total_batches if total_batches > 0 else 0.0

# --- エポック単位の検証 ---
def validate_one_epoch(model, dataloader, criterions, phase, num_event_1, num_event_2, num_event_3, device):
    model.eval()
    running_loss = 0.0
    total_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Validation Phase {phase}", unit="batch", leave=False)
    
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if phase == 1:
                target = labels['event_1'].to(device)
                loss = criterions['event1'](outputs, target)
            elif phase == 2:
                target1 = labels['event_1'].to(device)
                target2 = labels['event_2'].to(device)
                loss1 = criterions['event1'](outputs[:, :num_event_1], target1)
                loss2 = criterions['event2'](outputs[:, num_event_1:], target2)
                loss = loss1 + loss2
            elif phase == 3:
                target1 = labels['event_1'].to(device)
                target2 = labels['event_2'].to(device)
                target3 = labels['event_3'].to(device)
                loss1 = criterions['event1'](outputs[:, :num_event_1], target1)
                loss2 = criterions['event2'](outputs[:, num_event_1:num_event_1+num_event_2], target2)
                loss3 = criterions['event3'](outputs[:, num_event_1+num_event_2:], target3)
                loss = loss1 + loss2 + loss3

            running_loss += loss.item()
            total_batches += 1
            progress_bar.set_postfix(loss=running_loss / total_batches)
    
    return running_loss / total_batches if total_batches > 0 else 0.0

# --- 各フェーズの学習を実施（グローバル checkpoint を更新） ---
def train_phase(phase, model, train_loader, val_loader, criterions, optimizer,
                num_event_1, num_event_2, num_event_3, device, checkpoint_dir,
                max_epochs, patience, global_checkpoint):
    # 既存のフェーズ用 checkpoint があれば再開
    phase_checkpoint = global_checkpoint.get(phase, None)
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    metrics = {}
    
    if phase_checkpoint is not None and not phase_checkpoint.get('finished', False):
        model.load_state_dict(phase_checkpoint['model_state_dict'])
        optimizer.load_state_dict(phase_checkpoint['optimizer_state_dict'])
        start_epoch = phase_checkpoint['epoch'] + 1
        best_val_loss = phase_checkpoint['best_val_loss']
        metrics = phase_checkpoint.get('metrics', {})
        print(f"Resuming Phase {phase} training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, max_epochs):
        train_loss = train_one_epoch(model, train_loader, criterions, optimizer,
                                     phase, num_event_1, num_event_2, num_event_3, device)
        val_loss = validate_one_epoch(model, val_loader, criterions,
                                      phase, num_event_1, num_event_2, num_event_3, device)
        print(f"Phase {phase} Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        metrics[epoch+1] = {'train_loss': train_loss, 'val_loss': val_loss}
        
        # 更新されたフェーズ情報をグローバル checkpoint に反映
        global_checkpoint[phase] = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'metrics': metrics,
            'finished': False
        }
        save_global_checkpoint(global_checkpoint, checkpoint_dir)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Patience counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Validation loss did not improve for {patience} consecutive epochs. Moving to next phase.")
            break

    # フェーズ完了時に finished フラグを True に更新
    global_checkpoint[phase] = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'metrics': metrics,
        'finished': True
    }
    save_global_checkpoint(global_checkpoint, checkpoint_dir)
    return model, metrics

# --- main() ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SwinTransformerV2(img_size=224, in_chans=30, num_classes=10).to(device)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # グローバル checkpoint をロード
    global_checkpoint = load_global_checkpoint(Config.checkpoint_dir)
    
    # 既に完了済みのフェーズがあればスキップ
    for phase in [1, 2, 3]:
        phase_cp = global_checkpoint.get(phase, {})
        if phase_cp.get('finished', False):
            print(f"Phase {phase} is already completed. Skipping.")
            continue
        
        print(f"==========================")
        print(f"Starting training for Phase {phase}")
        print(f"==========================")
        
        # フェーズごとに出力層を再設定
        model, num_event1, num_event2, num_event3 = setup_model(model, phase, device)
        # train_dataset も返すように変更
        train_dataset, train_loader, val_loader = get_dataloaders(
            Config.annotated_frames_dir, Config.seane_len, Config.stride,
            Config.input_size, train_transform, val_transform, Config.batch_size, Config.val_split,
        )
        # フェーズごとに損失関数（クラス重み付き）を作成
        if phase == 1:
            weights_event1 = compute_class_weights(train_dataset, 'event_1', num_event1).to(device)
            criterions = {'event1': nn.CrossEntropyLoss(weight=weights_event1)}
        elif phase == 2:
            weights_event1 = compute_class_weights(train_dataset, 'event_1', num_event1).to(device)
            weights_event2 = compute_class_weights(train_dataset, 'event_2', num_event2).to(device)
            criterions = {
                'event1': nn.CrossEntropyLoss(weight=weights_event1),
                'event2': nn.CrossEntropyLoss(weight=weights_event2)
            }
        elif phase == 3:
            weights_event1 = compute_class_weights(train_dataset, 'event_1', num_event1).to(device)
            weights_event2 = compute_class_weights(train_dataset, 'event_2', num_event2).to(device)
            weights_event3 = compute_class_weights(train_dataset, 'event_3', num_event3).to(device)
            criterions = {
                'event1': nn.CrossEntropyLoss(weight=weights_event1),
                'event2': nn.CrossEntropyLoss(weight=weights_event2),
                'event3': nn.CrossEntropyLoss(weight=weights_event3)
            }
        
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
        
        model, _ = train_phase(phase, model, train_loader, val_loader, criterions, optimizer,
                               num_event1, num_event2, num_event3, device, Config.checkpoint_dir,
                               Config.epochs, Config.patience, global_checkpoint)
        print(f"Finished training for Phase {phase}\n")
        
if __name__ == "__main__":
    main()
