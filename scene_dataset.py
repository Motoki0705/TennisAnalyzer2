import os
import csv
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import Config


class SceneDataset(Dataset):
    """
    このデータセットは、annotated_frames ディレクトリ直下にある
    複数の video_XXXX フォルダ内の各 scene フォルダからサンプルを取得します。
    
    各シーンフォルダ内には連続したフレームが保存されており、seane_len 枚を
    stride 間隔で選択して、各画像 [3, H, W] をチャネル方向に連結し、
    [3 * seane_len, H, W] のテンソルとして出力します。
    
    CSVファイルは各 video_XXXX フォルダ内に保存され、各行は
    {'scene', 'event_1', 'event_2', 'event_3'} の情報を持ちます。
    各イベントラベルは既に数値（クラスID）になっているため、変換は行いません。
    
    さらに、コンストラクタで split を指定することで、トレーニング用 ('train') 
    と検証用 ('val') のデータセットを別々に出力できます。
    val_ratio で検証用データの割合を、seed で分割の再現性を確保します。
    """
    def __init__(self, annotated_frames_dir, seane_len=10, stride=5, 
                 input_size=(512, 512), transform=None,
                 split='train', val_ratio=0.2, seed=42):
        """
        :param annotated_frames_dir: "annotated_frames" ディレクトリのパス
        :param seane_len: 1シーン内に使用するフレーム数（例：10）
        :param stride: フレーム選択の間隔（例：5）
        :param input_size: (H, W) のタプル。画像はこのサイズにリサイズされる
        :param transform: torchvision.transforms。指定がない場合は Resize(input_size) と ToTensor() を適用
        :param split: データセットの種類。'train' または 'val' を指定
        :param val_ratio: 全体のうち検証用に使用する割合（例：0.2）
        :param seed: データ分割時のランダムシード（再現性のため）
        """
        self.annotated_frames_dir = annotated_frames_dir
        self.seane_len = seane_len
        self.stride = stride
        self.input_size = input_size
        self.split = split
        self.val_ratio = val_ratio
        self.seed = seed
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),  # 各画像は [3, H, W] となる
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # 各 video_XXXX フォルダ内の CSV を読み込み、(video_dir, scene_folder, label) のタプルを self.data に格納
        self.data = []
        for video_dir in sorted(os.listdir(self.annotated_frames_dir)):
            video_path = os.path.join(self.annotated_frames_dir, video_dir)
            if os.path.isdir(video_path) and video_dir.startswith("video_"):
                csv_filename = f"annotations_{video_dir}.csv"
                csv_path = os.path.join(video_path, csv_filename)
                if not os.path.exists(csv_path):
                    print(f"Warning: CSV file {csv_path} not found in {video_path}. Skipping this video.")
                    continue
                with open(csv_path, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label = {
                            'event_1': int(row['event_1']),
                            'event_2': int(row['event_2']),
                            'event_3': int(row['event_3'])
                        }
                        self.data.append((video_dir, row['scene'], label))
        
        # 分割のためにデータをシャッフル（seed により再現性あり）
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(self.data)
        
        # train / val の比率でデータを分割
        split_index = int(len(self.data) * (1 - self.val_ratio))
        if self.split == 'train':
            self.data = self.data[:split_index]
        elif self.split == 'val':
            self.data = self.data[split_index:]
        else:
            raise ValueError("split must be either 'train' or 'val'")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        video_dir, scene_folder, label = self.data[index]
        scene_path = os.path.join(self.annotated_frames_dir, video_dir, scene_folder)
        # シーンフォルダ内の画像ファイル一覧を取得し、ファイル名中の数値部分でソート
        file_list = sorted(
            [os.path.join(scene_path, fname)
             for fname in os.listdir(scene_path)
             if fname.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: int(re.search(r'frame_(\d+)', os.path.basename(x)).group(1))
        )
        
        total_frames = len(file_list)
        required_frames = self.seane_len * self.stride
        if total_frames < required_frames:
            raise ValueError(f"Scene {scene_folder} in {video_dir} has {total_frames} frames, required {required_frames}.")
        
        # ランダムな初期オフセット（0～stride-1）
        offset = random.randint(0, self.stride - 1)
        indices = [offset + k * self.stride for k in range(self.seane_len)]
        
        imgs = []
        for idx in indices:
            if idx >= total_frames:
                raise IndexError(f"Index {idx} out of range for scene {scene_folder} in {video_dir}.")
            img = Image.open(file_list[idx]).convert('RGB')
            img = self.transform(img)  # [3, H, W]
            imgs.append(img)
        # 各画像をチャネル方向に連結して [3 * seane_len, H, W] にする
        sample = torch.cat(imgs, dim=0)
        return sample, label

if __name__ == '__main__':
    dataset = SceneDataset(Config.annotated_frames_dir)
    dataloader = DataLoader(dataset)
    img, label = next(iter(dataloader))
    print(img.shape)
    print(label)