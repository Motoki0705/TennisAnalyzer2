import cv2
import os
import pandas as pd
from config import Config
import numpy as np

def get_reverse_mappings(config):
    """ イベントのマッピングを逆引き辞書として取得 """
    reverse_mapping_event_groups = {}
    for group, item in config.event_groups.items():
        reverse_mapping_event_groups[group] = {v: k for k, v in item.items()}
    
    return reverse_mapping_event_groups

def load_annotations(csv_path):
    """ アノテーションのCSVを読み込む """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None
    
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None

def process_frame(i, frame, row, reverse_mapping_event_groups):
    """ フレーム画像にアノテーションを描画し表示 """
    if frame is None:
        print(f'Failed to read frame_{i}')
        return
    
    y_offset = 150
    for group, item in reverse_mapping_event_groups.items():
        selected_event_val = row[group].iloc(0)
        print(selected_event_val)
        selected_event_key = item.get(selected_event_val)
        print(selected_event_key)
        cv2.putText(frame, selected_event_key, (30, y_offset), cv2.FONT_HERSHEY_PLAIN,  3.0, (0, 0, 0), 3)
        y_offset += 60

        cv2.imshow('Frame', frame)
        if cv2.waitKey(20) & 0xFF == 27:
            return True
    return False

def check_annotation(config: Config):
    """ アノテーションを検証するメイン処理 """
    video_dirs = [os.path.join(config.annotated_frames_dir, d) for d in os.listdir(config.annotated_frames_dir)]
    reverse_mapping_event_groups = get_reverse_mappings(config)
    
    for video_dir in video_dirs:
        video_basename = os.path.basename(video_dir)
        video_num = video_basename.split('_')[1]
        csv_path = os.path.join(video_dir, f'annotations_video_{video_num}.csv')
        df = load_annotations(csv_path)
        if df is None:
            continue
        
        scene_dirs = [os.path.join(video_dir, d) for d in os.listdir(video_dir) if d.startswith('scene')]
        for scene_dir in scene_dirs:
            scene_basename = os.path.basename(scene_dir)
            scene_basename = scene_basename.split('.')[0]
            row = df[df['scene'] == scene_basename]
            if row.empty:
                print(f"No annotation found for {scene_basename}")
                continue
            
            frames_data = np.load(scene_dir)
            frames_file = sorted(frames_data.files, key=lambda x: x.split('_')[1])
            indices = [3 + k * 7 for k in range(10)]
            
            for idx in range(70):
                
                frame = frames_data[frames_file[idx]]
                process_frame(idx, frame, row, reverse_mapping_event_groups)
    
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    check_annotation(Config)
