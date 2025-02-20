import cv2
import os
import pandas as pd
from config import Config

def get_reverse_mappings(config):
    """ イベントのマッピングを逆引き辞書として取得 """
    return (
        {v: k for k, v in config.event1_mapping.items()},
        {v: k for k, v in config.event2_mapping.items()},
        {v: k for k, v in config.event3_mapping.items()}
    )

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

def process_frame(frame_path, row, reverse_mapping_1, reverse_mapping_2, reverse_mapping_3):
    """ フレーム画像にアノテーションを描画し表示 """
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f'Failed to read {frame_path}')
        return
    
    event_1 = reverse_mapping_1.get(row['event_1'].iloc[0], "Unknown")
    event_2 = reverse_mapping_2.get(row['event_2'].iloc[0], "Unknown")
    event_3 = reverse_mapping_3.get(row['event_3'].iloc[0], "Unknown")
    
    cv2.putText(frame, event_1, (30, 150), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 0), 3)
    cv2.putText(frame, event_2, (30, 210), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 0), 3)
    cv2.putText(frame, event_3, (30, 270), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 0), 3)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == 27:
        return True
    return False

def check_annotation(config: Config):
    """ アノテーションを検証するメイン処理 """
    video_dirs = [os.path.join(config.annotated_frames_dir, d) for d in os.listdir(config.annotated_frames_dir)]
    reverse_mapping_1, reverse_mapping_2, reverse_mapping_3 = get_reverse_mappings(config)
    
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
            row = df[df['scene'] == scene_basename]
            if row.empty:
                print(f"No annotation found for {scene_basename}")
                continue
            
            frame_paths = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir)]
            for frame_path in frame_paths:
                if process_frame(frame_path, row, reverse_mapping_1, reverse_mapping_2, reverse_mapping_3):
                    break
    
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    check_annotation(Config)
