import cv2
import os
import csv
import shutil
from config import Config  # Configには video_paths や event_groups（または従来のevent1_mappingなど）が定義されている前提
from typing import Optional, List, Dict, Tuple
import numpy as np

class TennisAnnotationTool:
    def __init__(
        self,
        video_path: str = None,
        scale_factor: int = 2,
        config: Optional[Config] = None,
        event_groups: Optional[Dict[str, Dict[str, int]]] = None, 
        output_frames_dir: Optional[str] = None,
        event_button_font_scale: float = 1.0,
        event_button_padding: int = 15,
        lookback_frames: int = 10
    ):
        """
        複数の動画を順次アノテーションするツール
        - config に video_paths があれば複数動画対応。
        - event_groups: イベントグループのマッピング辞書。キーがグループ名、値がイベントマッピング辞書。
          例: {"event_1": {"forehand": 0, "backhand": 1}, "event_2": {"volley": 0}}
        """
        if config is not None:
            self.config = config
            if hasattr(config, "video_paths"):
                self.video_paths = config.video_paths
            else:
                self.video_paths = [config.video_path]
            # configにevent_groupsが定義されていれば利用、なければ従来の3グループを利用
            if hasattr(config, "event_groups"):
                self.event_groups = config.event_groups
            else:
                self.event_groups = {
                    "event_1": getattr(config, "event1_mapping", {"volley": 0}),
                    "event_2": getattr(config, "event2_mapping", {"fore": 0}),
                    "event_3": getattr(config, "event3_mapping", {"poating": 0})
                }
        else:
            self.config = None
            self.video_paths = [video_path]
            self.event_groups = event_groups if event_groups is not None else {
                "event_1": {"volley": 0},
                "event_2": {"fore": 0},
                "event_3": {"poating": 0}
            }

        self.scale_factor = scale_factor
        # 各グループの選択状態（初期は未選択）
        self.event_selected = {group: None for group in self.event_groups}

        # ボタンの位置情報。キーは (group, event) のタプル
        self.button_positions: Dict[Tuple[str, str], Tuple[int, int, int, int]] = {}

        # アノテーション情報
        self.annotations: List[Dict[str, any]] = []
        self.current_frame_idx: int = 0  # 現在のフレーム番号

        # 出力先のベースディレクトリ（各動画ごとにサブフォルダを作成）
        if output_frames_dir is not None:
            self.base_output_dir = output_frames_dir
        else:
            self.base_output_dir = os.path.join(os.path.dirname(self.video_paths[0]), "annotated_frames")
        os.makedirs(self.base_output_dir, exist_ok=True)

        # シーン（ブロック）の連番管理
        self.scene_index: int = 0

        # イベントボタンのサイズ指定
        self.event_button_font_scale = event_button_font_scale
        self.event_button_padding = event_button_padding

        # ルックバックフレーム数
        self.lookback_frames = lookback_frames

        # フレームマーカー: マーカーをつけたフレーム番号を記録（ブロック内の番号）
        self.marked_frames: set = set()

    def save_annotations_to_csv(self) -> None:
        """
        各動画ごとにアノテーション情報をCSVに保存します。
        CSVの項目は「scene, 各イベントグループ, marked_frames, start_frame」となります。
        """
        fieldnames = ['scene'] + list(self.event_groups.keys()) + ['marked_frames', 'start_frame']
        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for annotation in self.annotations:
                writer.writerow(annotation)
        print(f"アノテーションを {self.output_csv} に保存しました。")

    def save_event_annotation(self) -> None:
        """
        すべてのイベントグループでイベントが選択されている場合、現在のフレームから
        lookback_frames 分さかのぼって連続フレームを読み込み、シーンファイル（npz）に保存します。
        また、各イベントの値（必要ならマッピングを通して変換）とマーカー情報を記録します。
        """
        start_frame = max(0, self.current_frame_idx - self.lookback_frames + 1)
        cap2 = cv2.VideoCapture(self.video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        block = []
        for idx in range(start_frame, self.current_frame_idx + 1):
            ret, frame = cap2.read()
            if not ret:
                break
            # マーカーがあるフレームには、例として赤い円を描画
            if idx in self.marked_frames:
                cv2.circle(frame, (30, 30), 15, (0, 0, 255), 3)
            block.append((idx, frame))
        cap2.release()
        if not block:
            print("読み込めるフレームがありません。")
            return

        scene_folder_name = f"scene_{self.scene_index:04d}"
        npz_filename = f"{scene_folder_name}.npz"
        scene_path = os.path.join(self.output_frames_dir, npz_filename)
        frame_dict = {f'frame_{idx:04d}': frame for idx, frame in block}
        np.savez_compressed(scene_path, **frame_dict)

        # 各イベントグループの選択値をマッピングがあれば変換
        events_annotation = {}
        for group, mapping in self.event_groups.items():
            selected = self.event_selected.get(group)
            events_annotation[group] = mapping[selected] if selected in mapping else selected

        annotation = {
            'scene': scene_folder_name,
            **events_annotation,
            'marked_frames': max(sorted(list(self.marked_frames))),
            'start_frame': block[0][0]
        }
        self.annotations.append(annotation)
        print(f"npzファイル '{scene_path}' に、イベント {events_annotation} とマーカー {sorted(list(self.marked_frames))} "
              f"として {len(block)} フレームを保存しました。")
        self.current_frame_idx += 1
        self.scene_index += 1

        # 選択状態とマーカー状態をリセット
        for group in self.event_selected:
            self.event_selected[group] = None
        self.marked_frames.clear()

    def go_back_one_frame(self, cap: cv2.VideoCapture) -> None:
        """
        'a'キーが押された場合、直前の注釈シーンを取り消し、
        再生位置をその開始フレームに戻します。シーンがなければ1フレーム戻ります。
        """
        if self.annotations:
            last_annotation = self.annotations[-1]
            scene_folder_name = last_annotation['scene']
            npz_filename = f"{scene_folder_name}.npz"
            scene_path = os.path.join(self.output_frames_dir, npz_filename)
            if os.path.exists(scene_path):
                os.remove(scene_path)
            self.current_frame_idx = last_annotation['start_frame']
            self.annotations.pop()
            print(f"シーン '{scene_folder_name}' を取り消しました。再生位置をフレーム {self.current_frame_idx} に戻します。")
        else:
            if self.current_frame_idx > 0:
                self.current_frame_idx -= 1
                print(f"1フレーム戻ります。現在のフレーム: {self.current_frame_idx}")

    def draw_buttons(self, frame: any) -> None:
        """
        各イベントグループのボタンを描画します。  
        クリックされたボタンは緑色になり、選択状態が反映されます。
        """
        self.button_positions = {}
        font_scale = self.event_button_font_scale
        thickness = 2
        padding = self.event_button_padding

        x_start = 5
        y_offset = 5
        group_gap = 20

        for group, mapping in self.event_groups.items():
            cv2.putText(frame, group, (x_start, y_offset + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            y = y_offset + 30
            selected_event = self.event_selected.get(group)
            for ev in mapping:
                color = (0, 255, 0) if selected_event == ev else (255, 255, 255)
                text_size = cv2.getTextSize(ev, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                x2 = x_start + text_size[0] + padding
                y2 = y + text_size[1] + padding
                cv2.rectangle(frame, (x_start, y), (x2, y2), color, -1)
                cv2.putText(frame, ev, (x_start + padding // 2, y + text_size[1] + padding // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
                self.button_positions[(group, ev)] = (x_start, y, x2, y2)
                y += (y2 - y) + 5
            y_offset = y + group_gap

        instructions = ("Select one event from each group; then click any button to save the last {} frames; "
                        "'d': next; 'f': +10 frames; 'g': +50 frames; 'a': back; 'm': mark frame; 's': pause; 'q': quit").format(self.lookback_frames)
        cv2.putText(frame, instructions, (5, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """
        マウスクリック時に各ボタン領域をチェックし、クリックされたグループの選択状態を更新します。
        全グループで選択が完了していれば、保存処理を呼び出します。
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            for (group, ev), (x1, y1, x2, y2) in self.button_positions.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.event_selected[group] = ev
                    print(f"イベントボタン [{group}] '{ev}' がクリックされました。")
                    break
            if all(value is not None for value in self.event_selected.values()):
                print("すべてのイベントが選択されました。保存処理を開始します。")
                self.save_event_annotation()

    def run_video(self, video_path: str, video_index: int) -> None:
        """
        1本の動画に対してアノテーション処理を行います。  
        出力は base_output_dir/video_XXXX/ に保存し、CSVはそのサブフォルダに出力されます。
        """
        print(f"動画 '{video_path}' を処理中...")
        self.video_path = video_path
        self.current_frame_idx = 0
        self.scene_index = 0
        self.annotations = []

        video_output_dir = os.path.join(self.base_output_dir, f"video_{video_index:04d}")
        os.makedirs(video_output_dir, exist_ok=True)
        self.output_frames_dir = video_output_dir
        self.output_csv = os.path.join(video_output_dir, f"annotations_video_{video_index:04d}.csv")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"動画 '{self.video_path}' を開けません。")
            return

        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Frame', self.mouse_callback)

        # 既存のCSVがあれば読み込み、再開位置を推定
        if os.path.exists(self.output_csv):
            print(f"既存のアノテーション {self.output_csv} を読み込んでいます...")
            with open(self.output_csv, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # marked_framesは文字列からリストに変換
                    row['marked_frames'] = eval(row['marked_frames'])
                    self.annotations.append(row)
            print(f"注釈数: {len(self.annotations)}. 新規注釈は後続で追加されます。")
            last_frame = -1
            max_scene_index = -1
            for item in os.listdir(self.output_frames_dir):
                if item.startswith("scene_") and item.endswith(".npz"):
                    try:
                        scene_index = int(item.split("_")[1].split(".")[0])
                        max_scene_index = max(max_scene_index, scene_index)
                        frames_data = np.load(os.path.join(self.output_frames_dir, item))
                        for filename in frames_data.files:
                            if filename.startswith("frame_"):
                                frame_number = int(filename.split("_")[1])
                                last_frame = max(last_frame, frame_number)
                    except Exception as e:
                        print(f"シーンファイルの解析エラー: {e}")
            if last_frame >= 0:
                self.current_frame_idx = last_frame + 1
            if max_scene_index >= 0:
                self.scene_index = max_scene_index + 1
            print(f"再開位置: フレーム {self.current_frame_idx}, シーン番号 {self.scene_index}")
        else:
            print("新しいアノテーションを開始します。")

        last_frame_idx = self.current_frame_idx - 1
        resize_dims = None
        paused = False
        fps = 40
        while True:
            if self.current_frame_idx != last_frame_idx + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                print("動画の最後に到達しました。")
                break

            last_frame_idx = self.current_frame_idx
            if resize_dims is None:
                resize_dims = (frame.shape[1] * self.scale_factor, frame.shape[0] * self.scale_factor)
            display_frame = cv2.resize(frame, resize_dims)

            # 現在のフレームにマーカーがあれば、視覚的に表示（例：円を描画）
            if self.current_frame_idx in self.marked_frames:
                cv2.circle(display_frame, (50, 50), 20, (0, 0, 255), 3)

            self.draw_buttons(display_frame)
            info_text = f"Frame: {self.current_frame_idx}"
            cv2.putText(display_frame, info_text, (5, display_frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if paused:
                cv2.putText(display_frame, "PAUSED", (display_frame.shape[1] // 2 - 50, display_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            cv2.imshow('Frame', display_frame)
            key = cv2.waitKey(int(1e+3 / fps)) & 0xFF
            if key == ord('q'):
                self.save_annotations_to_csv()
                break
            elif key == ord('d'):
                self.current_frame_idx += 8
            elif key == ord('f'):
                self.current_frame_idx += 30
            elif key == ord('g'):
                self.current_frame_idx += 100
            elif key == ord('a'):
                self.go_back_one_frame(cap)
            elif key == ord('m'):
                # 現在のフレームのマーカーのON/OFFを切り替え
                if self.current_frame_idx in self.marked_frames:
                    self.marked_frames.remove(self.current_frame_idx)
                    print(f"フレーム {self.current_frame_idx} のマーカーを解除しました。")
                else:
                    self.marked_frames.add(self.current_frame_idx)
                    print(f"フレーム {self.current_frame_idx} にマーカーを設定しました。")
            elif key == ord('w'):
                if fps == 40:
                    fps = 10
                elif fps == 10:
                    fps = 40
            elif key == ord('s'):
                paused = not paused        
            elif not paused:
                self.current_frame_idx += 1

        self.save_annotations_to_csv()
        cap.release()
        cv2.destroyAllWindows()
        print(f"動画 '{video_path}' のアノテーションが完了しました。")

    def run(self) -> None:
        """
        Config に記載された複数の動画を順次アノテーションします。
        各動画は base_output_dir/video_XXXX/ に保存されます。
        """
        video_index = 1
        for video in self.video_paths:
            self.run_video(video, video_index)
            video_index += 1
        print("すべての動画のアノテーションが完了しました。")

if __name__ == "__main__":
    # Config に event_groups が定義されている前提。なければ従来の3グループを利用。
    tool = TennisAnnotationTool(
        config=Config,
        event_button_font_scale=1.0,
        event_button_padding=20,
        lookback_frames=70,
    )
    tool.run()
