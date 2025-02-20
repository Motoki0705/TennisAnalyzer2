class Config:
    #annotation
    video_paths = [r'C:\Users\kamim\Desktop\vscode\tennis_analyzer\TennisAnalyzer_2\2024全国中学校大会 男子団体戦⧸決勝 清明学園 vs 上青木(野田・林田vs 木原・奥田).mp4']
    annotated_frames_dir = r"C:\Users\kamim\Desktop\vscode\tennis_analyzer\TennisAnalyzer_2\annotated_frames"
    event1_mapping = {
        "stroke": 0,
        "volley": 1,
        "serve": 2,
        "smash": 3,
        }
    event2_mapping = {
        "none": 0,
        "fore": 1,
        "back": 2,
        "front": 3,
        "overhand ": 4,
        "cut": 5
    }
    event3_mapping = {
        "none": 0,
        "passing": 1,
        "slow": 2,
        "spank": 3,
        "pouch": 4,
        "high": 5,
        "low": 6,
        "slice": 7,
        "drop": 8,
        "rob": 9,
    }

    #train
    seane_len = 10
    stride = 5
    input_size = (256, 256)
    batch_size = 16
    epochs = 100
    val_split = 0.2
    learning_rate = 0.0001
    patience = 5
    checkpoint_dir = r'C:\Users\kamim\Desktop\vscode\tennis_analyzer\TennisAnalyzer_2\checkpoints'

    #model input
    input_shape = (seane_len * 3, *input_size)
    shift_size = 4
    window_size = 8
    patch_size = 2
    dim = 64
    num_heads = 4
    num_class = 10
    num_stage = 4
    num_cmt = [2, 3, 2, 2]

    
    