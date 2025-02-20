import yt_dlp

def download_video_with_resolution(url, resolution=500, output_template="%(title)s.%(ext)s"):
    # フォーマット指定: 指定解像度の動画を選択
    # 正確な解像度の場合:
    # format_selector = f'bestvideo[height={resolution}]+bestaudio/best[height={resolution}]'
    # 柔軟に720p以下を選ぶ場合:
    format_selector = f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]'
    
    ydl_opts = {
        'outtmpl': output_template,
        'format': format_selector,
        'merge_output_format': 'mp4',  # ffmpegが必要
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    # ダウンロードするYouTube動画のURLを指定
    video_url = r"https://www.youtube.com/watch?v=w7frqSR-iO8&t=146s&ab_channel=SOFTTENNISNavi"
    download_video_with_resolution(video_url, resolution=720)
