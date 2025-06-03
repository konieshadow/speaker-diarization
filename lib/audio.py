import pathlib
import os
import tempfile
import subprocess

import static_ffmpeg


class AudioPreProcessor:
    def __init__(self):
        tmpdir = pathlib.Path(tempfile.mkdtemp())
        self.output_path = str(tmpdir / 'audio.wav')
        self.error = None

    def process(self, audio_file):
        # converts audio file to 16kHz 16bit mono wav using static-ffmpeg...
        print('pre-processing audio file...')
        
        try:
            # 获取 static-ffmpeg 的可执行文件路径
            ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            
            # 构建 ffmpeg 命令
            cmd = [
                ffmpeg_path,
                '-i', audio_file,
                '-vn',  # 禁用视频
                '-acodec', 'pcm_s16le',  # 16位PCM编码
                '-ac', '1',  # 单声道
                '-ar', '16000',  # 16kHz采样率
                '-f', 'wav',  # WAV格式
                '-y',  # 覆盖输出文件
                self.output_path
            ]
            
            # 使用 subprocess 运行命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.error = result.stderr
        except Exception as e:
            self.error = str(e)

    def cleanup(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
