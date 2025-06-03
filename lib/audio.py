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
            # Get static-ffmpeg executable path
            ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            
            # Build ffmpeg command
            cmd = [
                ffmpeg_path,
                '-i', audio_file,
                '-vn',  # Disable video
                '-acodec', 'pcm_s16le',  # 16-bit PCM encoding
                '-ac', '1',  # Mono channel
                '-ar', '16000',  # 16kHz sample rate
                '-f', 'wav',  # WAV format
                '-y',  # Overwrite output file
                self.output_path
            ]
            
            # Run command using subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.error = result.stderr
        except Exception as e:
            self.error = str(e)

    def cleanup(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
