"""
优化的说话人分离预测器，使用核心模块
"""
import json
import tempfile
from typing import Optional

from cog import BasePredictor, Input, Path as CogPath
from lib.speaker_diarization_core import SpeakerDiarizationCore


class Predictor(BasePredictor):
    def setup(self):
        """初始化说话人分离核心"""
        self.core = SpeakerDiarizationCore(
            model_name="pyannote/speaker-diarization-3.1",
            use_gpu=True,
            enable_compile=True
        )
        self.core.setup()

    def predict(
        self,
        audio: CogPath = Input(
            description="Audio file",
            default="https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/lex-levin-4min.mp3"
        ),
        num_speakers: Optional[int] = Input(
            description="Number of speakers (if known)", 
            default=None
        ),
        min_speakers: Optional[int] = Input(
            description="Minimum number of speakers", 
            default=None
        ),
        max_speakers: Optional[int] = Input(
            description="Maximum number of speakers", 
            default=None
        ),
    ) -> CogPath:
        """运行优化的预测"""

        # 使用核心模块处理音频
        result = self.core.process_audio_file(
            audio_path=str(audio),
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        # 输出结果
        output = CogPath(tempfile.mkdtemp()) / "output.json"
        with open(output, "w") as f:
            f.write(json.dumps(result, indent=2))
        
        return output 