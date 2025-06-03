"""
Optimized speaker diarization predictor using core module
"""
import json
import tempfile
from typing import Optional

from cog import BasePredictor, Input, Path as CogPath
from lib.speaker_diarization_core import SpeakerDiarizationCore


class Predictor(BasePredictor):
    def setup(self):
        """Initialize speaker diarization core"""
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
            default="https://r2.getcastify.com/lex_ai_john_carmack_1.wav"
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
        """Run optimized prediction"""

        # Use core module to process audio
        result = self.core.process_audio_file(
            audio_path=str(audio),
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )

        # Output results
        output = CogPath(tempfile.mkdtemp()) / "output.json"
        with open(output, "w") as f:
            f.write(json.dumps(result, indent=2))
        
        return output 