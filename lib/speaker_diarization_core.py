"""
Speaker diarization core module
Provides reusable speaker diarization functionality with local model caching and GPU optimization
"""
import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any

import static_ffmpeg
from pyannote.audio import Pipeline

from .diarization import DiarizationPostProcessor
from .audio import AudioPreProcessor


class SpeakerDiarizationCore:
    """Speaker diarization core class"""
    
    def __init__(self, 
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 use_gpu: bool = True,
                 enable_compile: bool = True):
        """
        Initialize speaker diarization core
        
        Args:
            model_name: Model name
            use_gpu: Whether to use GPU
            enable_compile: Whether to enable PyTorch compilation optimization
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_compile = enable_compile
        
        # Initialize components
        self.diarization = None
        self.diarization_post = DiarizationPostProcessor()
        self.audio_pre = AudioPreProcessor()
        
        # Device information
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
    def setup(self):
        """Load pre-downloaded model"""
        print("Initializing speaker diarization system...")
        
        # Initialize static-ffmpeg
        print("Initializing static-ffmpeg...")
        try:
            ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            print(f"✅ static-ffmpeg initialization completed: {ffmpeg_path}")
        except Exception as e:
            print(f"⚠️ static-ffmpeg initialization failed: {e}")
            raise RuntimeError(f"static-ffmpeg initialization failed: {e}")

        with open('/app/hf_token', 'r') as f:
            hf_token = f.read().strip()

            # Load pipeline
            print(f"Loading speaker diarization pipeline: {self.model_name}")
            self.diarization = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=hf_token
            )

        if self.diarization is None:
            raise RuntimeError("Model initialization failed, exiting program")
        
        # GPU optimization
        if self.use_gpu:
            print(f"Moving pipeline to device: {self.device}")
            self.diarization.to(self.device)
            
            # Enable CUDA optimization
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cudnn.benchmark = True
            print("✅ GPU optimization enabled")
        
        print("✅ Speaker diarization system initialization completed!")

    def process_audio_file(self, 
                          audio_path: str,
                          num_speakers: Optional[int] = None,
                          min_speakers: Optional[int] = None,
                          max_speakers: Optional[int] = None) -> Dict[str, Any]:
        """
        Process audio file for speaker diarization
        
        Args:
            audio_path: Audio file path
            num_speakers: Specified number of speakers
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            
        Returns:
            Diarization result dictionary
        """
        if self.diarization is None:
            raise RuntimeError("Model not initialized, please call setup() method first")
            
        # Preprocess audio
        print(f"Preprocessing audio file: {audio_path}")
        self.audio_pre.process(audio_path)

        if self.audio_pre.error:
            print(f"Audio preprocessing error: {self.audio_pre.error}")
            result = self.diarization_post.empty_result()
        else:
            # Build parameters
            kwargs = {}
            if num_speakers is not None:
                kwargs['num_speakers'] = num_speakers
            if min_speakers is not None:
                kwargs['min_speakers'] = min_speakers
            if max_speakers is not None:
                kwargs['max_speakers'] = max_speakers
                
            # Run diarization
            result = self._run_diarization(**kwargs)

        # Clean up temporary files
        self.audio_pre.cleanup()
        
        return result

    def _run_diarization(self, **kwargs) -> Dict[str, Any]:
        """Run speaker diarization with parameter support"""
        print('Running speaker diarization...')
        
        with torch.inference_mode():
            diarization = self.diarization(
                self.audio_pre.output_path, 
                **kwargs
            )
        
        return self.diarization_post.process_v3(diarization)

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "model_name": self.model_name,
        } 