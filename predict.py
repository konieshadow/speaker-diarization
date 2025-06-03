"""
优化的说话人分离预测器，支持本地模型缓存
"""
import json
import tempfile
import torch
import os
from pathlib import Path
from typing import Optional

from cog import BasePredictor, Input, Path as CogPath
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download

from lib.diarization import DiarizationPostProcessor
from lib.audio import AudioPreProcessor


class Predictor(BasePredictor):
    def setup(self):
        """加载模型，优先使用本地缓存"""
        
        # 设置缓存目录
        self.cache_dir = os.getenv("HF_CACHE_DIR", "/root/.cache/huggingface/hub")
        self.model_name = "pyannote/speaker-diarization-3.1"
        
        # 检查并确保模型已下载
        self._ensure_models_downloaded()
        
        # 加载管道
        print("Loading speaker diarization pipeline...")
        self.diarization = Pipeline.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # GPU 优化
        if torch.cuda.is_available():
            print("Moving pipeline to GPU...")
            self.diarization.to(torch.device("cuda"))
            
            # 启用优化
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cudnn.benchmark = True
            
        # 编译优化（PyTorch 2.x）
        if hasattr(torch, 'compile'):
            try:
                print("Compiling pipeline for optimization...")
                self.diarization = torch.compile(
                    self.diarization, 
                    mode="reduce-overhead"
                )
            except Exception as e:
                print(f"Compilation failed, continuing without: {e}")
        
        # 初始化后处理器
        self.diarization_post = DiarizationPostProcessor()
        self.audio_pre = AudioPreProcessor()
        
        print("Setup completed successfully!")

    def _ensure_models_downloaded(self):
        """确保所有必需的模型已下载"""
        required_models = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/segmentation-3.0"
        ]
        
        for model_id in required_models:
            if not self._is_model_cached(model_id):
                print(f"Downloading {model_id}...")
                try:
                    snapshot_download(
                        repo_id=model_id,
                        cache_dir=self.cache_dir,
                        resume_download=True
                    )
                    print(f"✅ Downloaded {model_id}")
                except Exception as e:
                    print(f"❌ Failed to download {model_id}: {e}")
                    raise

    def _is_model_cached(self, model_id: str) -> bool:
        """检查模型是否已缓存"""
        # 简化的缓存检查
        model_path = Path(self.cache_dir) / f"models--{model_id.replace('/', '--')}"
        return model_path.exists() and any(model_path.iterdir())

    def run_diarization(self, **kwargs):
        """运行说话人分离，支持参数传递"""
        print('Running speaker diarization...')
        
        with torch.inference_mode():
            diarization = self.diarization(
                self.audio_pre.output_path, 
                **kwargs
            )
        
        return self.diarization_post.process_v3(diarization)

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

        # 预处理音频
        self.audio_pre.process(audio)

        if self.audio_pre.error:
            print(f"Audio preprocessing error: {self.audio_pre.error}")
            result = self.diarization_post.empty_result()
        else:
            # 构建参数
            kwargs = {}
            if num_speakers is not None:
                kwargs['num_speakers'] = num_speakers
            if min_speakers is not None:
                kwargs['min_speakers'] = min_speakers
            if max_speakers is not None:
                kwargs['max_speakers'] = max_speakers
                
            # 运行分离
            result = self.run_diarization(**kwargs)

        # 清理
        self.audio_pre.cleanup()

        # 输出结果
        output = CogPath(tempfile.mkdtemp()) / "output.json"
        with open(output, "w") as f:
            f.write(json.dumps(result, indent=2))
        
        return output 