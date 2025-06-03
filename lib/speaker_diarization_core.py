"""
说话人分离核心模块
提供可复用的说话人分离功能，支持本地模型缓存和GPU优化
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
    """说话人分离核心类"""
    
    def __init__(self, 
                 model_name: str = "pyannote/speaker-diarization-3.1",
                 use_gpu: bool = True,
                 enable_compile: bool = True):
        """
        初始化说话人分离核心
        
        Args:
            model_name: 模型名称
            use_gpu: 是否使用GPU
            enable_compile: 是否启用PyTorch编译优化
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.enable_compile = enable_compile
        
        # 初始化组件
        self.diarization = None
        self.diarization_post = DiarizationPostProcessor()
        self.audio_pre = AudioPreProcessor()
        
        # 设备信息
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
    def setup(self):
        """加载预下载的模型"""
        print("正在初始化说话人分离系统...")
        
        # 初始化 static-ffmpeg
        print("正在初始化 static-ffmpeg...")
        try:
            ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            print(f"✅ static-ffmpeg 初始化完成: {ffmpeg_path}")
        except Exception as e:
            print(f"⚠️ static-ffmpeg 初始化失败: {e}")
            raise RuntimeError(f"static-ffmpeg 初始化失败: {e}")

        with open('/app/hf_token', 'r') as f:
            hf_token = f.read().strip()

            # 加载管道
            print(f"正在加载说话人分离管道: {self.model_name}")
            self.diarization = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=hf_token
            )

        if self.diarization is None:
            raise RuntimeError("模型初始化失败，退出程序")
        
        # GPU 优化
        if self.use_gpu:
            print(f"正在将管道移动到device: {self.device}")
            self.diarization.to(self.device)
            
            # 启用CUDA优化
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cudnn.benchmark = True
            print("✅ GPU优化已启用")
        
        print("✅ 说话人分离系统初始化完成!")

    def process_audio_file(self, 
                          audio_path: str,
                          num_speakers: Optional[int] = None,
                          min_speakers: Optional[int] = None,
                          max_speakers: Optional[int] = None) -> Dict[str, Any]:
        """
        处理音频文件进行说话人分离
        
        Args:
            audio_path: 音频文件路径
            num_speakers: 指定说话人数量
            min_speakers: 最小说话人数量
            max_speakers: 最大说话人数量
            
        Returns:
            分离结果字典
        """
        if self.diarization is None:
            raise RuntimeError("模型未初始化，请先调用setup()方法")
            
        # 预处理音频
        print(f"正在预处理音频文件: {audio_path}")
        self.audio_pre.process(audio_path)

        if self.audio_pre.error:
            print(f"音频预处理错误: {self.audio_pre.error}")
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
            result = self._run_diarization(**kwargs)

        # 清理临时文件
        self.audio_pre.cleanup()
        
        return result

    def _run_diarization(self, **kwargs) -> Dict[str, Any]:
        """运行说话人分离，支持参数传递"""
        print('正在运行说话人分离...')
        
        with torch.inference_mode():
            diarization = self.diarization(
                self.audio_pre.output_path, 
                **kwargs
            )
        
        return self.diarization_post.process_v3(diarization)

    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "model_name": self.model_name,
        } 