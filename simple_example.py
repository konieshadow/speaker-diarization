#!/usr/bin/env python3
"""
简单使用示例
最基本的说话人分离用法演示
"""
import json
from lib.speaker_diarization_core import SpeakerDiarizationCore


def simple_example():
    """简单的使用示例"""
    
    # 音频文件路径（请替换为您的音频文件）
    audio_file = "/root/lex_ai_john_carmack_1.wav"  # 请修改为实际的音频文件路径
    
    print("🚀 开始说话人分离...")
    
    # 1. 创建核心实例
    core = SpeakerDiarizationCore()
    
    # 2. 初始化模型
    core.setup()
    
    # 3. 处理音频文件
    result = core.process_audio_file(audio_file)
    
    # 4. 显示结果
    print("✅ 处理完成!")
    print(f"检测到 {len(result.get('segments', []))} 个语音段")
    
    # 保存结果
    with open("simple_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("结果已保存到 simple_result.json")


def advanced_example():
    """高级使用示例"""
    
    audio_file = "/root/lex_ai_john_carmack_1.wav"  # 请修改为实际的音频文件路径
    
    print("🚀 开始高级说话人分离...")
    
    # 创建核心实例，自定义参数
    core = SpeakerDiarizationCore(
        model_name="pyannote/speaker-diarization-3.1",
        use_gpu=True,
        enable_compile=True
    )
    
    # 显示设备信息
    device_info = core.get_device_info()
    print("📊 设备信息:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # 初始化模型
    core.setup()
    
    # 处理音频文件，指定说话人数量范围
    result = core.process_audio_file(
        audio_file,
        min_speakers=2,
        max_speakers=5
    )
    
    print("✅ 高级处理完成!")
    
    # 详细分析结果
    if 'segments' in result:
        segments = result['segments']
        print(f"检测到 {len(segments)} 个语音段")
        
        # 显示前几个分段的详细信息
        print("\n语音段详情:")
        for i, segment in enumerate(segments[:5]):  # 只显示前5个
            print(f"  分段 {i+1}: {segment['speaker']} - {segment['start']} 到 {segment['stop']}")
        
        if len(segments) > 5:
            print(f"  ... 还有 {len(segments) - 5} 个分段")
    
    # 显示说话人统计
    if 'speakers' in result:
        speakers_info = result['speakers']
        print(f"\n说话人统计:")
        print(f"  检测到 {speakers_info['count']} 个说话人")
        print(f"  说话人标签: {', '.join(speakers_info['labels'])}")
    
    # 保存结果
    with open("advanced_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("结果已保存到 advanced_result.json")


if __name__ == "__main__":
    print("选择示例:")
    print("1. 简单示例")
    print("2. 高级示例")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    try:
        if choice == "1":
            simple_example()
        elif choice == "2":
            advanced_example()
        else:
            print("无效选择，运行简单示例...")
            simple_example()
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc() 