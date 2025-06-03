#!/usr/bin/env python3
"""
Simple usage example
Basic speaker diarization usage demonstration
"""
import json
from lib.speaker_diarization_core import SpeakerDiarizationCore


def simple_example():
    """Simple usage example"""
    
    # Audio file path (please replace with your audio file)
    audio_file = "/root/lex_ai_john_carmack_1.wav"  # Please modify to actual audio file path
    
    print("🚀 Starting speaker diarization...")
    
    # 1. Create core instance
    core = SpeakerDiarizationCore()
    
    # 2. Initialize model
    core.setup()
    
    # 3. Process audio file
    result = core.process_audio_file(audio_file)
    
    # 4. Display results
    print("✅ Processing completed!")
    print(f"Detected {len(result.get('segments', []))} speech segments")
    
    # Save results
    with open("simple_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("Results saved to simple_result.json")


def advanced_example():
    """Advanced usage example"""
    
    audio_file = "/root/lex_ai_john_carmack_1.wav"  # Please modify to actual audio file path
    
    print("🚀 Starting advanced speaker diarization...")
    
    # Create core instance with custom parameters
    core = SpeakerDiarizationCore(
        model_name="pyannote/speaker-diarization-3.1",
        use_gpu=True,
        enable_compile=True
    )
    
    # Display device information
    device_info = core.get_device_info()
    print("📊 Device information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Initialize model
    core.setup()
    
    # Process audio file with specified speaker count range
    result = core.process_audio_file(
        audio_file,
        min_speakers=2,
        max_speakers=5
    )
    
    print("✅ Advanced processing completed!")
    
    # Detailed result analysis
    if 'segments' in result:
        segments = result['segments']
        print(f"Detected {len(segments)} speech segments")
        
        # Display detailed information for first few segments
        print("\nSpeech segment details:")
        for i, segment in enumerate(segments[:5]):  # Only show first 5
            print(f"  Segment {i+1}: {segment['speaker']} - {segment['start']} to {segment['stop']}")
        
        if len(segments) > 5:
            print(f"  ... and {len(segments) - 5} more segments")
    
    # Display speaker statistics
    if 'speakers' in result:
        speakers_info = result['speakers']
        print(f"\nSpeaker statistics:")
        print(f"  Detected {speakers_info['count']} speakers")
        print(f"  Speaker labels: {', '.join(speakers_info['labels'])}")
    
    # Save results
    with open("advanced_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("Results saved to advanced_result.json")


if __name__ == "__main__":
    print("Choose example:")
    print("1. Simple example")
    print("2. Advanced example")
    
    choice = input("Please enter your choice (1 or 2): ").strip()
    
    try:
        if choice == "1":
            simple_example()
        elif choice == "2":
            advanced_example()
        else:
            print("Invalid choice, running simple example...")
            simple_example()
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc() 