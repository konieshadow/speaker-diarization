import collections
import datetime

import numpy as np


class SpeakerLabelGenerator:
    def __init__(self):
        self.speakers = {}
        self.labels = []
        self.next_speaker = ord('A')
        self.count = 0

    def get(self, name):
        if name not in self.speakers:
            current = chr(self.next_speaker)
            self.speakers[name] = current
            self.labels.append(current)
            self.next_speaker += 1
            self.count += 1
        return self.speakers[name]

    def get_all(self):
        return self.labels


class DiarizationPostProcessor:
    def __init__(self):
        self.MIN_SEGMENT_DURATION = 1.0
        self.labels = None

    def empty_result(self):
        return {
            "segments": [],
            "speakers": {
                "count": 0,
                "labels": [],
                "embeddings": {},
            },
        }

    def process_v3(self, diarization):
        """
        处理 pyannote 3.x 版本的 diarization 结果
        这个版本不需要单独的 embeddings 参数
        """
        print('正在后处理分离结果...')
        # 创建新的标签生成器
        self.labels = SpeakerLabelGenerator()

        # 处理分离结果
        clean_segments = self.clean_segments_v3(diarization)
        merged_segments = self.merge_segments(clean_segments)

        # 创建最终输出
        segments = self.format_segments(merged_segments)
        speaker_count = self.labels.count
        speaker_labels = self.labels.get_all()
        
        return {
            "segments": segments,
            "speakers": {
                "count": speaker_count,
                "labels": speaker_labels,
                "embeddings": {},  # v3 版本暂时不提供 embeddings
            },
        }

    def clean_segments_v3(self, diarization):
        """为 v3 版本清理分段，不需要 embeddings"""
        speaker_time = collections.defaultdict(float)
        total_time = 0.0
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            # 过滤掉太短的分段
            if segment.duration < self.MIN_SEGMENT_DURATION:
                continue
            speaker_time[speaker] += segment.duration
            total_time += segment.duration

        # 过滤掉说话时间太少的说话人
        # (这些可能是重叠部分被误分类为独立说话人)
        speakers = set([
            speaker
            for speaker, time in speaker_time.items()
            if time > total_time * 0.01
        ])

        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if (speaker not in speakers) or segment.duration < self.MIN_SEGMENT_DURATION:
                continue
            segments.append({
                "speaker": self.labels.get(speaker),
                "start": segment.start,
                "stop": segment.end,
            })
        return segments

    def merge_segments(self, clean_segments):
        # merge adjacent segments if they have the same speaker and are close enough
        merged = []
        for segment in clean_segments:
            if not merged:
                merged.append(segment)
                continue
            if merged[-1]["speaker"] == segment["speaker"]:
                if segment["start"] - merged[-1]["stop"] < 2.0 * self.MIN_SEGMENT_DURATION:
                    merged[-1]["stop"] = segment["stop"]
                    continue
            merged.append(segment)
        return merged

    def format_segments(self, emb_segments):
        def format_ts(ts):
            return str(datetime.timedelta(seconds=ts))

        segments = []
        for segment in emb_segments:
            segments.append({
                "speaker": segment["speaker"],
                "start": format_ts(segment["start"]),
                "stop": format_ts(segment["stop"]),
            })
        return segments
