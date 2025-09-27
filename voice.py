#!/usr/bin/env python3
"""
リアルタイム音声認識システム
Whisperを使用してマイクからの音声をリアルタイムで文字起こしします。
"""

import argparse
import queue
import sys
import threading
import time
from typing import Optional

import numpy as np
import pyaudio
import whisper
import torch


class RealTimeVoiceRecognizer:
    """リアルタイム音声認識クラス"""
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: Whisperモデル名 (tiny, base, small, medium, large)
            device: 使用するデバイス ("cpu" or "cuda")
        """
        print(f"Whisperモデル '{model_name}' を読み込み中...")
        
        # デバイスの自動選択
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = whisper.load_model(model_name, device=device)
        print(f"モデル読み込み完了 (デバイス: {device})")
        
        # 音声設定
        self.sample_rate = 16000
        self.chunk_duration = 3  # 秒
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.format = pyaudio.paFloat32
        self.channels = 1
        
        # PyAudio初期化
        self.audio = pyaudio.PyAudio()
        
        # 音声データキュー
        self.audio_queue = queue.Queue()
        
        # 制御フラグ
        self.is_recording = False
        self.is_processing = False
        
        print(f"音声設定: {self.sample_rate}Hz, {self.chunk_duration}秒チャンク")
    
    def list_audio_devices(self):
        """利用可能な音声デバイスを一覧表示"""
        print("\n利用可能な音声デバイス:")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            print(f"  {i}: {device_info['name']} "
                  f"(入力: {device_info['maxInputChannels']}, "
                  f"出力: {device_info['maxOutputChannels']})")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """音声コールバック関数"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # キューに音声データを追加
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            # キューが満杯の場合は古いデータを削除
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data)
            except queue.Empty:
                pass
        
        return (None, pyaudio.paContinue)
    
    def start_recording(self, device_index: Optional[int] = None):
        """録音開始"""
        try:
            # デフォルトデバイスの情報を取得
            if device_index is None:
                device_info = self.audio.get_default_input_device_info()
                device_index = device_info['index']
            else:
                device_info = self.audio.get_device_info_by_index(device_index)
            
            print(f"使用デバイス: {device_info['name']}")
            
            # 音声ストリーム開始
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            print("録音開始...")
            
        except Exception as e:
            print(f"録音開始エラー: {e}")
            self.list_audio_devices()
            return False
        
        return True
    
    def stop_recording(self):
        """録音停止"""
        if self.is_recording:
            self.is_recording = False
            self.stream.stop_stream()
            self.stream.close()
            print("録音停止")
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> str:
        """音声チャンクを処理して文字起こし"""
        try:
            # 音声の正規化
            audio_data = audio_data.astype(np.float32)
            
            # 無音チェック（閾値以下は処理しない）
            if np.abs(audio_data).mean() < 0.001:
                return ""
            
            # Whisperで音声認識
            result = self.model.transcribe(
                audio_data,
                language="ja",  # 日本語に設定
                task="transcribe"
            )
            
            return result["text"].strip()
            
        except Exception as e:
            print(f"音声処理エラー: {e}")
            return ""
    
    def processing_thread(self):
        """音声処理スレッド"""
        audio_buffer = np.array([], dtype=np.float32)
        
        while self.is_processing:
            try:
                # キューから音声データを取得
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # バッファが十分溜まったら処理
                if len(audio_buffer) >= self.chunk_size:
                    # 処理用データを切り出し
                    process_data = audio_buffer[:self.chunk_size]
                    audio_buffer = audio_buffer[self.chunk_size:]
                    
                    # 文字起こし実行
                    text = self.process_audio_chunk(process_data)
                    
                    # 結果表示
                    if text:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"[{timestamp}] {text}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"処理スレッドエラー: {e}")
    
    def run(self, device_index: Optional[int] = None):
        """メイン実行"""
        print("\n=== リアルタイム音声認識システム ===")
        print("Ctrl+C で終了")
        
        # 録音開始
        if not self.start_recording(device_index):
            return
        
        # 処理スレッド開始
        self.is_processing = True
        processing_thread = threading.Thread(target=self.processing_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        try:
            # メインループ
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n終了中...")
            
        finally:
            # クリーンアップ
            self.is_processing = False
            self.stop_recording()
            self.audio.terminate()
            print("システム終了")
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'audio'):
            self.audio.terminate()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="リアルタイム音声認識システム")
    parser.add_argument(
        "--model", 
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisperモデル名 (デフォルト: base)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="使用デバイス (自動選択)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="利用可能な音声デバイスを一覧表示"
    )
    parser.add_argument(
        "--device-index",
        type=int,
        help="使用する音声デバイスのインデックス"
    )
    
    args = parser.parse_args()
    
    # 音声認識システム初期化
    recognizer = RealTimeVoiceRecognizer(
        model_name=args.model,
        device=args.device
    )
    
    # デバイス一覧表示
    if args.list_devices:
        recognizer.list_audio_devices()
        return
    
    # システム実行
    recognizer.run(device_index=args.device_index)


if __name__ == "__main__":
    main()