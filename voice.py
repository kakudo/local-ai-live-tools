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
import requests
from flask import Flask, jsonify, request
import json


class RealTimeVoiceRecognizer:
    """リアルタイム音声認識クラス"""
    
    def __init__(self, model_name: str = "medium", device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: Whisperモデル名 (tiny, base, small, medium, large)
            device: 使用するデバイス ("cpu" or "cuda")
        """
        print(f"Whisperモデル '{model_name}' を読み込み中...")
        
        # デバイスの自動選択
        if device is None:
            cuda_available = torch.cuda.is_available()
            device = "cuda" if cuda_available else "cpu"
            print(f"デバイス自動選択: CUDA利用可能={cuda_available}")
            if cuda_available:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
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
        
        # 音声認識結果を蓄積するリスト
        self.recognized_texts = []
        self.text_lock = threading.Lock()
        
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
                # print("audio_data is silent, skipping transcription.")
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
                    
                    # 結果表示と蓄積
                    if text:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"[{timestamp}] {text}")
                        
                        # 認識結果をリストに蓄積
                        with self.text_lock:
                            self.recognized_texts.append({
                                'text': text,
                                'timestamp': timestamp,
                                'time': time.time()
                            })
                
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
    
    def get_recent_texts(self, since_timestamp: Optional[float] = None, limit: Optional[int] = None) -> list:
        """
        指定した条件に基づいて音声認識結果を取得
        
        Args:
            since_timestamp: この時刻以降のテキストを取得（None の場合は全て）
            limit: 取得する最大件数（新しい順、タイムスタンプフィルタ後に適用）
            
        Returns:
            list: 音声認識結果のリスト（時系列順）
        """
        with self.text_lock:
            # まず時刻でフィルタリング
            if since_timestamp is not None:
                filtered_texts = [item for item in self.recognized_texts if item['time'] > since_timestamp]
            else:
                filtered_texts = self.recognized_texts.copy()
            
            # 次に件数制限を適用（最新のN件）
            if limit is not None and limit > 0:
                filtered_texts = filtered_texts[-limit:]
            
            return filtered_texts
    
    def clear_texts(self) -> None:
        """蓄積された音声認識結果をクリア"""
        with self.text_lock:
            self.recognized_texts.clear()
    
    def get_and_clear_recent_texts(self, since_timestamp: Optional[float] = None) -> list:
        """
        指定したタイムスタンプ以降の音声認識結果を取得してからクリア
        
        Args:
            since_timestamp: この時刻以降のテキストを取得（None の場合は全て）
            
        Returns:
            list: 音声認識結果のリスト
        """
        with self.text_lock:
            if since_timestamp is None:
                result = self.recognized_texts.copy()
                self.recognized_texts.clear()
                return result
            else:
                recent_texts = [item for item in self.recognized_texts if item['time'] > since_timestamp]
                # since_timestamp以降のものを削除
                self.recognized_texts = [item for item in self.recognized_texts if item['time'] <= since_timestamp]
                return recent_texts

    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'audio'):
            self.audio.terminate()


class VoiceRecognitionServer:
    """音声認識WebAPIサーバー"""
    
    def __init__(self, model_name: str = "medium", device: Optional[str] = None, host: str = "0.0.0.0", port: int = 5000):
        """
        サーバー初期化
        
        Args:
            model_name: Whisperモデル名
            device: 使用するデバイス
            host: サーバーホスト (デフォルト: 0.0.0.0)
            port: サーバーポート (デフォルト: 5000)
        """
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.recognizer = RealTimeVoiceRecognizer(model_name=model_name, device=device)
        self.setup_routes()
        
    def setup_routes(self):
        """APIルートの設定"""
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """サーバー状態を取得"""
            return jsonify({
                'status': 'ok',
                'recording': self.recognizer.is_recording,
                'processing': self.recognizer.is_processing,
                'device': self.recognizer.device
            })
        
        @self.app.route('/start', methods=['POST'])
        def start_recording():
            """音声認識開始"""
            try:
                data = request.get_json() if request.is_json else {}
                device_index = data.get('device_index')
                
                if self.recognizer.is_recording:
                    return jsonify({'error': 'Already recording'}), 400
                
                success = self.recognizer.start_recording(device_index=device_index)
                if success:
                    # 処理スレッド開始
                    self.recognizer.is_processing = True
                    processing_thread = threading.Thread(target=self.recognizer.processing_thread)
                    processing_thread.daemon = True
                    processing_thread.start()
                    
                    return jsonify({'status': 'started', 'recording': True})
                else:
                    return jsonify({'error': 'Failed to start recording'}), 500
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/stop', methods=['POST'])
        def stop_recording():
            """音声認識停止"""
            try:
                self.recognizer.is_processing = False
                self.recognizer.stop_recording()
                return jsonify({'status': 'stopped', 'recording': False})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/texts', methods=['GET'])
        def get_texts():
            """認識されたテキストを取得"""
            try:
                since_timestamp = request.args.get('since_timestamp', type=float)
                limit = request.args.get('limit', type=int)
                
                texts = self.recognizer.get_recent_texts(
                    since_timestamp=since_timestamp,
                    limit=limit
                )
                
                return jsonify({
                    'texts': texts,
                    'count': len(texts)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/texts/clear', methods=['POST'])
        def clear_texts():
            """認識されたテキストをクリア"""
            try:
                self.recognizer.clear_texts()
                return jsonify({'status': 'cleared'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/texts/consume', methods=['POST'])
        def consume_texts():
            """認識されたテキストを取得してクリア"""
            try:
                data = request.get_json() if request.is_json else {}
                since_timestamp = data.get('since_timestamp')
                
                texts = self.recognizer.get_and_clear_recent_texts(
                    since_timestamp=since_timestamp
                )
                
                return jsonify({
                    'texts': texts,
                    'count': len(texts)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/devices', methods=['GET'])
        def get_devices():
            """利用可能な音声デバイス一覧を取得"""
            try:
                devices = []
                for i in range(self.recognizer.audio.get_device_count()):
                    device_info = self.recognizer.audio.get_device_info_by_index(i)
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'max_input_channels': device_info['maxInputChannels'],
                        'max_output_channels': device_info['maxOutputChannels']
                    })
                
                return jsonify({'devices': devices})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def run(self):
        """サーバー実行"""
        print(f"音声認識サーバーを起動中... (http://{self.host}:{self.port})")
        self.app.run(host=self.host, port=self.port, threaded=True)


class RemoteVoiceRecognizer:
    """リモート音声認識クライアント"""
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        """
        リモート音声認識クライアント初期化
        
        Args:
            server_url: 音声認識サーバーのURL
        """
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 10
        self._last_connection_check = 0
        self._connection_check_interval = 30  # 30秒ごとに接続チェック
    
    def _handle_request_error(self, e: Exception, operation: str):
        """リクエストエラーを処理"""
        if isinstance(e, requests.exceptions.ConnectionError):
            print(f"[Warning] 音声認識サーバーに接続できません ({operation}): {self.server_url}")
        elif isinstance(e, requests.exceptions.Timeout):
            print(f"[Warning] 音声認識サーバーからの応答がタイムアウトしました ({operation})")
        else:
            print(f"[Warning] 音声認識サーバーでエラーが発生しました ({operation}): {e}")
    
    def get_status(self) -> dict:
        """サーバー状態を取得"""
        try:
            response = self.session.get(f"{self.server_url}/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self._handle_request_error(e, "状態取得")
            raise
    
    def start_recording(self, device_index: Optional[int] = None) -> bool:
        """音声認識開始"""
        try:
            data = {}
            if device_index is not None:
                data['device_index'] = device_index
            
            response = self.session.post(f"{self.server_url}/start", json=data)
            return response.status_code == 200
        except Exception as e:
            self._handle_request_error(e, "録音開始")
            return False
    
    def stop_recording(self) -> bool:
        """音声認識停止"""
        try:
            response = self.session.post(f"{self.server_url}/stop")
            return response.status_code == 200
        except Exception as e:
            self._handle_request_error(e, "録音停止")
            return False
    
    def get_recent_texts(self, since_timestamp: Optional[float] = None, limit: Optional[int] = None) -> list:
        """認識されたテキストを取得"""
        try:
            params = {}
            if since_timestamp is not None:
                params['since_timestamp'] = since_timestamp
            if limit is not None:
                params['limit'] = limit
            
            response = self.session.get(f"{self.server_url}/texts", params=params)
            response.raise_for_status()
            return response.json()['texts']
        except Exception as e:
            self._handle_request_error(e, "テキスト取得")
            return []
    
    def clear_texts(self) -> bool:
        """認識されたテキストをクリア"""
        try:
            response = self.session.post(f"{self.server_url}/texts/clear")
            return response.status_code == 200
        except Exception as e:
            self._handle_request_error(e, "テキストクリア")
            return False
    
    def get_and_clear_recent_texts(self, since_timestamp: Optional[float] = None) -> list:
        """認識されたテキストを取得してクリア"""
        try:
            data = {}
            if since_timestamp is not None:
                data['since_timestamp'] = since_timestamp
            
            response = self.session.post(f"{self.server_url}/texts/consume", json=data)
            response.raise_for_status()
            return response.json()['texts']
        except Exception as e:
            self._handle_request_error(e, "テキスト取得・クリア")
            return []
    
    def get_devices(self) -> list:
        """利用可能な音声デバイス一覧を取得"""
        try:
            response = self.session.get(f"{self.server_url}/devices")
            response.raise_for_status()
            return response.json()['devices']
        except Exception as e:
            self._handle_request_error(e, "デバイス一覧取得")
            return []
    
    def is_available(self) -> bool:
        """サーバーが利用可能かチェック"""
        current_time = time.time()
        
        # 頻繁な接続チェックを避けるため、インターバルをチェック
        if current_time - self._last_connection_check < self._connection_check_interval:
            # 前回のチェックから時間が経っていない場合、簡易チェック
            try:
                response = self.session.get(f"{self.server_url}/status", timeout=3)
                return response.status_code == 200
            except Exception:
                return False
        
        # 定期的な詳細チェック
        try:
            self.get_status()
            self._last_connection_check = current_time
            return True
        except Exception:
            self._last_connection_check = current_time
            return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="リアルタイム音声認識システム")
    parser.add_argument(
        "--model", 
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisperモデル名 (デフォルト: medium)"
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
    parser.add_argument(
        "--local",
        action="store_true",
        help="ローカルモードで起動（デフォルトはWebAPIサーバーモード）"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="サーバーホスト (デフォルト: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="サーバーポート (デフォルト: 5000)"
    )
    
    args = parser.parse_args()
    
    if args.local:
        # ローカルモード
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
    else:
        # サーバーモード（デフォルト）
        server = VoiceRecognitionServer(
            model_name=args.model,
            device=args.device,
            host=args.host,
            port=args.port
        )
        
        # デバイス一覧表示
        if args.list_devices:
            server.recognizer.list_audio_devices()
            return
        
        # サーバー実行
        server.run()


if __name__ == "__main__":
    main()