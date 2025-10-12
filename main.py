import base64
import io
import json
import time
import requests
import threading
import queue
import random
from datetime import datetime
from PIL import Image
import pygetwindow as gw
import pyautogui
import xml.etree.ElementTree as ET
import os
from voice import RealTimeVoiceRecognizer, RemoteVoiceRecognizer
from config_manager import ConfigManager
from persona_manager import PersonaManager

# デフォルトのXMLファイルパス（プロジェクト直下の相対パス）
DEFAULT_XML_PATH = "comment.xml"

class OllamaVisionExplainer:
    def __init__(self, ollama_url="http://localhost:11434", model_name="gemma3:12b", comment_model_name="deepseek-r1:8b", xml_file=DEFAULT_XML_PATH, prompt_file="prompt.md", enable_voice=True, debug_mode=False, compression_ratio=3.0, jpeg_quality=75, voice_server_url=None, persona_config=None, image_buffer_size=30, images_per_request=10, max_logs=10):
        """
        Ollama Vision Explainer
        
        Args:
            ollama_url: OllamaサーバーのURL (デフォルト: http://localhost:11434)
            model_name: 画像解析用モデル名 (デフォルト: gemma3:12b)
            comment_model_name: コメント生成用モデル名 (デフォルト: deepseek-r1:8b)
            xml_file: ログ出力先のXMLファイル (デフォルト: comment.xml)
            prompt_file: プロンプトファイルのパス (デフォルト: prompt.md)
            enable_voice: 音声認識を有効にするかどうか (デフォルト: True)
            debug_mode: 画面解析デバッグモードを有効にするかどうか (デフォルト: False)
            compression_ratio: 画像圧縮倍率 (2.0なら縦横1/2、面積1/4に圧縮, デフォルト: 3.0)
            jpeg_quality: JPEG品質 (1-100, デフォルト: 75)
            voice_server_url: リモート音声認識サーバーのURL (Noneの場合はローカル音声認識を使用)
            persona_config: 人格設定 (PersonasConfig)
            image_buffer_size: 画像バッファサイズ (デフォルト: 30)
            images_per_request: 1回のリクエストで送信する画像数 (デフォルト: 10)
            max_logs: 要約実行のための解説蓄積上限数 (デフォルト: 10)
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.comment_model_name = comment_model_name
        self.api_url = f"{ollama_url}/api/generate"
        self.xml_file = xml_file
        self.prompt_file = prompt_file
        self.comment_counter = 0
        self.prompt_content = self.load_prompt()
        self.debug_mode = debug_mode
        
        # 画像圧縮設定
        self.compression_ratio = compression_ratio
        self.jpeg_quality = jpeg_quality
        
        # デバッグログファイル
        self.debug_log_file = "screen_analysis_debug.log" if debug_mode else None
        self.debug_analysis_file = "debug_analysis_responses.log" if debug_mode else None  # Ollama画像解析結果用
        self.debug_summary_file = "debug_summaries.log" if debug_mode else None  # 要約結果用
        
        # コメントキューシステム
        self.comment_queue = queue.Queue()
        self.xml_output_thread = None
        self.xml_thread_running = False
        
        # 音声認識機能
        self.enable_voice = enable_voice
        self.voice_server_url = voice_server_url
        self.voice_recognizer = None
        self.voice_thread = None
        self.last_ollama_request_time = time.time()
        
        # 人格管理システム
        self.persona_config = persona_config
        self.persona_manager = None
        if persona_config:
            self.persona_manager = PersonaManager(persona_config.personas_file)
        
        # 画像バッファリングシステム
        self.image_buffer_size = image_buffer_size
        self.images_per_request = images_per_request
        self.image_buffer = queue.Queue()  # スレッドセーフなキュー
        self.screenshot_thread = None
        self.screenshot_thread_running = False
        
        # 解説蓄積システム
        self.analysis_logs = []  # 解説を蓄積するリスト
        self.max_logs = max_logs
        self.summary_file = "game_summary.txt"  # 要約ファイルのパス
        
        # デバッグファイル初期化
        if debug_mode:
            self.init_debug_files()
        
        if self.enable_voice:
            self.init_voice_recognition()
    
    def remove_character_count(self, comment):
        """
        コメント末尾の文字数カウント表示を除去する
        
        Args:
            comment: str 処理対象のコメント
            
        Returns:
            str: 文字数カウント部分を除去したコメント
        """
        import re
        
        if not comment:
            return comment
            
        # 半角括弧パターン: (〇〇文字)
        comment = re.sub(r'\([0-9０-９]+文字\)$', '', comment)
        
        # 全角括弧パターン: （〇〇文字）
        comment = re.sub(r'（[0-9０-９]+文字）$', '', comment)
        
        # 半角括弧パターン: (〇〇)
        comment = re.sub(r'\([0-9０-９]+\)$', '', comment)
        
        # 全角括弧パターン: （〇〇）
        comment = re.sub(r'（[0-9０-９]+）$', '', comment)

        # 前後の空白を除去
        return comment.strip()

    def resize_image(self, image):
        """
        画像を圧縮率に基づいてリサイズして処理速度を向上させる
        
        Args:
            image: PIL.Image オリジナル画像
            
        Returns:
            PIL.Image: リサイズ済み画像
        """
        try:
            # 画像の有効性をチェック
            if image is None:
                print("[リサイズエラー] 画像がNoneです")
                return None
            
            original_size = image.size
            original_width, original_height = original_size
            
            # 元画像サイズの有効性をチェック
            if original_width <= 0 or original_height <= 0:
                print(f"[リサイズエラー] 無効な元画像サイズ: {original_size}")
                return image  # 元の画像をそのまま返す
            
            # 圧縮率に基づいて新しいサイズを計算
            new_width = int(original_width / self.compression_ratio)
            new_height = int(original_height / self.compression_ratio)
            
            # 最小サイズを保証（1x1以上）
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            # アスペクト比を維持してリサイズ
            image.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
            
            resized_size = image.size
            
            # デバッグモードの場合、リサイズ情報を表示
            if self.debug_mode:
                actual_compression = (original_size[0] * original_size[1]) / (resized_size[0] * resized_size[1])
                print(f"[画像圧縮] {original_size} → {resized_size} (面積圧縮率: {actual_compression:.2f}x)")
            
            return image
            
        except Exception as e:
            print(f"画像圧縮エラー: {e}")
            return image
    
    def init_debug_files(self):
        """
        デバッグファイルを初期化（開始時のヘッダー情報を書き込み）
        """
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 画像解析デバッグファイル初期化
            if self.debug_analysis_file:
                with open(self.debug_analysis_file, 'w', encoding='utf-8') as f:
                    f.write(f"Ollama画像解析デバッグログ\n")
                    f.write(f"開始日時: {current_time}\n")
                    f.write(f"画像解析モデル: {self.model_name}\n")
                    f.write(f"画像バッファサイズ: {self.image_buffer_size}\n")
                    f.write(f"送信画像数: {self.images_per_request}\n")
                    f.write(f"{'='*60}\n\n")
                print(f"[Debug] 画像解析デバッグファイルを初期化: {self.debug_analysis_file}")
            
            # 要約デバッグファイル初期化
            if self.debug_summary_file:
                with open(self.debug_summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"要約生成デバッグログ\n")
                    f.write(f"開始日時: {current_time}\n")
                    f.write(f"要約モデル: {self.model_name}\n")
                    f.write(f"要約トリガー: {self.max_logs}ログ蓄積時\n")
                    f.write(f"{'='*60}\n\n")
                print(f"[Debug] 要約デバッグファイルを初期化: {self.debug_summary_file}")
            
        except Exception as e:
            print(f"デバッグファイル初期化エラー: {e}")

    def load_prompt(self):
        """
        プロンプトファイルを読み込む
        
        Returns:
            str: プロンプト内容
        """
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"プロンプトファイルを読み込みました: {self.prompt_file}")
            return content
        except FileNotFoundError:
            print(f"プロンプトファイルが見つかりません: {self.prompt_file}")
            return ""
        except Exception as e:
            print(f"プロンプトファイル読み込みエラー: {e}")
            return ""
    
    def init_voice_recognition(self):
        """
        音声認識システムを初期化
        """
        try:
            if self.voice_server_url:
                print(f"🎤 リモート音声認識サーバーに接続中... ({self.voice_server_url})")
                self.voice_recognizer = RemoteVoiceRecognizer(server_url=self.voice_server_url)
                
                # サーバーの生存確認
                if not self.voice_recognizer.is_available():
                    raise Exception(f"音声認識サーバーに接続できませんでした: {self.voice_server_url}")
                    
                print("[OK] リモート音声認識システムに接続しました")
            else:
                print("🎤 ローカル音声認識システムを初期化中...")
                self.voice_recognizer = RealTimeVoiceRecognizer(model_name="medium")
                print("[OK] ローカル音声認識システムの初期化が完了しました")
                
        except Exception as e:
            print(f"[Warning] 音声認識システムの初期化に失敗しました: {e}")
            self.enable_voice = False
    
    def start_voice_recognition(self):
        """
        音声認識を別スレッドで開始
        """
        if not self.enable_voice or self.voice_recognizer is None:
            return False
        
        try:
            if isinstance(self.voice_recognizer, RemoteVoiceRecognizer):
                # リモート音声認識の場合
                print("🎤 リモート音声認識を開始します...")
                if self.voice_recognizer.start_recording():
                    print("リモート音声認識が開始されました")
                    return True
                else:
                    print("リモート音声認識の開始に失敗しました")
                    return False
            else:
                # ローカル音声認識の場合
                def voice_thread():
                    print("🎤 音声認識を開始します...")
                    if self.voice_recognizer.start_recording():
                        self.voice_recognizer.is_processing = True
                        processing_thread = threading.Thread(target=self.voice_recognizer.processing_thread)
                        processing_thread.daemon = True
                        processing_thread.start()
                        print("音声認識が開始されました")
                    else:
                        print("音声認識の開始に失敗しました")
                
                self.voice_thread = threading.Thread(target=voice_thread)
                self.voice_thread.daemon = True
                self.voice_thread.start()
                time.sleep(2)  # 初期化待ち
                return True
            
        except Exception as e:
            print(f"音声認識開始エラー: {e}")
            return False
    
    def stop_voice_recognition(self):
        """
        音声認識を停止
        """
        if self.voice_recognizer:
            try:
                if isinstance(self.voice_recognizer, RemoteVoiceRecognizer):
                    # リモート音声認識の場合
                    self.voice_recognizer.stop_recording()
                else:
                    # ローカル音声認識の場合
                    self.voice_recognizer.is_processing = False
                    self.voice_recognizer.stop_recording()
                print("[Mute] 音声認識を停止しました")
            except Exception as e:
                print(f"音声認識停止エラー: {e}")
    
    def start_screenshot_thread(self):
        """
        スクリーンショット取得スレッドを開始
        """
        if self.screenshot_thread is None or not self.screenshot_thread.is_alive():
            self.screenshot_thread_running = True
            self.screenshot_thread = threading.Thread(target=self._screenshot_worker)
            self.screenshot_thread.daemon = True
            self.screenshot_thread.start()
            print("[Thread] スクリーンショット取得スレッドを開始しました")
    
    def stop_screenshot_thread(self):
        """
        スクリーンショット取得スレッドを停止
        """
        self.screenshot_thread_running = False
        if self.screenshot_thread and self.screenshot_thread.is_alive():
            print("[Stop] スクリーンショット取得スレッドを停止中...")
            self.screenshot_thread.join(timeout=5)
            print("[OK] スクリーンショット取得スレッドを停止しました")
    
    def _screenshot_worker(self):
        """
        スクリーンショット取得ワーカースレッド（1秒間隔で実行）
        """
        print("[Start] スクリーンショット取得ワーカーを開始しました")
        
        while self.screenshot_thread_running:
            try:
                # スクリーンショットを取得
                screenshot = self.get_active_window_screenshot()
                
                if screenshot is not None:
                    # 画像をリサイズしてbase64変換
                    image_base64 = self.image_to_base64(screenshot)
                    
                    if image_base64 is not None:
                        # タイムスタンプ付きで画像データを作成
                        image_data = {
                            'timestamp': time.time(),
                            'base64': image_base64
                        }
                        
                        # バッファサイズ制限チェック
                        while self.image_buffer.qsize() >= self.image_buffer_size:
                            try:
                                # 最も古い画像を削除
                                self.image_buffer.get_nowait()
                                if self.debug_mode:
                                    print("[Buffer] 古い画像を削除（バッファサイズ制限）")
                            except queue.Empty:
                                break
                        
                        # 新しい画像をバッファに追加
                        self.image_buffer.put(image_data)
                        
                        if self.debug_mode:
                            buffer_count = self.image_buffer.qsize()
                            print(f"[Buffer] 画像を追加しました（バッファ: {buffer_count}/{self.image_buffer_size}）")
                
                # 2秒待機
                time.sleep(2.0)
                
            except Exception as e:
                print(f"スクリーンショット取得エラー: {e}")
                time.sleep(2.0)  # エラー時も1秒待機
        
        print("[Stop] スクリーンショット取得ワーカーを終了しました")
    
    def get_voice_context(self):
        """
        30秒以内かつ直近10個の音声認識結果を取得してフォーマット
        
        Returns:
            str: 音声認識結果をフォーマットした文字列
        """
        if not self.enable_voice or self.voice_recognizer is None:
            return ""
        
        try:
            # 30秒以内の発言を取得
            cutoff_time = time.time() - 30  # 30秒前
            recent_texts = self.voice_recognizer.get_recent_texts(
                since_timestamp=cutoff_time, 
                limit=10
            )
            
            if not recent_texts:
                return "配信者の発言: （直近30秒間の発言なし）"
            
            # 音声認識結果をフォーマット（時系列順）
            voice_content = "配信者の直近の発言履歴（30秒以内）: "
            for i, item in enumerate(recent_texts, 1):
                # 何秒前の発言かを表示
                seconds_ago = int(time.time() - item['time'])
                voice_content += f"[{seconds_ago}秒前] 「{item['text']}」 "
            
            return voice_content.strip()
            
        except Exception as e:
            print(f"音声コンテキスト取得エラー: {e}")
            return ""
    
    def get_active_window_screenshot(self):
        """
        アクティブウィンドウのスクリーンショットを取得
        
        Returns:
            PIL.Image: スクリーンショット画像、取得できない場合はNone
        """
        try:
            # アクティブウィンドウを取得
            active_window = gw.getActiveWindow()
            
            if active_window is None:
                print("アクティブウィンドウが見つかりません")
                return None
            
            # ウィンドウの座標とサイズを取得
            left, top, width, height = active_window.left, active_window.top, active_window.width, active_window.height
            
            # ウィンドウサイズの有効性をチェック
            if width <= 0 or height <= 0:
                if self.debug_mode:
                    print(f"[スクリーンショットエラー] 無効なウィンドウサイズ: {width}x{height}")
                return None
            
            # 最小サイズを保証
            if width < 10 or height < 10:
                if self.debug_mode:
                    print(f"[スクリーンショット警告] ウィンドウサイズが小さすぎます: {width}x{height}")
                return None
            
            # スクリーンショットを取得（指定された領域のみ）
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
            # 取得された画像の検証
            if screenshot is None:
                if self.debug_mode:
                    print("[スクリーンショットエラー] pyautogui.screenshotがNoneを返しました")
                return None
            
            return screenshot
            
        except Exception as e:
            print(f"スクリーンショット取得エラー: {e}")
            return None
    
    def image_to_base64(self, image):
        """
        PIL画像をリサイズしてBase64エンコードされた文字列に変換
        
        Args:
            image: PIL.Image オブジェクト
            
        Returns:
            str: Base64エンコードされた画像データ
        """
        try:
            # 画像の有効性をチェック
            if image is None:
                if self.debug_mode:
                    print("[画像エラー] 画像がNoneです")
                return None
            
            # 元のサイズを記録
            original_size = image.size
            
            # 画像サイズが有効かチェック
            if original_size[0] <= 0 or original_size[1] <= 0:
                if self.debug_mode:
                    print(f"[画像エラー] 無効な画像サイズ: {original_size}")
                return None
            
            # 画像をリサイズ
            resized_image = self.resize_image(image.copy())
            
            # リサイズ後の画像サイズもチェック
            if resized_image.size[0] <= 0 or resized_image.size[1] <= 0:
                if self.debug_mode:
                    print(f"[画像エラー] リサイズ後の無効なサイズ: {resized_image.size}")
                return None
            
            # RGBAの場合はRGBに変換（JPEG保存のため）
            if resized_image.mode == 'RGBA':
                rgb_image = Image.new('RGB', resized_image.size, (255, 255, 255))
                rgb_image.paste(resized_image, mask=resized_image.split()[-1])
                resized_image = rgb_image
            
            buffer = io.BytesIO()
            resized_image.save(buffer, format='JPEG', quality=self.jpeg_quality, optimize=True)
            buffer.seek(0)
            
            # ファイルサイズ情報を表示
            file_size_kb = len(buffer.getvalue()) / 1024
            if self.debug_mode:
                print(f"[JPEG圧縮] 品質: {self.jpeg_quality}, サイズ: {file_size_kb:.1f}KB")
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return image_base64
        except Exception as e:
            print(f"画像エンコードエラー: {e}")
            return None
    
    def save_debug_image(self, image):
        """
        デバッグ用に画像をimagesフォルダに保存
        
        Args:
            image: PIL.Image オブジェクト
            
        Returns:
            str: 保存されたファイルのパス
        """
        try:
            # imagesフォルダが存在しない場合は作成
            images_dir = "images"
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            
            # タイムスタンプ付きファイル名を生成
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # マイクロ秒の最後3桁を削除
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(images_dir, filename)
            
            # 画像を保存
            image.save(filepath, format='PNG')
            print(f"[Image] デバッグ用画像を保存しました: {filepath}")
            
            return filepath
        except Exception as e:
            print(f"デバッグ画像保存エラー: {e}")
            return None
    
    def create_prompt_with_prompt(self, base_prompt):
        """
        ベースプロンプトにプロンプト情報を追加したプロンプトを作成

        Args:
            base_prompt: ベースとなるプロンプト
            
        Returns:
            str: プロンプト情報を含む完全なプロンプト
        """
        if self.prompt_content:
            return f"{base_prompt}\n\n{self.prompt_content}"
        return base_prompt
    
    def load_previous_summary(self):
        """
        前回の要約ファイルを読み込む
        
        Returns:
            str: 前回の要約内容（ファイルがない場合は空文字列）
        """
        try:
            if os.path.exists(self.summary_file):
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if self.debug_mode:
                    print(f"[Summary] 前回の要約を読み込みました: {len(content)}文字")
                return content
            else:
                if self.debug_mode:
                    print("[Summary] 要約ファイルが存在しません（初回実行）")
                return ""
        except Exception as e:
            print(f"要約ファイル読み込みエラー: {e}")
            return ""
    
    def get_summary_context(self):
        """
        コメント生成用に要約情報を取得・フォーマット
        
        Returns:
            str: コメント生成に使用する要約情報（ファイルがない場合は空文字列）
        """
        try:
            if os.path.exists(self.summary_file):
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    # タイムスタンプ行を除去して要約本文のみを抽出
                    lines = content.split('\n')
                    summary_lines = []
                    
                    for line in lines:
                        # タイムスタンプ行をスキップ
                        if line.startswith('[更新日時:') and line.endswith(']'):
                            continue
                        summary_lines.append(line)
                    
                    summary_text = '\n'.join(summary_lines).strip()
                    
                    if summary_text:
                        if self.debug_mode:
                            print(f"[Summary] コメント生成に要約情報を使用: {len(summary_text)}文字")
                        
                        # コメント生成用にフォーマット
                        formatted_summary = f"""これまでのゲーム進行状況:
{summary_text}

**重要**: 上記の要約情報も考慮して、現在の画面状況に応じた適切なコメントを生成してください。過去の流れを踏まえた発言や、進展に対する反応も含めることができます。"""
                        
                        return formatted_summary
            
            return ""
            
        except Exception as e:
            if self.debug_mode:
                print(f"要約コンテキスト取得エラー: {e}")
            return ""
    
    def save_summary_to_file(self, summary_text):
        """
        要約をファイルに保存
        
        Args:
            summary_text: 要約テキスト
        """
        try:
            # タイムスタンプ付きで保存
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content = f"[更新日時: {timestamp}]\n{summary_text}"
            
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"[Summary] 要約をファイルに保存しました: {self.summary_file}")
            if self.debug_mode:
                print(f"[Summary] 保存内容: {len(summary_text)}文字")
                
        except Exception as e:
            print(f"要約ファイル保存エラー: {e}")
    
    def generate_summary_request_to_ollama(self, analysis_logs):
        """
        蓄積された解説ログを要約するリクエストをOllamaに送信
        
        Args:
            analysis_logs: 解説ログのリスト
            
        Returns:
            str: 要約結果（エラーの場合はエラーメッセージ）
        """
        try:
            # 前回の要約を読み込み
            previous_summary = self.load_previous_summary()
            
            # プロンプトを構築
            summary_prompt = """以下は、あるゲームのプレイ中にリアルタイムで記録された解説ログです。
このログをもとに、これまでのストーリーやプレイの流れ・達成したこと・発生した重要イベントなどを簡潔にまとめてください。
今までの要約がある場合は、それにも注意して、内容が重複しないようにしてください。

目的：
- 現在までのゲームプレイの全体像を把握するため
- 新しく視聴する人にも内容が伝わるようにするため

制約：
- 500〜1000文字程度
- 一般的な視聴者が理解しやすい言葉を使用
- 登場人物や地名、イベント名はできるだけ明記
- 単なる列挙ではなく、「流れ」としてまとめる

今までの要約：
"""
            
            if previous_summary:
                summary_prompt += f"{previous_summary}\n\n"
            else:
                summary_prompt += "（これが最初の要約です）\n\n"
            
            summary_prompt += "今回のログ：\n"
            
            # ログを番号付きで追加
            for i, log in enumerate(analysis_logs, 1):
                summary_prompt += f"【ログ{i}】\n{log}\n\n"
            
            payload = {
                "model": self.model_name,
                "prompt": summary_prompt,
                "images": [],  # 画像は送信しない
                "stream": False
            }
            
            if self.debug_mode:
                print(f"[Summary] 要約リクエストを送信中... (ログ数: {len(analysis_logs)})")
                print(f"[Summary] プロンプト長: {len(summary_prompt)}文字")
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            summary_text = result.get("response", "").strip()
            
            if self.debug_mode:
                print(f"[Summary] 要約生成完了: {len(summary_text)}文字")
                
                # 要約専用デバッグファイルに詳細保存
                if self.debug_summary_file:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.save_summary_debug_log(timestamp, len(analysis_logs), summary_text)
            
            return summary_text
            
        except requests.exceptions.ConnectionError:
            return "エラー: Ollamaサーバーに接続できません。"
        except requests.exceptions.Timeout:
            return "エラー: 要約リクエストがタイムアウトしました。"
        except requests.exceptions.RequestException as e:
            return f"エラー: 要約API呼び出しに失敗しました: {e}"
        except Exception as e:
            return f"エラー: 要約生成中に予期しないエラーが発生しました: {e}"
    
    def process_analysis_log(self, analysis_text):
        """
        解説テキストを蓄積し、必要に応じて要約を実行
        
        Args:
            analysis_text: 画像解析結果のテキスト
            
        Returns:
            bool: 要約を実行した場合True
        """
        # ログを蓄積
        self.analysis_logs.append(analysis_text)
        
        if self.debug_mode:
            print(f"[Logs] 解説ログを蓄積: {len(self.analysis_logs)}/{self.max_logs}")
        
        # 上限に達した場合は要約を実行
        if len(self.analysis_logs) >= self.max_logs:
            print(f"📄 [Summary] ログが{self.max_logs}個に達しました。要約を生成します...")
            
            # 要約を生成
            summary_result = self.generate_summary_request_to_ollama(self.analysis_logs)
            
            if not summary_result.startswith("エラー:"):
                # 要約をファイルに保存
                self.save_summary_to_file(summary_result)
                print(f"📄 [Summary] 要約生成完了! ({len(summary_result)}文字)")
                
                # ログをクリア
                self.analysis_logs = []
                print(f"📄 [Logs] 解説ログをクリア - 新しいサイクルを開始")
                
                return True
            else:
                print(f"❌ [Summary] 要約生成エラー: {summary_result}")
                return False
        
        return False

    def send_image_analysis_to_ollama(self):
        """
        バッファから画像を取得して画像の詳細説明を取得する（第1段階）
        
        Returns:
            str: 画像の詳細説明テキスト（エラーの場合はエラーメッセージ）
        """
        try:
            # バッファから指定数の画像を取得
            images_to_send = []
            buffer_images = []
            
            # バッファから最新の画像を取得（時系列は逆順で格納されているため）
            for _ in range(min(self.images_per_request, self.image_buffer.qsize())):
                try:
                    image_data = self.image_buffer.get_nowait()
                    buffer_images.append(image_data)
                except queue.Empty:
                    break
            
            if not buffer_images:
                return "エラー: 解析可能な画像がバッファにありません。"
            
            # 時系列順（古い順）にソートして送信用リストに変換
            buffer_images.sort(key=lambda x: x['timestamp'])
            images_to_send = [img['base64'] for img in buffer_images]
            
            if self.debug_mode:
                timestamps = [datetime.fromtimestamp(img['timestamp']).strftime("%H:%M:%S") for img in buffer_images]
                print(f"[Buffer] 解析用画像を取得: {len(images_to_send)}枚 (時系列: {' → '.join(timestamps)})")
            
            # 複数画像に対応したプロンプト
            if len(images_to_send) > 1:
                analysis_prompt = f"""提供された動画を分析して説明してください。

動画中で見えるものを詳細に説明し、変化や動きがあれば特に注目してください。
客観的かつ詳細に、実際に画面に表示されている内容を正確に記述してください。
2000字以内で記述してください。"""
            else:
                analysis_prompt = """この画像を詳しく分析して説明してください。

客観的かつ詳細に、見えるものをそのまま説明してください。
推測や解釈ではなく、実際に画面に表示されている内容を正確に記述してください。
2000字以内で記述してください。"""

            payload = {
                "model": self.model_name,
                "prompt": analysis_prompt,
                "images": images_to_send,
                "stream": False
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            analysis_text = result.get("response", "")
            
            if self.debug_mode:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[DEBUG][画像解析] [{timestamp}] ({len(images_to_send)}枚の画像) {analysis_text}")
                
                # 従来のログファイルに保存
                if self.debug_log_file:
                    with open(self.debug_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{timestamp}][画像解析][{len(images_to_send)}枚] {analysis_text}\n")
                
                # 画像解析専用デバッグファイルに詳細保存
                if self.debug_analysis_file:
                    self.save_analysis_debug_log(timestamp, len(images_to_send), analysis_text)
            
            return analysis_text.strip()
            
        except requests.exceptions.ConnectionError:
            return "エラー: Ollamaサーバーに接続できません。"
        except requests.exceptions.Timeout:
            return "エラー: 画像解析リクエストがタイムアウトしました。"
        except requests.exceptions.RequestException as e:
            return f"エラー: 画像解析API呼び出しに失敗しました: {e}"
        except Exception as e:
            return f"エラー: 画像解析中に予期しないエラーが発生しました: {e}"
    
    def send_comment_generation_to_ollama(self, image_analysis_text, voice_context=""):
        """
        画像解析結果を基にコメント生成を行う（第2段階）
        
        Args:
            image_analysis_text: 第1段階で取得した画像の詳細説明
            voice_context: 音声認識コンテキスト（オプション）
            
        Returns:
            dict: 解析されたJSON形式のレスポンス（エラーの場合は文字列）
        """
        try:
            # 人格管理システムを使用する場合
            if self.persona_manager:
                # ランダムに人格を選択
                always_include = self.persona_config.always_include if self.persona_config else []
                exclude_from_random = always_include.copy()
                
                # 固定人格を追加
                selected_personas = []
                for persona_id in always_include:
                    persona = self.persona_manager.get_persona(persona_id)
                    if persona:
                        selected_personas.append(persona)
                
                # 残りをランダム選択
                remaining_count = self.persona_config.select_count - len(selected_personas)
                if remaining_count > 0:
                    random_personas = self.persona_manager.get_random_personas(
                        remaining_count, exclude=exclude_from_random
                    )
                    selected_personas.extend(random_personas)
                
                # 要約情報を取得（存在する場合）
                summary_context = self.get_summary_context()
                
                # 人格用プロンプトを生成（要約情報も含める）
                enhanced_prompt = self.persona_manager.create_prompt_for_personas(
                    selected_personas, voice_context, self.prompt_file, summary_context
                )
                
                # 画像解析結果を組み込み
                enhanced_prompt = f"""===画像解析結果（第1段階で取得）===
{image_analysis_text}
===

上記の画像解析結果を基に、以下の指示に従ってコメントを生成してください。

**注意**: 画像解析結果にプログラミング環境、ブラウザ、デスクトップ、オフィスソフトなどが含まれている場合は、ゲーム画面でないと判定し、全て「none」でコメントしてください。

{enhanced_prompt}"""
                
                # JSON形式を動的に構築
                format_props = {}
                for persona in selected_personas:
                    format_props[persona.persona_id] = {"type": "string"}
                
                required_fields = [persona.persona_id for persona in selected_personas]
            
            payload = {
                "model": self.comment_model_name,
                "prompt": enhanced_prompt,
                "images": [],  # 画像は送信しない（テキストベース処理）
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": format_props,
                    "required": required_fields
                }
            }
            
            # デバッグ出力を追加
            if self.debug_mode:
                print(f"[DEBUG] コメント生成モデル: {self.comment_model_name}")
                if summary_context:
                    print(f"[DEBUG] 要約情報を含むコメント生成を実行")
                print(f"[DEBUG] 送信するプロンプト (最初の800文字):\n{enhanced_prompt[:800]}...")
                print(f"[DEBUG] プロンプト全体の文字数: {len(enhanced_prompt)}")
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result.get("response", "")
            
            # デバッグ: 生のレスポンスを確認
            if self.debug_mode:
                print(f"[DEBUG] Ollamaからの生レスポンス:\n{raw_response}")
            
            # JSONレスポンスを解析（選択された人格情報を渡す）
            expected_persona_ids = [persona.persona_id for persona in selected_personas] if 'selected_personas' in locals() else None
            parsed_response = self.parse_json_response(raw_response, expected_persona_ids)
            
            return parsed_response
            
        except requests.exceptions.ConnectionError:
            return "エラー: Ollamaサーバーに接続できません。"
        except requests.exceptions.Timeout:
            return "エラー: コメント生成リクエストがタイムアウトしました。"
        except requests.exceptions.RequestException as e:
            return f"エラー: コメント生成API呼び出しに失敗しました: {e}"
        except Exception as e:
            return f"エラー: コメント生成中に予期しないエラーが発生しました: {e}"

    def parse_json_response(self, raw_response, expected_persona_ids=None):
        """
        OllamaからのJSONレスポンスを解析
        
        Args:
            raw_response: Ollamaからの生のレスポンス文字列
            expected_persona_ids: 期待される人格IDのリスト（人格管理システム用）
            
        Returns:
            dict: 解析されたJSON（エラーの場合は文字列）
        """
        try:
            # JSONを解析
            parsed_json = json.loads(raw_response)
            
            # 期待される構造を検証
            if isinstance(parsed_json, dict):
                # 人格管理システムを使用している場合
                if expected_persona_ids:
                    # 期待される人格IDのいずれかが含まれているかをチェック
                    received_keys = set(parsed_json.keys())
                    expected_keys = set(expected_persona_ids)
                    
                    # 少なくとも1つの期待される人格が含まれていればOK
                    if received_keys.intersection(expected_keys):
                        return parsed_json
                    else:
                        print(f"[Warning] 期待される人格が見つかりません。期待: {expected_persona_ids}, 受信: {list(received_keys)}")
                        return parsed_json  # エラーにせず、そのまま返す（後続処理で適切にフィルタされる）
                
                # 従来の固定人格システム（後方互換性）
                else:
                    # 固定人格の既知キー
                    known_persona_keys = ["listener", "safety", "expert", "fan1", "fan2", "anti", "jikatari", "ero", "shogaku", "question", "kaomoji", "safety_monitor", "game_expert"]
                    
                    # 受信したキーのいずれかが既知の人格キーか確認
                    received_keys = set(parsed_json.keys())
                    known_keys = set(known_persona_keys)
                    
                    if received_keys.intersection(known_keys):
                        # 旧形式（safety_monitor, game_expert）があれば新形式に変換
                        if "safety_monitor" in parsed_json or "game_expert" in parsed_json:
                            converted = {}
                            # 既存のキーを新形式にマッピング
                            key_mapping = {
                                "safety_monitor": "safety",
                                "game_expert": "expert"
                            }
                            
                            for old_key, new_key in key_mapping.items():
                                if old_key in parsed_json:
                                    value = parsed_json[old_key]
                                    converted[new_key] = value if isinstance(value, str) else value.get("comment", "")
                            
                            # その他のキーもそのまま追加
                            for key, value in parsed_json.items():
                                if key not in ["safety_monitor", "game_expert"]:
                                    converted[key] = value
                            
                            return converted
                        else:
                            # 新形式の部分的な人格データとしてそのまま返す
                            return parsed_json
                    else:
                        print(f"[Warning] 未知の人格構造: {received_keys}")
                        return parsed_json  # そのまま返して後続処理に委ねる
            else:
                return f"JSON形式エラー: オブジェクト形式ではありません"
                
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON解析エラー: {e}")
            print(f"生のレスポンス: {raw_response}")
            # JSONパースに失敗した場合は、従来通りの文字列として処理
            return raw_response
        except Exception as e:
            return f"レスポンス解析エラー: {e}"
    
    def handle_debug_output(self, screen_analysis):
        """
        デバッグモードでの画面解析結果を処理
        
        Args:
            screen_analysis: 画面解析の詳細な説明文字列
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            debug_message = f"[{timestamp}] 画面解析: {screen_analysis}"
            
            # コンソールに出力（Windowsの文字化け対策）
            print(f"[DEBUG] {debug_message}")
            
            # ログファイルに保存
            if self.debug_log_file:
                with open(self.debug_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{debug_message}\n")
                    
        except Exception as e:
            print(f"デバッグ出力処理エラー: {e}")
    
    def save_analysis_debug_log(self, timestamp, image_count, analysis_text):
        """
        画像解析結果をデバッグファイルに詳細保存
        
        Args:
            timestamp: タイムスタンプ文字列
            image_count: 解析した画像枚数
            analysis_text: 解析結果テキスト
        """
        try:
            with open(self.debug_analysis_file, 'a', encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"日時: {timestamp}\n")
                f.write(f"送信画像数: {image_count}枚\n")
                f.write(f"モデル: {self.model_name}\n")
                f.write(f"解析結果:\n{analysis_text}\n")
                f.write(f"{'='*60}\n\n")
                
        except Exception as e:
            print(f"画像解析デバッグログ保存エラー: {e}")
    
    def save_summary_debug_log(self, timestamp, log_count, summary_text):
        """
        要約結果をデバッグファイルに詳細保存
        
        Args:
            timestamp: タイムスタンプ文字列
            log_count: 蓄積ログ数
            summary_text: 要約結果テキスト
        """
        try:
            with open(self.debug_summary_file, 'a', encoding='utf-8') as f:
                f.write(f"{'='*60}\n")
                f.write(f"日時: {timestamp}\n")
                f.write(f"蓄積ログ数: {log_count}個\n")
                f.write(f"要約モデル: {self.model_name}\n")
                f.write(f"要約結果:\n{summary_text}\n")
                f.write(f"{'='*60}\n\n")
                
        except Exception as e:
            print(f"要約デバッグログ保存エラー: {e}")

    
    def write_to_xml_log(self, response_data):
        """
        [非推奨] 旧来のXML書き込みメソッド
        新しいキューベースシステム（add_comments_to_queue）を使用してください
        """
        print("[Warning] write_to_xml_log は非推奨です。add_comments_to_queue を使用してください")
        self.add_comments_to_queue(response_data)
    
    def update_last_request_time(self):
        """
        最後のOllamaリクエスト時間を更新（現在は直近10件を常に送信するため実質的に未使用）
        """
        self.last_ollama_request_time = time.time()
    
    def is_non_game_comment(self, comment):
        """
        コメントがゲーム以外の内容を示しているかを判定
        
        Args:
            comment: コメント文字列
            
        Returns:
            bool: ゲーム以外の内容の場合True
        """
        non_game_keywords = [
            # プログラミング関連
            "python", "コード", "クラス", "関数", "メソッド", "変数", "import", "def ", "class ", 
            "vscode", "ide", "エディタ", "ハイライト", "構文", "実装", "デバッグ",
            # ブラウザ・ウェブ関連  
            "ブラウザ", "chrome", "firefox", "タブ", "url", "ウェブ", "検索", "google",
            # オフィス関連
            "word", "excel", "powerpoint", "文書", "スプレッドシート", "プレゼン",
            # システム関連
            "デスクトップ", "フォルダ", "ファイル", "エクスプローラ", "設定", "コントロールパネル"
        ]
        
        comment_lower = comment.lower()
        return any(keyword in comment_lower for keyword in non_game_keywords)
    
    def add_comments_to_queue(self, response_data):
        """
        レスポンスデータからコメントを抽出してキューに追加
        
        Args:
            response_data: dict または str
                - dict: JSON形式の複数人格からのコメント
                - str: 従来通りの単一コメント
        """
        if isinstance(response_data, dict):
            # 人格管理システムを使用する場合の人格情報を取得
            if self.persona_manager:
                persona_info = {}
                for persona_id, persona in self.persona_manager.get_all_personas().items():
                    persona_info[persona_id] = {
                        "handle": persona.handle,
                        "persona": persona.name
                    }
            else:
                # 従来の固定人格情報
                persona_info = {
                    "listener": {"handle": "リスナーbot", "persona": "リスナー"},
                    "safety": {"handle": "安全監視bot", "persona": "安全監視員"}, 
                    "expert": {"handle": "ゲーム専門bot", "persona": "ゲーム専門家"},
                    "fan1": {"handle": "ファン1", "persona": "配信者ファン1"},
                    "fan2": {"handle": "ファン2", "persona": "配信者ファン2"},
                    "anti": {"handle": "アンチ", "persona": "配信者アンチ"},
                    "jikatari": {"handle": "店長", "persona": "自分語り"},
                    "ero": {"handle": "エロ爺", "persona": "エロ爺"},
                    "shogaku": {"handle": "小学生", "persona": "小学生"},
                    "question": {"handle": "質問者", "persona": "質問の人"},
                    "kaomoji": {"handle": "顔文字", "persona": "顔文字の人"},
                    # 互換性のため旧形式も対応
                    "safety_monitor": {"handle": "安全監視bot", "persona": "安全監視員"}, 
                    "game_expert": {"handle": "ゲーム専門bot", "persona": "ゲーム専門家"}
                }
            
            # 有効なコメントを収集
            valid_comments = []
            filtered_count = 0
            
            for persona, comment_data in response_data.items():
                # デバッグフィールドはスキップ
                if persona in ["screen_analysis"]:
                    continue
                
                # 人格情報が存在するかチェック
                if persona in persona_info:
                    # prompt.md形式では直接文字列、旧形式ではオブジェクト
                    if isinstance(comment_data, dict):
                        comment = comment_data.get("comment", "")
                    else:
                        comment = comment_data
                    
                    # 基本的な有効性チェック
                    if (comment and comment.strip() != "" and 
                        "none" not in comment.lower()):
                        
                        # 非ゲーム内容の二重チェック
                        if self.is_non_game_comment(comment):
                            print(f"[Filter] 非ゲーム内容として除外: {persona} - {comment}")
                            filtered_count += 1
                            continue
                        
                        comment_item = {
                            "handle": persona_info[persona]["handle"],
                            "persona": persona_info[persona]["persona"],
                            "comment": comment
                            # timestampは削除 - XML出力時に生成する
                        }
                        valid_comments.append(comment_item)
                else:
                    # 未知の人格IDの場合のログ出力
                    if self.debug_mode:
                        print(f"[Debug] 未知の人格ID: {persona} - スキップしました")
            
            # 有効なコメントがある場合のみランダムな順序でキューに追加
            if valid_comments:
                import random
                random.shuffle(valid_comments)
                
                for comment_item in valid_comments:
                    self.comment_queue.put(comment_item)
                    print(f"[Queue] キューに追加: {comment_item['persona']} - {comment_item['comment']}")
            else:
                reason = f"非ゲーム画面（{filtered_count}件フィルタ）" if filtered_count > 0 else "有効なコメントなし"
                print(f"[XML] ゲーム画面でないため、コメントをスキップしました ({reason})")
        else:
            # 従来の単一コメント
            if (response_data and response_data.strip() != "" and 
                "none" not in response_data.lower() and
                not self.is_non_game_comment(response_data)):
                comment_item = {
                    "handle": "安全bot",
                    "persona": "レガシー", 
                    "comment": response_data
                    # timestampは削除 - XML出力時に生成する
                }
                self.comment_queue.put(comment_item)
                print(f"[Queue] キューに追加: レガシー - {response_data}")
            else:
                print("[XML] 非ゲーム内容またはnoneのため、コメントをスキップしました")
    
    def start_xml_output_thread(self):
        """
        XML出力用スレッドを開始
        """
        if self.xml_output_thread is None or not self.xml_output_thread.is_alive():
            self.xml_thread_running = True
            self.xml_output_thread = threading.Thread(target=self._xml_output_worker)
            self.xml_output_thread.daemon = True
            self.xml_output_thread.start()
            print("[Thread] XML出力スレッドを開始しました")
    
    def stop_xml_output_thread(self):
        """
        XML出力用スレッドを停止
        """
        self.xml_thread_running = False
        if self.xml_output_thread and self.xml_output_thread.is_alive():
            print("[Stop] XML出力スレッドを停止中...")
            # 終了シグナルをキューに送信
            self.comment_queue.put(None)
            self.xml_output_thread.join(timeout=5)
            print("[OK] XML出力スレッドを停止しました")
    
    def _xml_output_worker(self):
        """
        XML出力用ワーカースレッド（別スレッドで動作）
        """
        print("[Start] XML出力ワーカーを開始しました")
        
        while self.xml_thread_running:
            try:
                # キューの状態に応じて待機間隔を調整
                queue_size = self.comment_queue.qsize()
                
                if queue_size == 0:
                    # キューが空の場合：1-2秒のランダム間隔
                    wait_time = random.uniform(1.0, 2.0)
                    print(f"💤 キューが空です。{wait_time:.1f}秒待機...")
                elif queue_size <= 5:
                    # 少しコメントがある場合：0.8-5秒
                    wait_time = random.uniform(0.8, 5.0)
                elif queue_size <= 10:
                    # コメントが溜まっている場合：0.5-3.5秒
                    wait_time = random.uniform(0.5, 3.5)
                else:
                    # コメントが多く溜まっている場合：0.3-1秒
                    wait_time = random.uniform(0.3, 1.0)
                
                # 指定時間待機（途中で終了シグナルをチェック）
                time.sleep(wait_time)
                
                # キューからコメントを取得
                try:
                    comment_item = self.comment_queue.get(timeout=0.1)
                    
                    # 終了シグナルチェック
                    if comment_item is None:
                        break
                    
                    # XMLファイルに書き込み
                    self._write_single_comment_to_xml(comment_item)
                    
                    # キュータスク完了を通知
                    self.comment_queue.task_done()
                    
                except queue.Empty:
                    # タイムアウト（通常の動作）
                    continue
                    
            except Exception as e:
                print(f"XML出力ワーカーエラー: {e}")
                time.sleep(1)  # エラー時は1秒待機
        
        print("[Stop] XML出力ワーカーを終了しました")
    
    def _write_single_comment_to_xml(self, comment_item):
        """
        単一のコメントをXMLファイルに書き込み
        
        Args:
            comment_item: dict コメント情報
                - handle: ハンドル名
                - persona: 人格名
                - comment: コメント内容
        """
        try:
            # XMLファイルが存在しない場合は作成
            if not os.path.exists(self.xml_file):
                with open(self.xml_file, 'w', encoding='utf-8') as f:
                    f.write('<?xml version="1.0" encoding="utf-8"?>\n<log>\n</log>')
            
            # XML出力時点の現在時刻をUNIX時間として取得
            unix_time = int(time.time())
            handle = comment_item["handle"]
            comment = comment_item["comment"]
            persona = comment_item["persona"]
            
            # コメント末尾の文字数カウント表示を除去
            comment_cleaned = self.remove_character_count(comment)
            
            # XMLコメント要素を作成
            comment_xml = f'  <comment no="0" time="{unix_time}" owner="0" service="youtubelive" handle="{handle}">{comment_cleaned}</comment>'
            
            # ファイルを読み込み
            with open(self.xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # </log>の直前に新しいコメントを挿入
            content = content.replace('</log>', f'{comment_xml}\n</log>')
            
            # ファイルに書き戻し
            with open(self.xml_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # カウンターをインクリメント
            self.comment_counter += 1
            
            # 文字数カウント除去前後で異なる場合のみ情報を表示
            if comment != comment_cleaned:
                print(f"[XML] 文字数カウント除去: '{comment}' → '{comment_cleaned}'")
            print(f"[XML] XML出力: {persona} - {comment_cleaned}")
            
        except Exception as e:
            print(f"XML書き込みエラー: {e}")
    
    def run_continuous_analysis(self, interval=1):
        """
        定期的にスクリーンショットを取得してOllamaに送信し続ける
        
        Args:
            interval: 実行間隔（秒）
        """
        print(f"Ollama Vision Explainer を開始しました")
        print(f"画像解析モデル: {self.model_name}")
        print(f"コメント生成モデル: {self.comment_model_name}")
        print(f"解析間隔: {interval}秒")
        print(f"画像バッファサイズ: {self.image_buffer_size}")
        print(f"送信画像数: {self.images_per_request}")
        print(f"要約機能: {self.max_logs}個のログで自動要約")
        print(f"要約ファイル: {self.summary_file}")
        print(f"Ollama URL: {self.ollama_url}")
        print(f"音声認識: {'有効' if self.enable_voice else '無効'}")
        print(f"デバッグモード: {'有効' if self.debug_mode else '無効'}")
        if self.debug_mode:
            if self.debug_log_file:
                print(f"デバッグログ: {self.debug_log_file}")
            if self.debug_analysis_file:
                print(f"画像解析ログ: {self.debug_analysis_file}")
            if self.debug_summary_file:
                print(f"要約ログ: {self.debug_summary_file}")
        print("-" * 50)
        print("Ctrl+C で停止できます\n")
        
        # 音声認識を開始
        if self.enable_voice:
            self.start_voice_recognition()
        
        # XML出力スレッドを開始
        self.start_xml_output_thread()
        
        # スクリーンショット取得スレッドを開始
        self.start_screenshot_thread()
        
        while True:
            try:
                # 現在時刻を表示
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] 画像バッファ解析を実行中...")
                
                # バッファに十分な画像があるかチェック
                buffer_size = self.image_buffer.qsize()
                if buffer_size < self.images_per_request:
                    # 不足している画像数分だけ待機（1秒間隔でスクリーンショット取得されるため）
                    shortage = self.images_per_request - buffer_size
                    wait_time = shortage * 1.0  # 不足数 × 1秒
                    
                    if buffer_size == 0:
                        print(f"  画像バッファが空です。{self.images_per_request}枚必要 - {wait_time:.0f}秒待機")
                    else:
                        print(f"  画像バッファ不足: {buffer_size}/{self.images_per_request} 枚 - {wait_time:.0f}秒待機")
                    
                    time.sleep(wait_time)
                    continue
                
                # 【第1段階】バッファから画像を取得して詳細解析
                print(f"  [段階1] バッファから{self.images_per_request}枚の画像解析を実行中...")
                image_analysis = self.send_image_analysis_to_ollama()
                
                if image_analysis.startswith("エラー:"):
                    print(f"画像解析エラー: {image_analysis}")
                    time.sleep(interval)
                    continue
                
                print(f"  [段階1] 完了: 画像解析結果を取得しました")
                
                # 解説を蓄積し、必要に応じて要約を実行
                print(f"  [ログ蓄積] 解説を蓄積中: {len(self.analysis_logs) + 1}/{self.max_logs}")
                summary_generated = self.process_analysis_log(image_analysis)
                
                # 要約生成の有無に関わらず、コメント生成も実行
                print("  [段階2] コメント生成を実行中...")
                voice_context = self.get_voice_context()
                response = self.send_comment_generation_to_ollama(image_analysis, voice_context)
                
                if summary_generated:
                    print("  📄 要約も並行して生成されました")
                
                # 結果を表示
                print(f"[Screenshot] 最終結果:")
                if isinstance(response, dict):
                    # JSON形式の複数人格レスポンス
                    for persona, comment_data in response.items():
                        # デバッグフィールドはスキップ
                        if persona in ["screen_analysis"]:
                            continue
                        
                        persona_names = {
                            "listener": "リスナー",
                            "safety": "安全監視員",
                            "expert": "ゲーム専門家",
                            "fan1": "配信者ファン1",
                            "fan2": "配信者ファン2",
                            "anti": "配信者アンチ",
                            "jikatari": "自分語り",
                            "ero": "エロ爺",
                            "shogaku": "小学生",
                            "question": "質問の人",
                            "kaomoji": "顔文字の人",
                            # 旧形式との互換性
                            "safety_monitor": "安全監視員",
                            "game_expert": "ゲーム専門家"
                        }
                        
                        # コメント内容を取得
                        if isinstance(comment_data, dict):
                            comment = comment_data.get("comment", "")
                        else:
                            comment = comment_data
                        
                        persona_name = persona_names.get(persona, persona)
                        print(f"  {persona_name}: {comment}")
                    
                    # コメントをキューに追加（XML出力は別スレッドで順次実行）
                    self.add_comments_to_queue(response)
                else:
                    # エラーメッセージや従来の文字列レスポンス
                    print(f"{response}")
                    if not response.startswith("エラー:"):
                        self.add_comments_to_queue(response)
                
                print("-" * 50)
                
                # 指定された間隔で待機
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n\nアプリケーションを終了します...")
                break
            except Exception as e:
                print(f"予期しないエラー: {e}")
                time.sleep(interval)
        
        # 終了処理: すべてのスレッドを停止
        if self.enable_voice:
            self.stop_voice_recognition()
        self.stop_screenshot_thread()
        self.stop_xml_output_thread()


def main():
    """
    メイン関数
    """
    import sys
    import argparse
    
    try:
        print("Ollama Vision Explainer を起動しています...30秒待機します。その間にゲーム画面を起動してください")
        time.sleep(10.0)
        print("残り20秒...")
        time.sleep(10.0)
        print("残り10秒...")
        time.sleep(10.0)
        print("起動します")
        # コマンドライン引数の解析
        parser = argparse.ArgumentParser(description="リアルタイム画面解析・コメント生成システム")
        
        # 重要な設定のみコマンドライン引数として残す
        parser.add_argument("--config", "-c", help="設定ファイルのパス (デフォルト: config.yaml)")
        parser.add_argument("--debug", "-d", action="store_true", help="デバッグモードを有効にする")
        parser.add_argument("--no-voice", action="store_true", help="音声認識を無効にする")
        parser.add_argument("--ollama-url", help="OllamaサーバーのURL")
        parser.add_argument("--voice-server", help="リモート音声認識サーバーのURL")
        parser.add_argument("--xml-file", help="XMLコメントファイルの保存先")
        parser.add_argument("--interval", type=float, help="スクリーンショット解析の実行間隔（秒）")
        parser.add_argument("--create-config", action="store_true", help="config.sample.yaml を config.yaml にコピーして終了")
        parser.add_argument("--create-personas", action="store_true", help="personas.sample.yaml を personas.yaml にコピーして終了")
        
        args = parser.parse_args()
        
        # サンプル設定ファイル作成
        if args.create_config:
            config_manager = ConfigManager(suppress_warnings=True)
            config_manager.create_sample_config()
            return
        
        # サンプル人格ファイル作成
        if args.create_personas:
            persona_manager = PersonaManager(suppress_warnings=True)  # 警告を抑制
            persona_manager.create_personas_file()
            return
        
        # 設定ファイル読み込み（デフォルトでconfig.yamlを使用）
        config_file = args.config or "config.yaml"
        config_manager = ConfigManager(config_file)
        
        # コマンドライン引数で設定をオーバーライド
        config = config_manager.override_with_args(
            ollama_url=args.ollama_url,
            voice_server_url=args.voice_server,
            xml_file=args.xml_file,
            enable_voice=not args.no_voice if args.no_voice else None,
            debug_mode=args.debug if args.debug else None,
            analysis_interval=args.interval
        )
        
        # 設定を表示
        config_manager.print_config()
        
        # アプリケーションを初期化（設定ファイルベース）
        explainer = OllamaVisionExplainer(
            ollama_url=config.environment.ollama_url,
            model_name=config.models.image_analysis_model,
            comment_model_name=config.models.comment_generation_model,
            xml_file=config.environment.xml_file,
            prompt_file=config.system.prompt_file,
            enable_voice=config.behavior.enable_voice,
            debug_mode=config.behavior.debug_mode,
            compression_ratio=config.performance.image.compression_ratio,
            jpeg_quality=config.performance.image.jpeg_quality,
            voice_server_url=config.environment.voice_server_url,
            persona_config=config.personas,
            image_buffer_size=config.performance.image.buffer_size,
            images_per_request=config.performance.image.images_per_request,
            max_logs=config.system.max_logs
        )
        
        if config.behavior.debug_mode:
            print("[Debug] デバッグモードが有効です。画面解析の詳細情報が表示されます。")
        
        # 継続的な解析を開始
        explainer.run_continuous_analysis(interval=config.behavior.analysis_interval)
        
    except Exception as e:
        print(f"アプリケーション起動エラー: {e}")


if __name__ == "__main__":
    main()
