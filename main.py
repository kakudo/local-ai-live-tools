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
    def __init__(self, ollama_url="http://localhost:11434", model_name="gemma3:12b", comment_model_name="deepseek-r1:8b", xml_file=DEFAULT_XML_PATH, prompt_file="prompt.md", enable_voice=True, debug_mode=False, compression_ratio=2.0, jpeg_quality=75, voice_server_url=None, persona_config=None):
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
            compression_ratio: 画像圧縮倍率 (2.0なら縦横1/2、面積1/4に圧縮, デフォルト: 2.0)
            jpeg_quality: JPEG品質 (1-100, デフォルト: 75)
            voice_server_url: リモート音声認識サーバーのURL (Noneの場合はローカル音声認識を使用)
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
            original_size = image.size
            original_width, original_height = original_size
            
            # 圧縮率に基づいて新しいサイズを計算
            new_width = int(original_width / self.compression_ratio)
            new_height = int(original_height / self.compression_ratio)
            
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
            
            # スクリーンショットを取得（指定された領域のみ）
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
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
            # 元のサイズを記録
            original_size = image.size
            
            # 画像をリサイズ
            resized_image = self.resize_image(image.copy())
            
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
    
    def send_image_analysis_to_ollama(self, image_base64):
        """
        画像の詳細説明のみを取得する（第1段階）
        
        Args:
            image_base64: Base64エンコードされた画像
            
        Returns:
            str: 画像の詳細説明テキスト（エラーの場合はエラーメッセージ）
        """
        try:
            # 画像解析専用のシンプルなプロンプト
            analysis_prompt = """この画像を詳しく分析して説明してください。

客観的かつ詳細に、見えるものをそのまま説明してください。
推測や解釈ではなく、実際に画面に表示されている内容を正確に記述してください。"""

            payload = {
                "model": self.model_name,
                "prompt": analysis_prompt,
                "images": [image_base64],
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
                print(f"[DEBUG][画像解析] [{timestamp}] {analysis_text}")
                
                # ログファイルに保存
                if self.debug_log_file:
                    with open(self.debug_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"[{timestamp}][画像解析] {analysis_text}\n")
            
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
                
                # 人格用プロンプトを生成
                enhanced_prompt = self.persona_manager.create_prompt_for_personas(
                    selected_personas, voice_context, self.prompt_file
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
        print(f"実行間隔: {interval}秒")
        print(f"Ollama URL: {self.ollama_url}")
        print(f"音声認識: {'有効' if self.enable_voice else '無効'}")
        print(f"デバッグモード: {'有効' if self.debug_mode else '無効'}")
        if self.debug_mode and self.debug_log_file:
            print(f"デバッグログ: {self.debug_log_file}")
        print("-" * 50)
        print("Ctrl+C で停止できます\n")
        
        # 音声認識を開始
        if self.enable_voice:
            self.start_voice_recognition()
        
        # XML出力スレッドを開始
        self.start_xml_output_thread()
        
        while True:
            try:
                # 現在時刻を表示
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}] スクリーンショット解析を実行中...")
                
                # スクリーンショットを取得
                screenshot = self.get_active_window_screenshot()
                
                if screenshot is None:
                    print("スクリーンショットが取得できませんでした")
                    time.sleep(interval)
                    continue
                
                # 画像をBase64に変換
                image_base64 = self.image_to_base64(screenshot)
                
                if image_base64 is None:
                    print("画像のエンコードに失敗しました")
                    time.sleep(interval)
                    continue
                
                # 【第1段階】画像の詳細解析
                print("  [段階1] 画像の詳細解析を実行中...")
                image_analysis = self.send_image_analysis_to_ollama(image_base64)
                
                if image_analysis.startswith("エラー:"):
                    print(f"画像解析エラー: {image_analysis}")
                    time.sleep(interval)
                    continue
                
                print(f"  [段階1] 完了: 画像解析結果を取得しました")
                
                # 【第2段階】音声コンテキストを取得してコメント生成
                print("  [段階2] コメント生成を実行中...")
                voice_context = self.get_voice_context()
                response = self.send_comment_generation_to_ollama(image_analysis, voice_context)
                
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
        
        # 終了処理: 音声認識とXML出力スレッドを停止
        if self.enable_voice:
            self.stop_voice_recognition()
        self.stop_xml_output_thread()


def main():
    """
    メイン関数
    """
    import sys
    import argparse
    
    try:
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
            persona_config=config.personas
        )
        
        if config.behavior.debug_mode:
            print("[Debug] デバッグモードが有効です。画面解析の詳細情報が表示されます。")
        
        # 継続的な解析を開始
        explainer.run_continuous_analysis(interval=config.behavior.analysis_interval)
        
    except Exception as e:
        print(f"アプリケーション起動エラー: {e}")


if __name__ == "__main__":
    main()
