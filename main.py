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
from voice import RealTimeVoiceRecognizer

COMMENT_PATH = "C:\\MultiCommentViewer\\CommentGenerator0.0.8b\\anzen-live-helper\\public\\comment.xml"

class OllamaVisionExplainer:
    def __init__(self, ollama_url="http://localhost:11434", model_name="gemma3:12b", comment_model_name="deepseek-r1:8b", xml_file=COMMENT_PATH, prompt_file="prompt.md", enable_voice=True, debug_mode=False):
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
        
        # デバッグログファイル
        self.debug_log_file = "screen_analysis_debug.log" if debug_mode else None
        
        # コメントキューシステム
        self.comment_queue = queue.Queue()
        self.xml_output_thread = None
        self.xml_thread_running = False
        
        # 音声認識機能
        self.enable_voice = enable_voice
        self.voice_recognizer = None
        self.voice_thread = None
        self.last_ollama_request_time = time.time()
        
        if self.enable_voice:
            self.init_voice_recognition()
    
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
            print("🎤 音声認識システムを初期化中...")
            self.voice_recognizer = RealTimeVoiceRecognizer(model_name="medium")
            print("[OK] 音声認識システムの初期化が完了しました")
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
        PIL画像をBase64エンコードされた文字列に変換
        
        Args:
            image: PIL.Image オブジェクト
            
        Returns:
            str: Base64エンコードされた画像データ
        """
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
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
            # 画像解析結果を含む完全なプロンプトを作成
            voice_section = f"\n\n音声入力から認識されたテキスト:\n{voice_context}" if voice_context else ""
            
            # 画像解析結果を組み込んだプロンプト
            enhanced_prompt = f"""{self.prompt_content}

===画像解析結果（第1段階で取得）===
{image_analysis_text}
===

上記の画像解析結果を基に、プロンプトの指示に従って11人の人格からのコメントを生成してください。{voice_section}

**注意**: 画像解析結果にプログラミング環境、ブラウザ、デスクトップ、オフィスソフトなどが含まれている場合は、ゲーム画面でないと判定し、全て「none」でコメントしてください。

**重要**: 出力は必ず以下のJSON形式で、各フィールドには実際のコメント文字列を入力してください：
```json
{{
  "listener": "実際のコメント文字列",
  "safety": "実際のコメント文字列", 
  "expert": "実際のコメント文字列",
  "fan1": "実際のコメント文字列",
  "fan2": "実際のコメント文字列",
  "anti": "実際のコメント文字列",
  "jikatari": "実際のコメント文字列",
  "ero": "実際のコメント文字列",
  "shogaku": "実際のコメント文字列",
  "question": "実際のコメント文字列",
  "kaomoji": "実際のコメント文字列"
}}
```"""
            
            # prompt.mdの出力形式に合わせたフォーマット（11人構成）
            format_props = {
                "listener": {"type": "string"},
                "safety": {"type": "string"},
                "expert": {"type": "string"},
                "fan1": {"type": "string"},
                "fan2": {"type": "string"},
                "anti": {"type": "string"},
                "jikatari": {"type": "string"},
                "ero": {"type": "string"},
                "shogaku": {"type": "string"},
                "question": {"type": "string"},
                "kaomoji": {"type": "string"}
            }
            
            required_fields = ["listener", "safety", "expert", "fan1", "fan2", "anti", "jikatari", "ero", "shogaku", "question", "kaomoji"]
            
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
            
            # JSONレスポンスを解析
            parsed_response = self.parse_json_response(raw_response)
            
            return parsed_response
            
        except requests.exceptions.ConnectionError:
            return "エラー: Ollamaサーバーに接続できません。"
        except requests.exceptions.Timeout:
            return "エラー: コメント生成リクエストがタイムアウトしました。"
        except requests.exceptions.RequestException as e:
            return f"エラー: コメント生成API呼び出しに失敗しました: {e}"
        except Exception as e:
            return f"エラー: コメント生成中に予期しないエラーが発生しました: {e}"
    
    def send_to_ollama(self, image_base64, image=None, prompt=""):
        """
        [非推奨] 旧来の統合処理メソッド
        新しい2段階処理（send_image_analysis_to_ollama + send_comment_generation_to_ollama）を使用してください
        
        Args:
            image_base64: Base64エンコードされた画像
            image: PIL.Image オブジェクト（デバッグ保存用、オプション）
            prompt: 送信するプロンプト
            
        Returns:
            dict: 解析されたJSON形式のレスポンス（エラーの場合は文字列）
        """
        try:
            # デバッグ用に画像を保存
            if image is not None:
                self.save_debug_image(image)
            
            # 音声認識結果を取得
            voice_context = self.get_voice_context()
            
            # ファイルからプロンプトを補完する
            full_prompt = self.create_prompt_with_prompt(prompt)

            # 音声情報を含むプロンプトを補完する
            if voice_context:
                enhanced_prompt = f"{full_prompt}\n\n-----\n以下の配信者の発言履歴も考慮してコメントしてください。\n※直近30秒以内の発言で、[]内は何秒前の発言かを示しています。\n\n{voice_context}\n-----"
            else:
                enhanced_prompt = f"{full_prompt}\n\n配信者は直近30秒間発言していません。"

            # デバッグ: 送信するプロンプトを確認
            if self.debug_mode:
                print(f"[DEBUG] コメント生成モデル: {self.comment_model_name}")
                print(f"[DEBUG] 送信するプロンプト (最初の500文字):\n{enhanced_prompt[:500]}...")
                print(f"[DEBUG] プロンプト全体の文字数: {len(enhanced_prompt)}")
            
            
            # 11人構成用のフォーマット
            format_props = {
                "listener": {"type": "string"},
                "safety": {"type": "string"},
                "expert": {"type": "string"},
                "fan1": {"type": "string"},
                "fan2": {"type": "string"},
                "anti": {"type": "string"},
                "jikatari": {"type": "string"},
                "ero": {"type": "string"},
                "shogaku": {"type": "string"},
                "question": {"type": "string"},
                "kaomoji": {"type": "string"}
            }
            
            required_fields = ["listener", "safety", "expert", "fan1", "fan2", "anti", "jikatari", "ero", "shogaku", "question", "kaomoji"]
            
            payload = {
                "model": self.comment_model_name,
                "prompt": enhanced_prompt,
                "images": [image_base64],
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": format_props,
                    "required": required_fields
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            raw_response = result.get("response", "")
            
            # デバッグ: 生のレスポンスを確認
            if self.debug_mode:
                print(f"[DEBUG] Ollamaからの生レスポンス:\n{raw_response}")
            
            # リクエスト送信後の処理
            self.update_last_request_time()
            
            # JSONレスポンスを解析
            parsed_response = self.parse_json_response(raw_response)
            

            
            return parsed_response
            
        except requests.exceptions.ConnectionError:
            return "エラー: Ollamaサーバーに接続できません。Ollamaが起動していることを確認してください。"
        except requests.exceptions.Timeout:
            return "エラー: コメント生成リクエストがタイムアウトしました。"
        except requests.exceptions.RequestException as e:
            return f"エラー: API呼び出しに失敗しました: {e}"
        except Exception as e:
            return f"エラー: 予期しないエラーが発生しました: {e}"
    
    def parse_json_response(self, raw_response):
        """
        OllamaからのJSONレスポンスを解析
        
        Args:
            raw_response: Ollamaからの生のレスポンス文字列
            
        Returns:
            dict: 解析されたJSON（エラーの場合は文字列）
        """
        try:
            # JSONを解析
            parsed_json = json.loads(raw_response)
            
            # 期待される構造を検証
            if isinstance(parsed_json, dict):
                # 必要なキーが存在するかチェック（11人構成）
                expected_keys = ["listener", "safety", "expert", "fan1", "fan2", "anti", "jikatari", "ero", "shogaku", "question", "kaomoji"]
                
                # キーの互換性チェック（新形式）
                if all(key in parsed_json for key in expected_keys):
                    return parsed_json
                elif "safety_monitor" in parsed_json and "game_expert" in parsed_json:
                    # 旧3人形式の場合、新11人形式に変換（残り8人は"none"で埋める）
                    converted = {
                        "listener": parsed_json.get("listener", {"comment": ""}) if isinstance(parsed_json.get("listener"), dict) else parsed_json.get("listener", ""),
                        "safety": parsed_json.get("safety_monitor", {"comment": ""}) if isinstance(parsed_json.get("safety_monitor"), dict) else parsed_json.get("safety_monitor", ""),
                        "expert": parsed_json.get("game_expert", {"comment": ""}) if isinstance(parsed_json.get("game_expert"), dict) else parsed_json.get("game_expert", ""),
                        "fan1": "none",
                        "fan2": "none", 
                        "anti": "none",
                        "jikatari": "none",
                        "ero": "none",
                        "shogaku": "none",
                        "question": "none",
                        "kaomoji": "none"
                    }
                    return converted
                else:
                    print(f"[Warning] JSON構造が期待と異なります: {parsed_json}")
                    return f"JSON構造エラー: 期待されるキー({expected_keys})が不足しています"
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
            # JSON形式の複数人格レスポンス（11人構成）
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
                    # キューが空の場合：1-4秒のランダム間隔
                    wait_time = random.uniform(1.0, 2.0)
                    print(f"💤 キューが空です。{wait_time:.1f}秒待機...")
                elif queue_size <= 5:
                    # 少しコメントがある場合：0.8-2秒
                    wait_time = random.uniform(0.8, 5.0)
                elif queue_size <= 10:
                    # コメントが溜まっている場合：0.5-1.5秒
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
            
            # XMLコメント要素を作成
            comment_xml = f'  <comment no="0" time="{unix_time}" owner="0" service="youtubelive" handle="{handle}">{comment}</comment>'
            
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
            
            print(f"[XML] XML出力: {persona} - {comment}")
            
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
    
    try:
        # コマンドライン引数でデバッグモードを確認
        debug_mode = "--debug" in sys.argv or "-d" in sys.argv
        
        # アプリケーションを初期化
        explainer = OllamaVisionExplainer(
            model_name="gemma3:12b",  # 1段階目：画像解析用
            comment_model_name="gemma3:12b",  # 2段階目：コメント生成用（一時的に同じモデル）
            debug_mode=debug_mode
        )
        
        if debug_mode:
            print("[Debug] デバッグモードが有効です。画面解析の詳細情報が表示されます。")
        
        # 継続的な解析を開始（5秒間隔）
        explainer.run_continuous_analysis(interval=1)
        
    except Exception as e:
        print(f"アプリケーション起動エラー: {e}")


if __name__ == "__main__":
    main()
