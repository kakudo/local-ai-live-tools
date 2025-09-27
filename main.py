import base64
import io
import json
import time
import requests
from datetime import datetime
from PIL import Image
import pygetwindow as gw
import pyautogui
import xml.etree.ElementTree as ET
import os

COMMENT_PATH = "C:\\MultiCommentViewer\\CommentGenerator0.0.8b\\comment.xml"

class OllamaVisionExplainer:
    def __init__(self, ollama_url="http://localhost:11434", model_name="gemma3:12b", xml_file=COMMENT_PATH, persona_file="anzen_bot_persona.md"):
        """
        Ollama Vision Explainer
        
        Args:
            ollama_url: OllamaサーバーのURL (デフォルト: http://localhost:11434)
            model_name: 使用するモデル名 (デフォルト: gemma3:12b)
            xml_file: ログ出力先のXMLファイル (デフォルト: comment.xml)
            persona_file: ペルソナファイルのパス (デフォルト: anzen_bot_persona.md)
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.api_url = f"{ollama_url}/api/generate"
        self.xml_file = xml_file
        self.persona_file = persona_file
        self.comment_counter = 0
        self.persona_content = self.load_persona()
    
    def load_persona(self):
        """
        ペルソナファイルを読み込む
        
        Returns:
            str: ペルソナの内容、読み込めない場合は空文字列
        """
        try:
            if os.path.exists(self.persona_file):
                with open(self.persona_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                print(f"📋 ペルソナファイルを読み込みました: {self.persona_file}")
                return content
            else:
                print(f"⚠️ ペルソナファイルが見つかりません: {self.persona_file}")
                return ""
        except Exception as e:
            print(f"ペルソナファイル読み込みエラー: {e}")
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
    
    def create_prompt_with_persona(self, base_prompt):
        """
        ベースプロンプトにペルソナ情報を追加したプロンプトを作成
        
        Args:
            base_prompt: ベースとなるプロンプト
            
        Returns:
            str: ペルソナ情報を含む完全なプロンプト
        """
#         if self.persona_content:
#             full_prompt = f"""あなたは以下のペルソナを持つキャラクターです：

# {self.persona_content}

# このキャラクターとして、以下の指示に従ってください：
# {base_prompt}"""
#             return full_prompt
#         else:
        return base_prompt
    
    def send_to_ollama(self, image_base64, prompt="あなたはゲームの配信を見ているリスナーです。画面に映っている物に対してコメントをしてください。画面上の構造物や視界で安全上問題があると思われる場合はその事についてコメントしてください。画面がゲーム画面でないと思われる場合や、特にコメントが無い場合は「none」とだけ返答してください。日本語で回答してください。余計な言葉を付けずにコメントだけ１言、１行で出力してください。"):
        """
        OllamaのAPIに画像とプロンプトを送信
        
        Args:
            image_base64: Base64エンコードされた画像
            prompt: 送信するプロンプト
            
        Returns:
            str: Ollamaからのレスポンス
        """
        try:
            # ペルソナ情報を含む完全なプロンプトを作成
            full_prompt = self.create_prompt_with_persona(prompt)
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "images": [image_base64],
                "stream": False
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "レスポンスが空です")
            
        except requests.exceptions.ConnectionError:
            return "エラー: Ollamaサーバーに接続できません。Ollamaが起動していることを確認してください。"
        except requests.exceptions.Timeout:
            return "エラー: リクエストがタイムアウトしました。"
        except requests.exceptions.RequestException as e:
            return f"エラー: API呼び出しに失敗しました: {e}"
        except Exception as e:
            return f"エラー: 予期しないエラーが発生しました: {e}"
    
    def write_to_xml_log(self, response_text):
        """
        レスポンスをXMLファイルのlogタグ内に追加（可読性重視の手動書き込み）
        
        Args:
            response_text: Ollamaからのレスポンステキスト
        """
        try:
            # XMLファイルが存在しない場合は作成
            if not os.path.exists(self.xml_file):
                with open(self.xml_file, 'w', encoding='utf-8') as f:
                    f.write('<?xml version="1.0" encoding="utf-8"?>\n<log>\n</log>')
            
            # 現在のUNIX時間を取得
            unix_time = int(time.time())
            
            # XMLコメント要素を作成
            comment_xml = f'  <comment no="0" time="{unix_time}" owner="0" service="youtubelive" handle="安全bot">{response_text}</comment>'
            
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
            
            print(f"📝 ログをXMLファイルに保存しました (No.{self.comment_counter - 1})")
            
        except Exception as e:
            print(f"XMLログ書き込みエラー: {e}")
    
    def run_continuous_analysis(self, interval=1):
        """
        定期的にスクリーンショットを取得してOllamaに送信し続ける
        
        Args:
            interval: 実行間隔（秒）
        """
        print(f"Ollama Vision Explainer を開始しました")
        print(f"モデル: {self.model_name}")
        print(f"実行間隔: {interval}秒")
        print(f"Ollama URL: {self.ollama_url}")
        print("-" * 50)
        print("Ctrl+C で停止できます\n")
        
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
                
                # Ollamaに送信
                response = self.send_to_ollama(image_base64)
                
                # 結果を表示
                print(f"📸 解析結果:")
                print(f"{response}")
                
                # response にnoneかNoneが含まれず、空文字列でない場合のみログを書き込む
                if "none" not in response and "None" not in response and response.strip() != "":
                    # XMLファイルにログを書き出し
                    self.write_to_xml_log(response)
                
                print("-" * 50)
                
                # 指定された間隔で待機
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n\nアプリケーションを終了します...")
                break
            except Exception as e:
                print(f"予期しないエラー: {e}")
                time.sleep(interval)


def main():
    """
    メイン関数
    """
    try:
        # アプリケーションを初期化
        explainer = OllamaVisionExplainer()
        
        # 継続的な解析を開始（10秒間隔）
        explainer.run_continuous_analysis(interval=1)
        
    except Exception as e:
        print(f"アプリケーション起動エラー: {e}")


if __name__ == "__main__":
    main()
