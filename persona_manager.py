#!/usr/bin/env python3
"""
人格管理システム
personas.yamlファイルから人格定義を読み込み、ランダム選択機能を提供
"""

import os
import yaml
import random
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Persona:
    """人格定義"""
    persona_id: str
    name: str
    handle: str
    description: str
    style: str
    example: str


class PersonaManager:
    """人格管理クラス"""
    
    DEFAULT_PERSONAS_FILE = "personas.yaml"
    
    def __init__(self, personas_file: Optional[str] = None, suppress_warnings: bool = False):
        """
        初期化
        
        Args:
            personas_file: 人格定義ファイルのパス（Noneの場合はデフォルト）
            suppress_warnings: 警告メッセージを抑制するかどうか
        """
        self.personas_file = personas_file or self.DEFAULT_PERSONAS_FILE
        self.suppress_warnings = suppress_warnings
        self.personas: Dict[str, Persona] = {}
        self._load_personas()
    
    def _load_personas(self):
        """人格定義ファイルを読み込み"""
        try:
            if os.path.exists(self.personas_file):
                with open(self.personas_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                personas_data = data.get('personas', {})
                for persona_id, persona_info in personas_data.items():
                    self.personas[persona_id] = Persona(
                        persona_id=persona_id,
                        name=persona_info.get('name', persona_id),
                        handle=persona_info.get('handle', persona_id),
                        description=persona_info.get('description', ''),
                        style=persona_info.get('style', ''),
                        example=persona_info.get('example', '')
                    )
                
                if not self.suppress_warnings:
                    print(f"📋 {len(self.personas)}個の人格を読み込みました: {self.personas_file}")
                
            else:
                if not self.suppress_warnings:
                    print(f"⚠️  人格定義ファイルが見つかりません: {self.personas_file}")
                    print("   デフォルト人格を使用します。")
                    print("   人格をカスタマイズするには:")
                    print(f"   python main.py --create-personas")
                self._load_default_personas()
                
        except Exception as e:
            if not self.suppress_warnings:
                print(f"❌ 人格定義ファイル読み込みエラー: {e}")
                print("   デフォルト人格を使用します。")
            self._load_default_personas()
    
    def _load_default_personas(self):
        """デフォルト人格を読み込み"""
        default_personas = {
            'listener': Persona('listener', 'リスナー', 'リスナーbot', '一般的な視聴者', '短く反応', '右上HPが赤色！'),
            'safety': Persona('safety', '安全監視員', '安全監視bot', '危険を指摘', '警告形式', '落下穴に注意'),
            'expert': Persona('expert', 'ゲーム専門家', 'ゲーム専門bot', '技術的解説', '解説形式', 'ジャンプが有効'),
            'fan1': Persona('fan1', '配信者ファン1', 'ファン1', '冗談好き', '親しみやすく', 'そのジャンプナイス！w'),
            'fan2': Persona('fan2', '配信者ファン2', 'ファン2', '真面目で冷静', '丁寧に', '操作が上達してますね'),
        }
        self.personas = default_personas
    
    def get_all_personas(self) -> Dict[str, Persona]:
        """全ての人格を取得"""
        return self.personas.copy()
    
    def get_persona(self, persona_id: str) -> Optional[Persona]:
        """指定IDの人格を取得"""
        return self.personas.get(persona_id)
    
    def get_random_personas(self, count: int, exclude: Optional[List[str]] = None) -> List[Persona]:
        """
        ランダムに指定数の人格を選択
        
        Args:
            count: 選択する人格数
            exclude: 除外する人格IDのリスト
            
        Returns:
            選択された人格のリスト
        """
        exclude = exclude or []
        available_personas = {k: v for k, v in self.personas.items() if k not in exclude}
        
        if len(available_personas) < count:
            print(f"⚠️  利用可能な人格数({len(available_personas)})が要求数({count})より少ないです")
            count = len(available_personas)
        
        selected_ids = random.sample(list(available_personas.keys()), count)
        return [available_personas[persona_id] for persona_id in selected_ids]
    
    def create_prompt_for_personas(self, selected_personas: List[Persona], voice_context: str = "", prompt_file: str = "prompt.md") -> str:
        """
        選択された人格用のプロンプトを生成
        
        Args:
            selected_personas: 選択された人格のリスト
            voice_context: 音声認識コンテキスト
            prompt_file: プロンプトテンプレートファイルのパス
            
        Returns:
            生成されたプロンプト
        """
        # プロンプトテンプレートを読み込み
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        except FileNotFoundError:
            print(f"⚠️  プロンプトファイルが見つかりません: {prompt_file}")
            # フォールバック用の最小限のプロンプト
            prompt_template = """あなたは複数の異なる人格として、ゲーム配信に対してコメントしてください。

{PERSONA_DESCRIPTIONS}

{VOICE_CONTEXT}

**出力形式：**
```json
{
{JSON_FIELDS}
}
```"""
        except Exception as e:
            print(f"❌ プロンプトファイル読み込みエラー: {e}")
            prompt_template = "エラーが発生しました。"
        
        # 人格リストを生成
        persona_descriptions = []
        json_fields = []
        
        for i, persona in enumerate(selected_personas, 1):
            persona_descriptions.append(
                f"{i}. **{persona.name}**: {persona.description}\n   スタイル: {persona.style}\n   例: {persona.example}"
            )
            json_fields.append(f'  "{persona.persona_id}": "画面の具体的要素1つに言及した短いコメント（20文字以内、ゲーム画面でない場合は「none」）"')
        
        # 音声セクションを生成
        voice_section = f"\n\n**音声認識結果:**\n{voice_context}" if voice_context else ""
        
        # プレースホルダーを置換
        prompt = prompt_template.replace("{PERSONA_DESCRIPTIONS}", chr(10).join(persona_descriptions))
        prompt = prompt.replace("{VOICE_CONTEXT}", voice_section)
        prompt = prompt.replace("{JSON_FIELDS}", chr(10).join(json_fields))
        
        return prompt
    
    def get_persona_mapping(self, selected_personas: List[Persona]) -> Dict[str, Dict[str, str]]:
        """
        選択された人格のマッピング情報を取得（main.pyとの互換性用）
        
        Args:
            selected_personas: 選択された人格のリスト
            
        Returns:
            人格マッピング辞書
        """
        mapping = {}
        for persona in selected_personas:
            mapping[persona.persona_id] = {
                "handle": persona.handle,
                "persona": persona.name
            }
        return mapping
    
    def list_personas(self):
        """利用可能な人格一覧を表示"""
        print(f"\n=== 利用可能な人格一覧 ({len(self.personas)}個) ===")
        for persona_id, persona in self.personas.items():
            print(f"  {persona_id}: {persona.name} ({persona.handle})")
            print(f"    - {persona.description}")
            print(f"    - 例: \"{persona.example}\"")
        print("=" * 50)
    
    def create_personas_file(self, output_path: str = None) -> str:
        """
        サンプル人格ファイルをコピー
        
        Args:
            output_path: 出力先ファイルパス（Noneの場合はpersonas.yaml）
            
        Returns:
            作成されたファイルのパス（失敗した場合はNone）
        """
        import shutil
        
        source_path = "personas.sample.yaml"
        if output_path is None:
            output_path = "personas.yaml"
        
        # サンプルファイルが存在するかチェック
        if not os.path.exists(source_path):
            print(f"エラー: サンプル人格ファイルが見つかりません: {source_path}")
            print("リポジトリが正しくクローンされているか確認してください。")
            return None
        
        # 出力先が既に存在する場合は確認
        if os.path.exists(output_path):
            print(f"警告: {output_path} は既に存在します。")
            response = input("上書きしますか？ (y/N): ").lower()
            if response != 'y' and response != 'yes':
                print("操作をキャンセルしました。")
                return None
        
        # ファイルをコピー
        try:
            shutil.copy2(source_path, output_path)
            print(f"人格ファイルを作成しました: {output_path}")
            print(f"人格をカスタマイズするには {output_path} を編集してください。")
            return output_path
        except Exception as e:
            print(f"エラー: ファイルのコピーに失敗しました: {e}")
            return None