#!/usr/bin/env python3
"""
設定管理モジュール
YAML設定ファイルを読み込み、コマンドライン引数でオーバーライド可能な設定を提供
"""

import os
import yaml
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """環境固有設定"""
    ollama_url: str = "http://localhost:11434"
    voice_server_url: Optional[str] = None
    xml_file: str = "comment.xml"


@dataclass
class BehaviorConfig:
    """動作設定"""
    enable_voice: bool = True
    debug_mode: bool = False
    analysis_interval: float = 0.1


@dataclass  
class ModelsConfig:
    """AIモデル設定"""
    image_analysis_model: str = "gemma3:12b"
    comment_generation_model: str = "gemma3:12b"


@dataclass
class ImageConfig:
    """画像処理設定"""
    compression_ratio: float = 2.0
    jpeg_quality: int = 75


@dataclass
class PerformanceConfig:
    """パフォーマンス設定"""
    image: ImageConfig


@dataclass
class PersonasConfig:
    """人格設定"""
    personas_file: str = "personas.yaml"
    select_count: int = 5
    always_include: list = None

    def __post_init__(self):
        if self.always_include is None:
            self.always_include = []


@dataclass
class SystemConfig:
    """システム設定"""
    prompt_file: str = "prompt.md"


@dataclass
class AppConfig:
    """アプリケーション全体の設定"""
    environment: EnvironmentConfig
    behavior: BehaviorConfig
    models: ModelsConfig
    performance: PerformanceConfig
    personas: PersonasConfig
    system: SystemConfig


class ConfigManager:
    """設定管理クラス"""
    
    DEFAULT_CONFIG_FILE = "config.yaml"
    
    def __init__(self, config_file: Optional[str] = None, suppress_warnings: bool = False):
        """
        初期化
        
        Args:
            config_file: 設定ファイルのパス（Noneの場合はデフォルト）
            suppress_warnings: 警告メッセージを抑制するかどうか
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.suppress_warnings = suppress_warnings
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """設定ファイルを読み込み"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if not self.suppress_warnings:
                    print(f"📄 設定ファイルを読み込みました: {self.config_file}")
                return self._parse_config(data)
            else:
                if not self.suppress_warnings:
                    print(f"⚠️  設定ファイルが見つかりません: {self.config_file}")
                    print("   デフォルト設定で実行します。")
                    print("   設定をカスタマイズするには:")
                    print(f"   python main.py --create-config")
                    print("")
                return self._default_config()
                
        except Exception as e:
            if not self.suppress_warnings:
                print(f"❌ 設定ファイル読み込みエラー: {e}")
                print("   デフォルト設定で実行します。")
            return self._default_config()
    
    def _parse_config(self, data: Dict[str, Any]) -> AppConfig:
        """YAML データから設定オブジェクトを構築"""
        env_data = data.get('environment', {})
        behavior_data = data.get('behavior', {})
        models_data = data.get('models', {})
        perf_data = data.get('performance', {})
        personas_data = data.get('personas', {})
        system_data = data.get('system', {})
        
        # 画像設定を展開
        image_data = perf_data.get('image', {})
        
        return AppConfig(
            environment=EnvironmentConfig(
                ollama_url=env_data.get('ollama_url', "http://localhost:11434"),
                voice_server_url=env_data.get('voice_server_url'),
                xml_file=env_data.get('xml_file', "comment.xml")
            ),
            behavior=BehaviorConfig(
                enable_voice=behavior_data.get('enable_voice', True),
                debug_mode=behavior_data.get('debug_mode', False),
                analysis_interval=behavior_data.get('analysis_interval', 0.1)
            ),
            models=ModelsConfig(
                image_analysis_model=models_data.get('image_analysis_model', "gemma3:12b"),
                comment_generation_model=models_data.get('comment_generation_model', "gemma3:12b")
            ),
            performance=PerformanceConfig(
                image=ImageConfig(
                    compression_ratio=image_data.get('compression_ratio', 2.0),
                    jpeg_quality=image_data.get('jpeg_quality', 75)
                )
            ),
            personas=PersonasConfig(
                personas_file=personas_data.get('personas_file', "personas.yaml"),
                select_count=personas_data.get('select_count', 5),
                always_include=personas_data.get('always_include', [])
            ),
            system=SystemConfig(
                prompt_file=system_data.get('prompt_file', "prompt.md")
            )
        )
    
    def _default_config(self) -> AppConfig:
        """デフォルト設定を作成"""
        return AppConfig(
            environment=EnvironmentConfig(),
            behavior=BehaviorConfig(),
            models=ModelsConfig(),
            performance=PerformanceConfig(
                image=ImageConfig()
            ),
            personas=PersonasConfig(),
            system=SystemConfig()
        )
    
    def create_sample_config(self, output_path: str = None) -> str:
        """サンプル設定ファイルをコピー"""
        import shutil
        
        source_path = "config.sample.yaml"
        if output_path is None:
            output_path = "config.yaml"
        
        # サンプルファイルが存在するかチェック
        if not os.path.exists(source_path):
            print(f"エラー: サンプル設定ファイルが見つかりません: {source_path}")
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
            print(f"設定ファイルを作成しました: {output_path}")
            print(f"設定をカスタマイズするには {output_path} を編集してください。")
            return output_path
        except Exception as e:
            print(f"エラー: ファイルのコピーに失敗しました: {e}")
            return None
    
    def override_with_args(self, **kwargs) -> AppConfig:
        """コマンドライン引数で設定をオーバーライド"""
        config = self.config
        
        # 環境設定のオーバーライド
        if 'ollama_url' in kwargs and kwargs['ollama_url'] is not None:
            config.environment.ollama_url = kwargs['ollama_url']
        
        if 'voice_server_url' in kwargs and kwargs['voice_server_url'] is not None:
            config.environment.voice_server_url = kwargs['voice_server_url']
        
        if 'xml_file' in kwargs and kwargs['xml_file'] is not None:
            config.environment.xml_file = kwargs['xml_file']
        
        # 動作設定のオーバーライド
        if 'enable_voice' in kwargs:
            config.behavior.enable_voice = kwargs['enable_voice']
        
        if 'debug_mode' in kwargs:
            config.behavior.debug_mode = kwargs['debug_mode']
        
        if 'analysis_interval' in kwargs and kwargs['analysis_interval'] is not None:
            config.behavior.analysis_interval = kwargs['analysis_interval']
        
        return config
    
    def get_config(self) -> AppConfig:
        """現在の設定を取得"""
        return self.config
    
    def print_config(self):
        """現在の設定を表示"""
        print("=== 現在の設定 ===")
        print(f"Ollama URL: {self.config.environment.ollama_url}")
        print(f"音声認識サーバー: {self.config.environment.voice_server_url or 'ローカル'}")
        print(f"XML出力先: {self.config.environment.xml_file}")
        print(f"音声認識: {'有効' if self.config.behavior.enable_voice else '無効'}")
        print(f"デバッグモード: {'有効' if self.config.behavior.debug_mode else '無効'}")
        print(f"解析間隔: {self.config.behavior.analysis_interval}秒")
        print(f"画像解析モデル: {self.config.models.image_analysis_model}")
        print(f"コメント生成モデル: {self.config.models.comment_generation_model}")
        print(f"画像圧縮倍率: {self.config.performance.image.compression_ratio}x")
        print(f"JPEG品質: {self.config.performance.image.jpeg_quality}")
        print(f"人格ファイル: {self.config.personas.personas_file}")
        print(f"選択人格数: {self.config.personas.select_count}")
        if self.config.personas.always_include:
            print(f"固定人格: {', '.join(self.config.personas.always_include)}")
        print("================")