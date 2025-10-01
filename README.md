# Local AI Live Tools

ゲーム配信向けAIコメントシステム。画面解析と音声認識を組み合わせて、11人の異なる人格によるリアルタイムコメントを生成します。
出力されるcomment.xmlは、NiCommentGeneratorなどでそのまま利用できます。

## 前提条件

1. **Ollama** がローカルで起動していること
2. **Vision対応モデル** がOllamaにインストールされていること
3. **MultiCommentViewer** (コメント表示用、オプション)

### Ollamaのセットアップ

```bash
# Ollamaをインストール（まだの場合）
# https://ollama.ai からダウンロードしてインストール

# 必要なモデルをプル
ollama pull gemma3:12b        # メインモデル（画像解析・コメント生成）
ollama pull deepseek-r1:8b    # 軽量コメント生成用（オプション）

# Ollamaサーバーを起動
ollama serve
```

## インストール

### 基本的な依存関係

```bash
# 基本依存関係をインストール
uv sync
```

### PyTorch (CUDA版) のインストール

音声認識機能でGPUを使用する場合は、CUDA版のPyTorchを手動でインストールする必要があります：

```bash
# CUDA版PyTorchをインストール（GPU使用の場合）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU版PyTorchをインストール（GPU不要の場合）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**注意**: システムのCUDAバージョンに合わせて適切なインデックスURLを選択してください。CUDA 12.1の場合は `cu121`、CUDA 11.8の場合は `cu118` を使用します。

## 使用方法

### メインシステム（AI コメント生成）

```bash
# 通常実行
python main.py

# デバッグモードで実行（詳細なログ出力）
python main.py --debug
python main.py -d
```

### 音声認識システム

```bash
# 音声認識システムを実行
python voice.py

# 利用可能な音声デバイスを確認
python voice.py --list-devices

# 特定のデバイスを指定して実行
python voice.py --device-index 1

# 特定のWhisperモデルを使用
python voice.py --model large

# CPUを強制使用
python voice.py --device cpu
```

## 機能

### 🎮 AI コメントシステム（main.py）

#### 2段階処理アーキテクチャ
- **段階1：画像解析** - 高精度な画面内容の詳細分析
- **段階2：コメント生成** - 11人の人格による多様なコメント生成

#### 11人のコメンター人格
1. **リスナー** - 画面の出来事に興奮・感想を表現
2. **安全監視員** - 画面内の危険要素を指摘
3. **ゲーム専門家** - 技術的・戦略的解説
4. **配信者ファン1** - 冗談好きで面白いコメント
5. **配信者ファン2** - 真面目で冷静な分析
6. **配信者アンチ** - ネガティブで皮肉な反応
7. **自分語り** - 50代男性店長の体験談
8. **エロ爺** - やや性的なコメント
9. **小学生** - 子供っぽい発言、www多用
10. **質問の人** - 疑問点を投げかける
11. **顔文字の人** - 顔文字で反応

#### 主な機能
- **アクティブウィンドウの自動検出**: 現在フォーカスされているウィンドウを自動特定
- **定期的スクリーンショット**: 1秒間隔でリアルタイム解析（設定可能）
- **ゲーム画面判定**: 非ゲーム画面（プログラミング環境等）を自動フィルタリング
- **音声認識連携**: 配信者の発言を考慮したコメント生成
- **XMLコメント出力**: MultiCommentViewerでの表示に対応
- **キューベースシステム**: 自然なタイミングでコメントを順次出力
- **デュアルモデル対応**: 画像解析用とコメント生成用で異なるモデル使用可能

### 音声認識システム（voice.py）

- **リアルタイム音声認識**: Whisperを使用してマイクからの音声をリアルタイムで文字起こし
- **GPU加速**: CUDA対応GPUを自動検出してWhisperモデルをGPUで高速実行
- **複数音声デバイス対応**: 接続されている音声入力デバイスを自動検出・選択可能
- **日本語対応**: 日本語音声の高精度認識
- **複数Whisperモデル対応**: tiny, base, small, medium, large, large-v2, large-v3から選択可能
- **音声認識結果の蓄積**: 認識結果の取得・クリア機能

## 設定

`main.py`内で以下の設定を変更できます：

```python
# Ollamaの設定
OLLAMA_URL = "http://localhost:11434"  # OllamaサーバーのURL
IMAGE_MODEL = "gemma3:12b"  # 画像解析用モデル（段階1）
COMMENT_MODEL = "gemma3:12b"  # コメント生成用モデル（段階2）

# 実行間隔の設定
INTERVAL = 1  # 1秒間隔でスクリーンショット取得

# コメント出力設定
OUTPUT_COMMENTS_TO_FILE = True  # XMLファイル出力の有効/無効
COMMENT_OUTPUT_FILE = "comment.xml"  # 出力ファイル名
```

## 使用例

### 基本的な使用手順

1. **Ollamaサーバーを起動**（必須）
2. **メインシステムを実行**: `python main.py`
3. **任意のゲームをアクティブにする**（ブラウザ、Steam、エミュレータなど）
4. **1秒ごとに11人のAIコメンターがリアルタイム反応**
5. **オプション**: `python voice.py` で音声認識も同時実行

### 実際の出力例

```
[ゲーム画面検出時]
リスナー: わぁ！すごいアクションだね！
安全監視員: その動きは危険です
ゲーム専門家: このスキルコンボが効率的ですね
小学生: えーwwwやばいwww

[非ゲーム画面検出時]
全コメンター: none（出力なし）
```

## 停止方法

`Ctrl+C` でアプリケーションを停止できます。

## 注意事項

- **ゲーム画面検出**: プログラミング環境などの非ゲーム画面では自動的にコメント出力を停止
- **Ollamaサーバー**: 必ず起動している必要があります（接続エラー時はエラーメッセージ表示）
- **処理時間**: 2段階処理により高品質だが、やや処理時間が必要
- **MultiCommentViewer連携**: `comment.xml`ファイルを監視することでコメントビューアーに表示可能
- **音声認識**: オプション機能として別プロセスで実行可能
