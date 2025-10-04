# Local AI Live Tools

ゲーム配信向けAIコメントシステム。画面解析と音声認識を組み合わせて、カスタマイズ可能な複数人格によるリアルタイムコメントを生成します。
出力されるcomment.xmlは、NiCommentGeneratorなどでそのまま利用できます。

## 🌟 新機能

- **🎭 柔軟な人格システム**: 20+の人格から任意の数をランダム選択
- **⚙️ 設定ファイル管理**: YAML形式での簡単な設定管理
- **🌐 分散処理対応**: 音声認識とOllamaサーバーを別マシンで実行可能
- **📏 スマート画像圧縮**: アスペクト比維持の圧縮率指定
- **🎨 カスタマイズ可能**: プロンプト、人格、設定すべてを自由に調整

## 前提条件

1. **Ollama** がローカルで起動していること
2. **Vision対応モデル** がOllamaにインストールされていること
3. **NiCommentGeneratorなど** (コメント表示用) 読み込むcomment.xmlのパスを本ツールのXML出力先にする

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

## 🚀 クイックスタート

### 1. 初回セットアップ

```bash
# 設定ファイルを作成
python main.py --create-config

# 人格定義ファイルを作成
python main.py --create-personas

# 設定をカスタマイズ
# config.yaml - 基本設定（サーバーURL、画像圧縮率など）
# personas.yaml - 人格定義（コメントする人格をカスタマイズ）
```

### 2. 基本実行

```bash
# 通常実行（設定ファイル使用）
python main.py

# デバッグモード（詳細ログ出力）
python main.py --debug

# 一時的なオーバーライド
python main.py --ollama-url http://192.168.1.50:11434 --debug
```

### 3. 分散実行（複数マシン構成）

```bash
# マシン1: 音声認識サーバー起動（デフォルトでサーバーモード）
python voice.py --host 0.0.0.0 --port 5000

# マシン2: メインシステム実行
python main.py --voice-server http://192.168.1.100:5000 --ollama-url http://192.168.1.50:11434
```

## 📋 コマンドライン オプション

### メインシステム（main.py）

```bash
# ヘルプ表示
python main.py --help

# 設定関連
python main.py --create-config          # 設定ファイル作成
python main.py --create-personas        # 人格ファイル作成
python main.py --config custom.yaml     # カスタム設定ファイル指定

# サーバー指定
python main.py --ollama-url http://192.168.1.50:11434     # Ollamaサーバー
python main.py --voice-server http://192.168.1.100:5000   # 音声認識サーバー
python main.py --xml-file ./output/comment.xml            # XML出力先

# 動作制御
python main.py --debug                   # デバッグモード
python main.py --no-voice              # 音声認識無効
python main.py --interval 0.5          # 解析間隔（秒）
```

### 音声認識システム（voice.py）

```bash
# サーバーモード（デフォルト - リモート利用）
python voice.py                         # WebAPIサーバー起動
python voice.py --port 5000             # ポート指定
python voice.py --host 127.0.0.1        # ローカルホストのみ
python voice.py --model large           # Whisperモデル指定

# ローカルモード（直接実行）
python voice.py --local                 # ローカル直接実行
python voice.py --local --list-devices  # デバイス一覧表示
python voice.py --local --device-index 1 # デバイス指定
```

## 🎮 主な機能

### AI コメントシステム（main.py）

#### 2段階処理アーキテクチャ
- **段階1：画像解析** - 高精度な画面内容の詳細分析
- **段階2：コメント生成** - 選択された人格による多様なコメント生成

#### 柔軟な人格システム
- **20+の多彩な人格**: リスナー、安全監視員、ゲーム専門家、配信者ファン、アンチ、戦術家、RTA勢、初心者、古参など
- **ランダム選択**: 毎回異なる組み合わせで自然なバラエティ
- **カスタマイズ可能**: `personas.yaml`で自由に人格を追加・編集・削除
- **選択数調整**: 設定ファイルで毎回選ぶ人格数を指定（例：20人中5人を選択）
- **固定人格**: 特定の人格を必ず含める設定も可能

#### サンプル人格（20種類以上）
- **リスナー** - 画面の出来事に興奮・感想
- **安全監視員** - 危険要素を指摘
- **ゲーム専門家** - 技術的・戦略的解説  
- **配信者ファン1/2** - 配信者への反応
- **配信者アンチ** - 皮肉な反応
- **戦術家** - 戦闘戦略の提案
- **コレクター** - アイテム収集に注目
- **RTA勢** - 効率性重視
- **初心者** - 基本的な質問
- **古参** - 懐古的なコメント
- **ネタ民** - ミーム系コメント
- **真面目さん** - 建設的コメント
- **雰囲気重視** - BGMやグラフィック
- その他多数...

#### 高度な機能
- **アクティブウィンドウ自動検出**: 現在のゲーム画面を自動特定
- **スマート画像圧縮**: 圧縮率指定でアスペクト比維持（2.0倍なら縦横1/2、面積1/4）
- **ゲーム画面判定**: 非ゲーム画面を自動フィルタリング
- **音声認識連携**: 配信者の発言を考慮したコメント生成
- **分散処理対応**: 音声認識とOllamaを別マシンで実行可能
- **設定ファイル管理**: YAML形式での柔軟な設定
- **XMLコメント出力**: MultiCommentViewer対応
- **キューベースシステム**: 自然なタイミングでコメント出力

### 音声認識システム（voice.py）

#### ローカル音声認識
- **リアルタイム音声認識**: Whisperによる高精度な日本語認識
- **GPU加速**: CUDA自動検出でWhisperを高速実行
- **複数デバイス対応**: 音声入力デバイス自動検出・選択
- **複数Whisperモデル**: tiny～large-v3まで選択可能

#### リモート音声認識（WebAPI）
- **HTTP REST API**: 別マシンからのアクセス対応
- **音声認識サーバー**: `--server`モードでWebAPIとして起動
- **ネットワーク分散**: 音声処理を専用マシンで実行可能
- **接続監視**: 自動接続チェックとエラーハンドリング

## ⚙️ 設定ファイル

### config.yaml（基本設定）

```yaml
# 環境固有設定（必須カスタマイズ）
environment:
  ollama_url: "http://localhost:11434"      # OllamaサーバーURL
  voice_server_url: null                    # 音声認識サーバーURL（nullでローカル）
  xml_file: "comment.xml"                   # XML出力先

# 動作設定
behavior:
  enable_voice: true                        # 音声認識有効/無効
  debug_mode: false                         # デバッグモード
  analysis_interval: 0.1                   # 解析間隔（秒）

# AIモデル設定
models:
  image_analysis_model: "gemma3:12b"        # 画像解析用モデル
  comment_generation_model: "gemma3:12b"    # コメント生成用モデル

# パフォーマンス設定
performance:
  image:
    compression_ratio: 2.0                  # 画像圧縮倍率（2.0＝縦横1/2）
    jpeg_quality: 75                        # JPEG品質

# 人格・コメント設定
personas:
  personas_file: "personas.yaml"           # 人格定義ファイル
  select_count: 5                          # 毎回選択する人格数
  always_include: []                       # 固定で含める人格ID
```

### personas.yaml（人格定義）

```yaml
personas:
  # 既存の人格をカスタマイズ
  listener:
    name: "リスナー"
    handle: "リスナーbot"
    description: "一般的な視聴者。画面の出来事に興奮や感想を表現"
    style: "「[画面要素]が[状態]！」形式で短く反応"
    example: "右上HPが赤色！"
  
  # 新しい人格を追加
  custom_persona:
    name: "カスタム人格"
    handle: "カスタム"
    description: "独自の反応をする人格"
    style: "「カスタム: [内容]」形式"
    example: "カスタム: すごい！"
```

## 💡 使用例

### シングルマシン構成（基本）

```bash
# 1. 初回セットアップ
python main.py --create-config
python main.py --create-personas

# 2. config.yaml を編集（Ollama URL など）
# 3. personas.yaml を編集（人格をカスタマイズ）

# 4. 実行
python main.py --debug
```

### 複数マシン構成（分散処理）

```bash
# マシンA: 音声認識専用サーバー（デフォルトでサーバーモード）
python voice.py --host 0.0.0.0 --port 5000

# マシンB: Ollama専用サーバー
ollama serve --host 0.0.0.0

# マシンC: メイン処理（ゲーム画面キャプチャ）
python main.py \
  --voice-server http://192.168.1.10:5000 \
  --ollama-url http://192.168.1.11:11434
```

### カスタム設定例

```bash
# 高速処理設定（画質を下げて高速化）
# config.yaml で compression_ratio: 4.0, jpeg_quality: 60

# 大量人格からランダム選択
# personas.yaml で50個の人格を定義
# config.yaml で select_count: 8（50人中8人をランダム選択）

# 特定人格固定 + ランダム
# config.yaml で always_include: ["listener", "safety"]（必ず含める）
# + select_count: 5（残り3人をランダム選択）
```

### 実際の出力例

```xml
<!-- comment.xml -->
<comment handle="リスナーbot">右上HPが赤色！</comment>
<comment handle="戦術bot">左から回り込みがおすすめ</comment>
<comment handle="RTA">ショートカットでタイム短縮</comment>
<comment handle="古参">昔はもっと難しかった</comment>
<comment handle="顔文字">(^o^)</comment>
```

## 🛠️ トラブルシューティング

### よくある問題

**Q: 設定ファイルが見つからない**
```bash
A: python main.py --create-config
```

**Q: 人格ファイルが見つからない**
```bash
A: python main.py --create-personas
```

**Q: Ollamaに接続できない**
```bash
A: ollama serve  # Ollamaサーバーを起動
```

**Q: 音声認識サーバーに接続できない**
```bash
A: python voice.py --server  # 音声認識サーバーを起動
```

**Q: 処理が重い**
```bash
# config.yaml で画像圧縮を調整
performance:
  image:
    compression_ratio: 3.0  # より強い圧縮
    jpeg_quality: 60        # 品質を下げる
```

## 📝 注意事項

- **ゲーム画面判定**: 非ゲーム画面では自動的にコメント出力を停止
- **分散処理**: ネットワーク遅延を考慮して適切な間隔を設定
- **GPU使用**: CUDA対応GPUがあれば音声認識が大幅に高速化
- **MultiCommentViewer連携**: XMLファイル監視でリアルタイム表示

## ⏹️ 停止方法

`Ctrl+C` でアプリケーションを停止できます。
