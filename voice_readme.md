# voice.py 使用方法

リアルタイム音声認識システムです。Whisperを使用してマイクからの音声をリアルタイムで文字起こしします。

## 機能

- **リアルタイム音声認識**: マイクからの音声を3秒チャンクでリアルタイム処理
- **Whisperモデル選択**: tiny, base, small, medium, large, large-v2, large-v3から選択可能
- **デバイス選択**: 利用可能な音声デバイスから選択可能
- **自動GPU検出**: CUDAが利用可能な場合は自動でGPUを使用
- **日本語対応**: 日本語音声認識に最適化

## 使用方法

### 基本的な使い方
```bash
python voice.py
```

### オプション

#### モデルを指定して実行
```bash
# tinyモデル (最速、精度低)
python voice.py --model tiny

# smallモデル (バランス)
python voice.py --model small

# largeモデル (最高精度、重い)
python voice.py --model large
```

#### 利用可能な音声デバイスを確認
```bash
python voice.py --list-devices
```

#### 特定の音声デバイスを使用
```bash
# デバイスインデックス1を使用
python voice.py --device-index 1
```

#### CPUを強制使用
```bash
python voice.py --device cpu
```

### 全オプション
```bash
python voice.py --help
```

## 推奨設定

### 性能重視
```bash
python voice.py --model tiny --device cuda
```

### バランス（推奨）
```bash
python voice.py --model base
```

### 精度重視
```bash
python voice.py --model large --device cuda
```

## 注意事項

1. **初回実行時**: 選択したWhisperモデルのダウンロードが必要です（100MB～数GB）
2. **マイク権限**: システムのマイクアクセス許可が必要です
3. **終了方法**: Ctrl+C で安全に終了できます
4. **無音検出**: 音量が小さすぎる場合は認識されません

## トラブルシューティング

### PyAudioエラーが出る場合
Windowsの場合、PyAudioのインストールに問題がある可能性があります：
```bash
uv add pyaudio
```

### マイクが認識されない場合
1. `--list-devices` で利用可能なデバイスを確認
2. `--device-index` で適切なデバイスを指定

### メモリエラーが出る場合
1. より小さなモデル（tiny/base）を使用
2. `--device cpu` でCPU使用を強制

## モデル比較

| モデル | サイズ | 速度 | 精度 | 推奨用途 |
|--------|--------|------|------|----------|
| tiny   | ~39MB  | 最速 | 低   | リアルタイム重視 |
| base   | ~74MB  | 高速 | 中   | バランス（推奨） |
| small  | ~244MB | 中   | 高   | 精度とバランス |
| medium | ~769MB | 低   | 高   | 精度重視 |
| large  | ~1550MB| 最低 | 最高 | 最高精度 |