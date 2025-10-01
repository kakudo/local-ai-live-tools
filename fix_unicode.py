#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

# ユニコード絵文字をテキストに置き換えるマッピング
emoji_replacements = {
    '🔄': '[Queue]',
    '📝': '[XML]',
    '🧵': '[Thread]', 
    '🔴': '[Stop]',
    '🟢': '[Start]',
    '✅': '[OK]',
    '❌': '[Error]',
    '🔇': '[Mute]',
    '📋': '[File]',
    '⚠️': '[Warning]',
    '🖼️': '[Image]',
    '📸': '[Screenshot]',
    '🔍': '[Debug]'
}

def fix_unicode_in_file(filepath):
    """ファイル内のユニコード絵文字を置き換える"""
    try:
        # ファイルを読み込み
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 各絵文字を置き換え
        for emoji, replacement in emoji_replacements.items():
            content = content.replace(emoji, replacement)
        
        # 変更があった場合のみファイルを更新
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"ファイル '{filepath}' の絵文字を置き換えました")
            return True
        else:
            print(f"ファイル '{filepath}' に変更はありませんでした")
            return False
            
    except Exception as e:
        print(f"ファイル処理エラー '{filepath}': {e}")
        return False

if __name__ == "__main__":
    # main.pyファイルの絵文字を修正
    fixed = fix_unicode_in_file('main.py')
    if fixed:
        print("絵文字の置き換えが完了しました")
    else:
        print("置き換える絵文字が見つかりませんでした")