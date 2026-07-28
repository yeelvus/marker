#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 surya-ocr + llama.cpp 在 Mac 上 balanced 模式失败的问题。

上游 bug: layout/table JSON schema 的 bbox pattern 使用 \\d，
llama-server 转 GBNF grammar 时无法解析，导致:
  Failed to initialize samplers: failed to parse grammar
  Layout inference failed for page N; leaving page empty.

官方 PR: https://github.com/datalab-to/surya/pull/539
Issue:   https://github.com/datalab-to/surya/issues/542

用法:
  conda activate marker
  python scripts/patch_surya_llamacpp_grammar.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import surya
    except ImportError:
        print("[ERROR] 未安装 surya-ocr，请先 conda activate marker", file=sys.stderr)
        return 1

    surya_root = Path(surya.__file__).resolve().parent
    prompts = surya_root / "inference" / "prompts.py"
    if not prompts.is_file():
        print(f"[ERROR] 找不到 {prompts}", file=sys.stderr)
        return 1

    text = prompts.read_text(encoding="utf-8")
    old = r'r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$"'
    new = r'r"^[0-9]{1,4} [0-9]{1,4} [0-9]{1,4} [0-9]{1,4}$"'

    if new in text and old not in text:
        print(f"[OK] 已打过补丁: {prompts}")
        return 0
    if old not in text:
        print(
            f"[WARN] 未找到待替换 pattern，可能上游已修复或版本变化。\n文件: {prompts}",
            file=sys.stderr,
        )
        # 打印相关行便于排查
        for i, line in enumerate(text.splitlines(), 1):
            if "pattern" in line and "1,4" in line:
                print(f"  L{i}: {line}")
        return 2

    bak = prompts.with_suffix(".py.bak")
    if not bak.exists():
        bak.write_text(text, encoding="utf-8")
        print(f"[INFO] 已备份: {bak}")

    prompts.write_text(text.replace(old, new), encoding="utf-8")
    n = text.count(old)
    print(f"[OK] 已修补 {n} 处 \\d → [0-9]: {prompts}")
    print("请重新用 --mode balanced 转换验证。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
