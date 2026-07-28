#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marker 稳定批处理入口：支持单个文件或整个文件夹。

用法:
  python process_docs.py -p /path/to/file.pdf -o /path/to/output
  python process_docs.py -p /path/to/folder -o /path/to/output --mode fast
  python process_docs.py -p /path/to/file.pdf -o /path/to/output --disable_ocr
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


# PDF / 图片；安装 [full] 后还可处理 office / epub / html
SUPPORTED_EXTS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".epub",
}


def find_marker_bins() -> tuple[str, str]:
    """定位 marker_single / marker 可执行文件。"""
    single = shutil.which("marker_single")
    multi = shutil.which("marker")
    fallback_root = Path("/opt/anaconda3/envs/marker/bin")
    if not single:
        cand = fallback_root / "marker_single"
        if cand.is_file():
            single = str(cand)
    if not multi:
        cand = fallback_root / "marker"
        if cand.is_file():
            multi = str(cand)
    if not single or not multi:
        raise RuntimeError(
            "找不到 marker_single / marker。请先: conda activate marker && "
            'pip install -e ".[full]"'
        )
    return single, multi


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"不支持的文件类型: {path.name} ({path.suffix})")
        return [path]
    files = sorted(
        p.resolve()
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not files:
        raise ValueError(f"目录中没有可处理文件: {path}")
    return files


APP_DIR = Path(__file__).resolve().parent

LLM_SERVICES = {
    "gemini": "marker.services.gemini.GoogleGeminiService",
    "openai": "marker.services.openai.OpenAIService",
    "ollama": "marker.services.ollama.OllamaService",
    "openrouter": "marker.services.openrouter.OpenRouterService",
}


def load_dotenv_files() -> None:
    """加载 Projects/marker/local.env 与 .env（不覆盖已有环境变量）。"""
    for name in ("local.env", ".env"):
        path = APP_DIR / name
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            pass


def resolve_llm_service(name: str) -> str:
    name = (name or "gemini").strip()
    if name in LLM_SERVICES:
        return LLM_SERVICES[name]
    if "." in name:
        return name  # 已是完整 import path
    raise ValueError(f"未知 LLM 服务: {name}，可选: {', '.join(LLM_SERVICES)}")


def check_llm_ready(args: argparse.Namespace) -> str | None:
    """
    启用 use_llm 时检查必要密钥。
    返回 None 表示 OK，否则返回错误说明。
    """
    if not args.use_llm:
        return None
    service = resolve_llm_service(args.llm_service)
    gemini_key = (
        args.gemini_api_key
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    ).strip()
    openai_key = (
        args.openai_api_key or os.environ.get("OPENAI_API_KEY") or ""
    ).strip()
    openrouter_key = (
        args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY") or ""
    ).strip()

    if service.endswith("GoogleGeminiService"):
        if not gemini_key:
            return (
                "已启用 --use_llm，默认使用 Google Gemini，但未配置 API Key。\n"
                "请任选其一：\n"
                "  1) 不勾选「使用 LLM 增强」（推荐，本地即可转换）\n"
                "  2) 设置环境变量: export GOOGLE_API_KEY=你的密钥\n"
                "  3) 在 Projects/marker/local.env 写入: GOOGLE_API_KEY=你的密钥\n"
                "  4) 命令行加: --gemini_api_key 你的密钥\n"
                "申请地址: https://aistudio.google.com/apikey"
            )
    elif service.endswith("OpenAIService"):
        if not openai_key:
            return (
                "已启用 OpenAI LLM，但未配置 OPENAI_API_KEY。\n"
                "请设置 export OPENAI_API_KEY=... 或在 local.env 中配置。"
            )
    elif service.endswith("OpenRouterService"):
        if not openrouter_key:
            return (
                "已启用 OpenRouter，但未配置 OPENROUTER_API_KEY。\n"
                "请设置 export OPENROUTER_API_KEY=... 或在 local.env 中配置。"
            )
    elif service.endswith("OllamaService"):
        # Ollama 本地无需 key；仅提示
        return None
    return None


def build_common_args(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = [
        "--output_dir",
        str(args.output),
        "--output_format",
        args.output_format,
        "--mode",
        args.mode,
    ]
    if args.disable_ocr:
        cmd.append("--disable_ocr")
    if args.force_ocr:
        cmd.append("--force_ocr")
    if args.use_llm:
        cmd.append("--use_llm")
        cmd.extend(["--llm_service", resolve_llm_service(args.llm_service)])
        gemini_key = (
            args.gemini_api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        ).strip()
        if gemini_key:
            cmd.extend(["--gemini_api_key", gemini_key])
        openai_key = (
            args.openai_api_key or os.environ.get("OPENAI_API_KEY") or ""
        ).strip()
        if openai_key:
            cmd.extend(["--openai_api_key", openai_key])
        if args.openai_base_url:
            cmd.extend(["--openai_base_url", args.openai_base_url])
        if args.openai_model:
            cmd.extend(["--openai_model", args.openai_model])
        if args.ollama_base_url:
            cmd.extend(["--ollama_base_url", args.ollama_base_url])
        if args.ollama_model:
            cmd.extend(["--ollama_model", args.ollama_model])
        openrouter_key = (
            args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY") or ""
        ).strip()
        if openrouter_key:
            cmd.extend(["--openrouter_api_key", openrouter_key])
    if args.page_range:
        cmd.extend(["--page_range", args.page_range])
    if args.paginate_output:
        cmd.append("--paginate_output")
    if args.disable_image_extraction:
        cmd.append("--disable_image_extraction")
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    return cmd


def main() -> int:
    load_dotenv_files()

    parser = argparse.ArgumentParser(description="Marker document conversion (single file or folder)")
    parser.add_argument("-p", "--path", required=True, help="PDF/image/office file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        default="fast",
        choices=["fast", "balanced"],
        help="Conversion mode (Mac 推荐 fast；GPU 可用 balanced)",
    )
    parser.add_argument(
        "--output_format",
        default="markdown",
        choices=["markdown", "json", "html", "chunks"],
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--disable_ocr",
        action="store_true",
        help="纯文本层提取，不启动 VLM（最快，扫描件/公式会差）",
    )
    parser.add_argument("--force_ocr", action="store_true", help="强制全页 OCR")
    parser.add_argument(
        "--use_llm",
        action="store_true",
        help="使用 LLM 提升质量（默认 Gemini，需 GOOGLE_API_KEY）",
    )
    parser.add_argument(
        "--llm_service",
        default="gemini",
        help="LLM 后端: gemini | openai | ollama | openrouter（或完整 import path）",
    )
    parser.add_argument("--gemini_api_key", default="", help="Google Gemini API Key")
    parser.add_argument("--openai_api_key", default="", help="OpenAI API Key")
    parser.add_argument("--openai_base_url", default="", help="OpenAI 兼容 base URL")
    parser.add_argument("--openai_model", default="", help="OpenAI 模型名")
    parser.add_argument("--openrouter_api_key", default="", help="OpenRouter API Key")
    parser.add_argument("--ollama_base_url", default="", help="Ollama base URL")
    parser.add_argument("--ollama_model", default="", help="Ollama 模型名")
    parser.add_argument("--page_range", default="", help='页码范围，如 "0,5-10"')
    parser.add_argument("--paginate_output", action="store_true", help="输出按页分页")
    parser.add_argument("--disable_image_extraction", action="store_true", help="不提取图片")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="跳过输出目录中已有结果的文件（仅文件夹批量模式）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="批量 workers 数（默认自动；Mac 建议 1）",
    )
    parser.add_argument(
        "--one_by_one",
        action="store_true",
        help="文件夹时逐个调用 marker_single（更省内存，默认对文件夹用 marker 批量）",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Apple Silicon 默认走 llama.cpp 后端（surya 会自动 spawn llama-server）
    os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")
    # MPS 小模型 + 兼容回退
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    llm_err = check_llm_ready(args)
    if llm_err:
        print(f"[ERROR] {llm_err}", file=sys.stderr)
        return 4

    input_path = Path(args.path).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] 输入路径不存在: {input_path}", file=sys.stderr)
        return 2

    try:
        marker_single, marker_batch = find_marker_bins()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 3

    try:
        files = collect_files(input_path)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    common = build_common_args(args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # 同步给 marker settings（GOOGLE_API_KEY -> gemini_api_key）
    gemini_key = (
        args.gemini_api_key
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    ).strip()
    if gemini_key:
        env["GOOGLE_API_KEY"] = gemini_key

    print(f"[INFO] 输入: {input_path}")
    print(f"[INFO] 输出: {output_dir}")
    print(f"[INFO] 文件数: {len(files)}")
    print(f"[INFO] 模式: {args.mode} | 格式: {args.output_format}")
    print(f"[INFO] disable_ocr={args.disable_ocr} force_ocr={args.force_ocr} use_llm={args.use_llm}")
    if args.use_llm:
        print(f"[INFO] llm_service={resolve_llm_service(args.llm_service)}")
        print(f"[INFO] gemini_key={'已配置' if gemini_key else '未配置'}")
    print(f"[INFO] marker_single: {marker_single}")
    print(f"[INFO] llama-server: {shutil.which('llama-server') or '未找到（OCR/VLM 需要 brew install llama.cpp）'}")
    for f in files:
        print(f"  - {f.name}")

    # 单文件 或 要求逐个：用 marker_single
    # 多文件默认：用 marker 文件夹批处理（更高效，共享 inference server）
    use_batch = input_path.is_dir() and not args.one_by_one and len(files) > 1

    def redact_cmd(cmd: list[str]) -> str:
        """日志中隐藏 API Key。"""
        out: list[str] = []
        hide_next = False
        secret_flags = {
            "--gemini_api_key",
            "--openai_api_key",
            "--openrouter_api_key",
            "--claude_api_key",
            "--azure_api_key",
        }
        for c in cmd:
            if hide_next:
                out.append("***")
                hide_next = False
                continue
            if c in secret_flags:
                out.append(c)
                hide_next = True
                continue
            out.append(c)
        return " ".join(out)

    if use_batch:
        cmd = [marker_batch, str(input_path), *common]
        # 文件夹批处理时 skip_existing / workers 已在 common 里
        print(f"\n[INFO] 批量模式: {redact_cmd(cmd)}")
        try:
            proc = subprocess.run(cmd, env=env, cwd=str(output_dir.parent))
            if proc.returncode != 0:
                print(f"[FAIL] marker 批量退出码: {proc.returncode}", file=sys.stderr)
                return proc.returncode or 1
            print("\n" + "=" * 50)
            print(f"批量完成。结果目录: {output_dir}")
            return 0
        except Exception as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            traceback.print_exc()
            return 1

    ok, failed = 0, []
    for idx, file_path in enumerate(files, 1):
        print(f"\n[INFO] ({idx}/{len(files)}) 开始处理: {file_path.name}")
        cmd = [marker_single, str(file_path), *common]
        # skip_existing / workers 仅对 batch CLI 有意义，单文件去掉
        cmd = [c for c in cmd if c not in ("--skip_existing",)]
        # 去掉 --workers N
        cleaned: list[str] = []
        skip_next = False
        for i, c in enumerate(cmd):
            if skip_next:
                skip_next = False
                continue
            if c == "--workers":
                skip_next = True
                continue
            cleaned.append(c)
        cmd = cleaned
        print(f"[INFO] CMD: {redact_cmd(cmd)}")
        try:
            proc = subprocess.run(cmd, env=env)
            if proc.returncode != 0:
                raise RuntimeError(f"退出码 {proc.returncode}")
            print(f"[OK] 完成: {file_path.name}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {file_path.name}: {e}", file=sys.stderr)
            traceback.print_exc()
            failed.append((file_path.name, str(e) or repr(e)))

    print("\n" + "=" * 50)
    print(f"成功: {ok}/{len(files)}")
    if failed:
        print("失败列表:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        return 1
    print("全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
