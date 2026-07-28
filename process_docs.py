#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marker 稳定批处理入口：支持单个文件或整个文件夹。

用法:
  python process_docs.py -p /path/to/file.pdf -o /path/to/output
  python process_docs.py -p /path/to/folder -o /path/to/output --mode fast --preset speed
  python process_docs.py -p /path/to/file.pdf -o /path/to/output --disable_ocr
  python process_docs.py -p /path/to/file.pdf -o /path/to/output --chunk_pages 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
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

APP_DIR = Path(__file__).resolve().parent

LLM_SERVICES = {
    "gemini": "marker.services.gemini.GoogleGeminiService",
    "openai": "marker.services.openai.OpenAIService",
    "ollama": "marker.services.ollama.OllamaService",
    "openrouter": "marker.services.openrouter.OpenRouterService",
}

# 性能预设：在默认参数上覆盖（用户显式参数优先）
PRESETS: dict[str, dict] = {
    "speed": {
        "mode": "fast",
        "lowres_image_dpi": 72,
        "highres_image_dpi": 144,
        "disable_image_extraction": True,
        "chunk_pages": 8,
        "desc": "最快：较低 DPI、不抽图、长文档分页块处理",
    },
    "balanced": {
        "mode": "fast",
        "lowres_image_dpi": 96,
        "highres_image_dpi": 192,
        "disable_image_extraction": False,
        "chunk_pages": 0,
        "desc": "均衡：默认 DPI，完整图片提取",
    },
    "quality": {
        "mode": "balanced",
        "lowres_image_dpi": 96,
        "highres_image_dpi": 192,
        "disable_image_extraction": False,
        "chunk_pages": 0,
        "desc": "质量优先：VLM 布局（Mac 更慢）",
    },
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


def ensure_surya_llamacpp_grammar_patch() -> None:
    """
    Mac/llama.cpp 上 balanced 模式需要此补丁，否则 layout 全失败、输出空白。
    上游: https://github.com/datalab-to/surya/pull/539
    """
    try:
        import surya  # type: ignore
    except Exception:
        return
    prompts = Path(surya.__file__).resolve().parent / "inference" / "prompts.py"
    if not prompts.is_file():
        return
    try:
        text = prompts.read_text(encoding="utf-8")
    except Exception:
        return
    old = r'r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$"'
    new = r'r"^[0-9]{1,4} [0-9]{1,4} [0-9]{1,4} [0-9]{1,4}$"'
    if old not in text:
        return
    try:
        bak = prompts.with_suffix(".py.bak")
        if not bak.exists():
            bak.write_text(text, encoding="utf-8")
        prompts.write_text(text.replace(old, new), encoding="utf-8")
        print(
            "[INFO] 已自动修补 surya bbox schema（\\d→[0-9]），"
            "修复 Mac balanced 模式 layout 失败问题。"
        )
    except Exception as e:
        print(f"[WARN] 无法自动修补 surya: {e}", file=sys.stderr)


def apply_mac_perf_env(env: dict[str, str], *, disable_ocr: bool) -> dict[str, str]:
    """
    尽量吃满本机算力（不覆盖用户已在环境/local.env 里设的值）。

    说明：无法「超频」M1，只能把 GPU(Metal)/CPU 线程与并发配到合适点。
    16GB 统一内存上并发过高会触发 Swap，反而更慢。
    """
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # 小模型走 Apple GPU（Metal/MPS）
    env.setdefault("TORCH_DEVICE", "mps")
    # 尽量把 GGUF 层放进 GPU（99 = 全部能放的层）
    env.setdefault("LLAMA_CPP_NGL", "99")
    # 跨多次调用保持 llama-server，避免每次冷启动 10–30s+ 模型加载
    env.setdefault("SURYA_INFERENCE_KEEP_ALIVE", "1")

    cpu = os.cpu_count() or 4
    # 给 llama.cpp 的 CPU 线程：M1 Pro 建议接近物理核数
    llama_threads = max(4, min(cpu, 8))
    # PyTorch/OpenMP 线程略少，避免和 llama Metal 抢满
    omp_threads = max(2, min(4, cpu // 2))
    env.setdefault("OMP_NUM_THREADS", str(omp_threads))
    env.setdefault("MKL_NUM_THREADS", str(omp_threads))
    env.setdefault("VECLIB_MAXIMUM_THREADS", str(omp_threads))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    # llama-server 额外参数：线程数、Flash Attention、提高进程优先级
    # 用户可在 local.env 用 LLAMA_CPP_EXTRA_ARGS 覆盖
    env.setdefault(
        "LLAMA_CPP_EXTRA_ARGS",
        f"--threads {llama_threads} --threads-batch {llama_threads} "
        f"--flash-attn on --prio 2",
    )

    # VLM 并发槽位：16GB 上 2 较稳；内存充裕可在 local.env 改成 4
    # 并发越高越吃统一内存，OOM/Swap 会整体变慢
    if not disable_ocr:
        env.setdefault("SURYA_INFERENCE_PARALLEL", "2")
        # 每槽上下文；过小截断，过大占内存
        env.setdefault("SURYA_INFERENCE_CTX_PER_SLOT", "8192")
    return env


def resolve_llm_service(name: str) -> str:
    name = (name or "gemini").strip()
    if name in LLM_SERVICES:
        return LLM_SERVICES[name]
    if "." in name:
        return name
    raise ValueError(f"未知 LLM 服务: {name}，可选: {', '.join(LLM_SERVICES)}")


def check_llm_ready(args: argparse.Namespace) -> str | None:
    if not args.use_llm:
        return None
    service = resolve_llm_service(args.llm_service)
    gemini_key = (
        args.gemini_api_key
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    ).strip()
    openai_key = (args.openai_api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
    openrouter_key = (
        args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY") or ""
    ).strip()

    if service.endswith("GoogleGeminiService"):
        if not gemini_key:
            return (
                "已启用 --use_llm，默认使用 Google Gemini，但未配置 API Key。\n"
                "请取消 LLM，或设置 GOOGLE_API_KEY / local.env / --gemini_api_key。"
            )
    elif service.endswith("OpenAIService") and not openai_key:
        return "已启用 OpenAI LLM，但未配置 OPENAI_API_KEY。"
    elif service.endswith("OpenRouterService") and not openrouter_key:
        return "已启用 OpenRouter，但未配置 OPENROUTER_API_KEY。"
    return None


def pdf_page_count(path: Path) -> int | None:
    if path.suffix.lower() != ".pdf":
        return None
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(path))
        n = len(doc)
        doc.close()
        return int(n)
    except Exception:
        return None


def page_chunks(total: int, chunk_size: int) -> list[tuple[int, int]]:
    """返回 0-based inclusive (start, end) 页块。"""
    if total <= 0:
        return []
    if chunk_size <= 0 or chunk_size >= total:
        return [(0, total - 1)]
    chunks = []
    for start in range(0, total, chunk_size):
        end = min(total - 1, start + chunk_size - 1)
        chunks.append((start, end))
    return chunks


# ---------------------------------------------------------------------------
# 输出目录进度（解决「处理中输出目录空白」）
# ---------------------------------------------------------------------------


class ProgressTracker:
    """在输出目录写入可读进度文件，Finder 中可看到处理状态。"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path_md = output_dir / "_marker_progress.md"
        self.path_json = output_dir / "_marker_progress.json"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state: dict = {
            "status": "starting",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_sec": 0,
            "current_file": "",
            "file_index": 0,
            "file_total": 0,
            "page_info": "",
            "message": "初始化…",
            "files_done": [],
            "files_failed": [],
        }
        self._t0 = time.time()

    def update(self, **kwargs) -> None:
        with self._lock:
            self.state.update(kwargs)
            self.state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self.state["elapsed_sec"] = int(time.time() - self._t0)
            self._write()

    def _write(self) -> None:
        s = self.state
        lines = [
            f"# Marker 处理进度",
            f"",
            f"- **状态**: `{s.get('status')}`",
            f"- **开始**: {s.get('started_at')}",
            f"- **更新**: {s.get('updated_at')}",
            f"- **已用时**: {s.get('elapsed_sec')} 秒",
            f"- **当前文件**: {s.get('current_file') or '-'} "
            f"({s.get('file_index')}/{s.get('file_total')})",
            f"- **页进度**: {s.get('page_info') or '-'}",
            f"- **说明**: {s.get('message')}",
            f"",
            f"> Marker 默认在整份文档处理完后才写入最终 `.md`。"
            f" 若开启分页块（`--chunk_pages`），会在每块完成后更新结果。",
            f"",
        ]
        done = s.get("files_done") or []
        failed = s.get("files_failed") or []
        if done:
            lines.append("## 已完成")
            for name in done:
                lines.append(f"- ✅ {name}")
            lines.append("")
        if failed:
            lines.append("## 失败")
            for name in failed:
                lines.append(f"- ❌ {name}")
            lines.append("")
        try:
            self.path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.path_json.write_text(
                json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            # 触发目录 mtime 变化，便于 Finder 刷新
            os.utime(self.output_dir, None)
        except Exception as e:
            print(f"[WARN] 写进度文件失败: {e}", file=sys.stderr)

    def start_heartbeat(self, interval: float = 3.0) -> None:
        def _loop() -> None:
            while not self._stop.wait(interval):
                with self._lock:
                    self.state["elapsed_sec"] = int(time.time() - self._t0)
                    self.state["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    # 心跳时刷新「仍在处理」提示
                    msg = self.state.get("message") or ""
                    if self.state.get("status") == "running" and "处理中" not in msg:
                        self.state["message"] = f"{msg}（处理中…）" if msg else "处理中…"
                    self._write()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def mark_file_processing(self, file_path: Path) -> Path:
        """为当前文件创建结果子目录并写入「处理中」标记。"""
        dest = self.output_dir / file_path.stem
        dest.mkdir(parents=True, exist_ok=True)
        flag = dest / "_processing.txt"
        flag.write_text(
            f"正在处理: {file_path.name}\n"
            f"开始时间: {datetime.now().isoformat(timespec='seconds')}\n"
            f"完成后此文件会被删除，并生成 .md / 图片。\n",
            encoding="utf-8",
        )
        os.utime(dest, None)
        return dest

    def clear_file_processing(self, file_path: Path) -> None:
        flag = self.output_dir / file_path.stem / "_processing.txt"
        try:
            if flag.is_file():
                flag.unlink()
        except Exception:
            pass


def redact_cmd(cmd: list[str]) -> str:
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
    if args.lowres_image_dpi is not None:
        cmd.extend(["--lowres_image_dpi", str(args.lowres_image_dpi)])
    if args.highres_image_dpi is not None:
        cmd.extend(["--highres_image_dpi", str(args.highres_image_dpi)])
    return cmd


def run_cmd(cmd: list[str], env: dict[str, str]) -> int:
    print(f"[INFO] CMD: {redact_cmd(cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env)
    return proc.returncode or 0


def merge_chunk_markdown(chunk_dirs: list[Path], final_dir: Path, stem: str) -> None:
    """把分页块输出的 markdown 合并到 final_dir/stem.md。"""
    parts: list[str] = []
    images_copied = 0
    final_dir.mkdir(parents=True, exist_ok=True)
    for cdir in chunk_dirs:
        md_candidates = list(cdir.glob("*.md"))
        if not md_candidates:
            # 嵌套：output/chunk/stem/stem.md
            md_candidates = list(cdir.glob("**/*.md"))
            md_candidates = [p for p in md_candidates if p.name != "_marker_progress.md"]
        for md in sorted(md_candidates):
            text = md.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                parts.append(text)
            # 复制图片到最终目录
            for img in md.parent.glob("*.jpeg"):
                dest = final_dir / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                    images_copied += 1
            for img in md.parent.glob("*.jpg"):
                dest = final_dir / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                    images_copied += 1
            for img in md.parent.glob("*.png"):
                dest = final_dir / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                    images_copied += 1

    out_md = final_dir / f"{stem}.md"
    out_md.write_text("\n\n".join(parts) + ("\n" if parts else ""), encoding="utf-8")
    meta = {
        "merged_chunks": len(chunk_dirs),
        "images_copied": images_copied,
        "merged_at": datetime.now().isoformat(timespec="seconds"),
    }
    (final_dir / f"{stem}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[INFO] 已合并 {len(parts)} 段 → {out_md}", flush=True)


_FLAGS_WITH_VALUE = {
    "--output_dir",
    "--page_range",
    "--workers",
    "--output_format",
    "--mode",
    "--llm_service",
    "--gemini_api_key",
    "--openai_api_key",
    "--openai_base_url",
    "--openai_model",
    "--openrouter_api_key",
    "--ollama_base_url",
    "--ollama_model",
    "--lowres_image_dpi",
    "--highres_image_dpi",
}


def strip_cli_flags(args_list: list[str], flags: set[str]) -> list[str]:
    """从 CLI 参数列表中去掉指定 flag（及其取值）。"""
    out: list[str] = []
    i = 0
    while i < len(args_list):
        c = args_list[i]
        if c in flags:
            if c in _FLAGS_WITH_VALUE:
                i += 2
            else:
                i += 1
            continue
        out.append(c)
        i += 1
    return out


def convert_one_file(
    *,
    file_path: Path,
    marker_single: str,
    common: list[str],
    env: dict[str, str],
    args: argparse.Namespace,
    progress: ProgressTracker,
    file_index: int,
    file_total: int,
) -> None:
    """转换单个文件；可选分页块以便输出目录尽早出现结果。"""
    progress.mark_file_processing(file_path)
    progress.update(
        status="running",
        current_file=file_path.name,
        file_index=file_index,
        file_total=file_total,
        page_info="",
        message=f"开始处理 {file_path.name}",
    )

    base_common = strip_cli_flags(
        common,
        {"--skip_existing", "--workers", "--page_range"},
    )

    user_range = (args.page_range or "").strip()
    chunk_pages = int(args.chunk_pages or 0)
    pages = pdf_page_count(file_path) if not user_range else None

    use_chunks = (
        not user_range
        and chunk_pages > 0
        and pages is not None
        and pages > chunk_pages
        and file_path.suffix.lower() == ".pdf"
    )

    if not use_chunks:
        cmd = [marker_single, str(file_path), *base_common]
        if user_range:
            cmd.extend(["--page_range", user_range])
        progress.update(
            message=f"整份转换中（共 {pages or '?'} 页）…",
            page_info=f"全部 / {pages or '?'}" if pages else (user_range or "全部"),
        )
        code = run_cmd(cmd, env)
        progress.clear_file_processing(file_path)
        if code != 0:
            raise RuntimeError(f"退出码 {code}")
        out_sub = progress.output_dir / file_path.stem
        mds = list(out_sub.glob("*.md")) if out_sub.is_dir() else []
        progress.update(
            message=f"完成 {file_path.name}"
            + (f"（生成 {len(mds)} 个 md）" if mds else ""),
        )
        return

    # —— 分页块：每块完成后合并到最终目录，Finder 可见逐步增长的 .md ——
    chunks = page_chunks(pages, chunk_pages)
    final_dir = progress.output_dir / file_path.stem
    final_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = final_dir / "_chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_dirs: list[Path] = []

    # base_common 自带全局 --output_dir，分块时需要替换
    flags_no_outdir = strip_cli_flags(base_common, {"--output_dir"})

    print(
        f"[INFO] 分页块处理: {pages} 页 → {len(chunks)} 块（每块最多 {chunk_pages} 页）",
        flush=True,
    )
    for ci, (start, end) in enumerate(chunks, 1):
        cdir = chunk_root / f"p{start:04d}-{end:04d}"
        cdir.mkdir(parents=True, exist_ok=True)
        pr = f"{start}-{end}" if start != end else str(start)
        progress.update(
            page_info=f"块 {ci}/{len(chunks)}  页 {start}-{end}（共 {pages}）",
            message=f"正在转换第 {start}-{end} 页…",
        )
        (cdir / "_chunk_status.txt").write_text(
            f"processing pages {start}-{end}\n"
            f"started {datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
        cmd = [
            marker_single,
            str(file_path),
            "--output_dir",
            str(cdir),
            *flags_no_outdir,
            "--page_range",
            pr,
        ]
        code = run_cmd(cmd, env)
        if code != 0:
            progress.clear_file_processing(file_path)
            raise RuntimeError(f"页块 {start}-{end} 退出码 {code}")

        (cdir / "_chunk_status.txt").write_text(
            f"done pages {start}-{end}\n"
            f"finished {datetime.now().isoformat(timespec='seconds')}\n",
            encoding="utf-8",
        )
        chunk_dirs.append(cdir)
        merge_chunk_markdown(chunk_dirs, final_dir, file_path.stem)
        progress.update(
            message=f"已完成页 {start}-{end}，结果已写入 {final_dir.name}/",
        )

    progress.clear_file_processing(file_path)


def apply_preset_to_args(args: argparse.Namespace) -> None:
    """用 preset 填充默认字段。显式 CLI 参数优先于预设。"""
    preset_name = getattr(args, "preset", "balanced") or "balanced"
    preset = PRESETS.get(preset_name, PRESETS["balanced"])

    # mode：quality 预设切 balanced；speed 强制 fast（除非用户写了 --mode）
    # 用 sys.argv 判断用户是否显式传了 --mode
    user_set_mode = any(a == "--mode" or a.startswith("--mode=") for a in sys.argv)
    if not user_set_mode:
        if preset_name == "quality":
            args.mode = "balanced"
        elif preset_name == "speed":
            args.mode = "fast"

    if args.lowres_image_dpi is None and "lowres_image_dpi" in preset:
        args.lowres_image_dpi = preset["lowres_image_dpi"]
    if args.highres_image_dpi is None and "highres_image_dpi" in preset:
        args.highres_image_dpi = preset["highres_image_dpi"]

    # chunk_pages: -1 = follow preset
    if args.chunk_pages < 0:
        args.chunk_pages = int(preset.get("chunk_pages") or 0)

    user_set_no_img = "--disable_image_extraction" in sys.argv
    if preset_name == "speed" and not args.keep_images and not user_set_no_img:
        args.disable_image_extraction = True
    if args.keep_images:
        args.disable_image_extraction = False


def main() -> int:
    load_dotenv_files()

    parser = argparse.ArgumentParser(
        description="Marker document conversion (single file or folder)"
    )
    parser.add_argument("-p", "--path", required=True, help="PDF/image/office file or directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--preset",
        default="balanced",
        choices=list(PRESETS.keys()),
        help="性能预设: speed（最快）/ balanced（默认）/ quality（最准更慢）",
    )
    parser.add_argument(
        "--mode",
        default="fast",
        choices=["fast", "balanced"],
        help="Conversion mode (Mac 推荐 fast)",
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
        help="纯文本层提取，不启动 VLM（电子 PDF 最快）",
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
        help="LLM 后端: gemini | openai | ollama | openrouter",
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
    parser.add_argument(
        "--disable_image_extraction",
        action="store_true",
        help="不提取图片（更快、输出更小）",
    )
    parser.add_argument(
        "--keep_images",
        action="store_true",
        help="强制提取图片（覆盖 speed 预设的默认不抽图）",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="跳过输出目录中已有结果的文件（仅文件夹批量模式）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="批量 workers 数（默认自动；OCR 时 Mac 建议 1）",
    )
    parser.add_argument(
        "--one_by_one",
        action="store_true",
        help="文件夹时逐个调用 marker_single",
    )
    parser.add_argument(
        "--chunk_pages",
        type=int,
        default=-1,
        help="长 PDF 按 N 页一块处理并即时写入输出（0=关闭，-1=跟随预设）",
    )
    parser.add_argument(
        "--lowres_image_dpi",
        type=int,
        default=None,
        help="布局用低分辨率 DPI（默认 96；speed 预设 72）",
    )
    parser.add_argument(
        "--highres_image_dpi",
        type=int,
        default=None,
        help="OCR 用高分辨率 DPI（默认 192；speed 预设 144）",
    )
    args = parser.parse_args()
    apply_preset_to_args(args)

    ensure_surya_llamacpp_grammar_patch()
    if args.mode == "balanced":
        print(
            "[INFO] balanced 模式在 Mac 上较慢；若异常请改用 --mode fast 或 --preset speed。",
            flush=True,
        )

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

    # skip_existing（单文件路径也支持）
    if args.skip_existing:
        kept = []
        for f in files:
            out_sub = output_dir / f.stem
            if out_sub.is_dir() and any(out_sub.glob("*.md")):
                print(f"[SKIP] 已有输出: {f.name}")
            else:
                kept.append(f)
        files = kept
        if not files:
            print("[INFO] 全部已存在，无需处理。")
            return 0

    common = build_common_args(args)
    env = apply_mac_perf_env(os.environ.copy(), disable_ocr=args.disable_ocr)
    gemini_key = (
        args.gemini_api_key
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    ).strip()
    if gemini_key:
        env["GOOGLE_API_KEY"] = gemini_key

    progress = ProgressTracker(output_dir)
    progress.update(
        status="running",
        file_total=len(files),
        message=f"预设={args.preset} mode={args.mode} 共 {len(files)} 个文件",
    )
    progress.start_heartbeat(3.0)

    print(f"[INFO] 输入: {input_path}", flush=True)
    print(f"[INFO] 输出: {output_dir}", flush=True)
    print(f"[INFO] 进度文件: {progress.path_md}", flush=True)
    print(f"[INFO] 文件数: {len(files)}", flush=True)
    print(
        f"[INFO] 预设: {args.preset} | 模式: {args.mode} | 格式: {args.output_format}",
        flush=True,
    )
    print(
        f"[INFO] disable_ocr={args.disable_ocr} force_ocr={args.force_ocr} "
        f"use_llm={args.use_llm} no_images={args.disable_image_extraction}",
        flush=True,
    )
    print(
        f"[INFO] DPI low={args.lowres_image_dpi} high={args.highres_image_dpi} "
        f"chunk_pages={args.chunk_pages}",
        flush=True,
    )
    print(
        f"[INFO] KEEP_ALIVE={env.get('SURYA_INFERENCE_KEEP_ALIVE')} "
        f"TORCH_DEVICE={env.get('TORCH_DEVICE')} "
        f"OMP={env.get('OMP_NUM_THREADS')}",
        flush=True,
    )
    print(f"[INFO] marker_single: {marker_single}", flush=True)
    print(
        f"[INFO] llama-server: {shutil.which('llama-server') or '未找到'}",
        flush=True,
    )
    for f in files:
        print(f"  - {f.name}", flush=True)

    # 多文件批量：仅当不分块且不要求逐个时用官方 marker 批处理
    use_batch = (
        input_path.is_dir()
        and not args.one_by_one
        and len(files) > 1
        and int(args.chunk_pages or 0) <= 0
    )

    try:
        if use_batch:
            # 批量时 workers：OCR 默认 1，disable_ocr 可用 2
            if args.workers is None:
                w = 2 if args.disable_ocr else 1
                common_batch = list(common)
                if "--workers" not in common_batch:
                    common_batch.extend(["--workers", str(w)])
            else:
                common_batch = common
            cmd = [marker_batch, str(input_path), *common_batch]
            progress.update(message="批量转换中…", current_file="(batch)")
            # 为每个文件预创建处理中标记
            for f in files:
                progress.mark_file_processing(f)
            code = run_cmd(cmd, env)
            for f in files:
                progress.clear_file_processing(f)
            if code != 0:
                progress.update(status="failed", message=f"批量失败 code={code}")
                print(f"[FAIL] marker 批量退出码: {code}", file=sys.stderr)
                return code or 1
            progress.update(
                status="done",
                message="批量完成",
                files_done=[f.name for f in files],
            )
            print("\n" + "=" * 50)
            print(f"批量完成。结果目录: {output_dir}")
            print(f"进度文件: {progress.path_md}")
            return 0

        ok, failed = 0, []
        done_names: list[str] = []
        fail_names: list[str] = []
        for idx, file_path in enumerate(files, 1):
            print(f"\n[INFO] ({idx}/{len(files)}) 开始处理: {file_path.name}", flush=True)
            try:
                convert_one_file(
                    file_path=file_path,
                    marker_single=marker_single,
                    common=common,
                    env=env,
                    args=args,
                    progress=progress,
                    file_index=idx,
                    file_total=len(files),
                )
                print(f"[OK] 完成: {file_path.name}", flush=True)
                ok += 1
                done_names.append(file_path.name)
                progress.update(files_done=list(done_names), files_failed=list(fail_names))
            except Exception as e:
                print(f"[FAIL] {file_path.name}: {e}", file=sys.stderr)
                traceback.print_exc()
                failed.append((file_path.name, str(e) or repr(e)))
                fail_names.append(file_path.name)
                progress.update(files_done=list(done_names), files_failed=list(fail_names))
                progress.clear_file_processing(file_path)

        print("\n" + "=" * 50)
        print(f"成功: {ok}/{len(files)}")
        print(f"输出目录: {output_dir}")
        print(f"进度文件: {progress.path_md}")
        if failed:
            print("失败列表:")
            for name, msg in failed:
                print(f"  - {name}: {msg}")
            progress.update(
                status="failed",
                message=f"完成 {ok}/{len(files)}，有失败",
            )
            return 1
        progress.update(status="done", message="全部完成")
        print("全部完成。")
        return 0
    finally:
        progress.stop()


if __name__ == "__main__":
    sys.exit(main())
