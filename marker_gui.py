#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Marker 图形界面：选择 PDF 文件或文件夹，一键转为 Markdown / JSON / HTML。

用法:
    conda activate marker
    python marker_gui.py

或:
    ./start_gui.sh
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from tkinter import (
    END,
    BooleanVar,
    StringVar,
    Tk,
    filedialog,
    messagebox,
    scrolledtext,
    ttk,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = APP_DIR / "output"
PROCESS_SCRIPT = APP_DIR / "process_docs.py"

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

MODE_OPTIONS = [
    ("fast", "fast（推荐 Mac：更快，CPU/MPS 默认）"),
    ("balanced", "balanced（更高精度，更吃资源）"),
]

FORMAT_OPTIONS = ["markdown", "json", "html", "chunks"]

LLM_SERVICE_OPTIONS = [
    ("gemini", "Gemini（需 GOOGLE_API_KEY）"),
    ("openai", "OpenAI 兼容（需 OPENAI_API_KEY）"),
    ("ollama", "Ollama 本地（无需 Key）"),
    ("openrouter", "OpenRouter（需 OPENROUTER_API_KEY）"),
]


def load_dotenv_files() -> None:
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


def find_python_bin() -> str:
    """定位 marker conda 环境中的 python。"""
    if sys.executable and "marker" in sys.executable:
        return sys.executable
    candidates = [
        Path("/opt/anaconda3/envs/marker/bin/python"),
        Path.home() / "anaconda3/envs/marker/bin/python",
        Path.home() / "miniconda3/envs/marker/bin/python",
        APP_DIR / ".venv/bin/python",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    which = subprocess.run(["which", "python"], capture_output=True, text=True)
    if which.returncode == 0 and which.stdout.strip():
        return which.stdout.strip()
    return sys.executable or "python"


def has_gemini_key() -> bool:
    return bool(
        (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        ).strip()
    )


class MarkerGUI:
    def __init__(self, root: Tk) -> None:
        load_dotenv_files()
        self.root = root
        self.root.title("Marker 文档解析工具")
        self.root.geometry("900x780")
        self.root.minsize(720, 640)

        self.input_path = StringVar()
        self.output_path = StringVar(value=str(DEFAULT_OUTPUT))
        self.mode = StringVar(value="fast")
        self.output_format = StringVar(value="markdown")
        self.disable_ocr = BooleanVar(value=False)
        self.force_ocr = BooleanVar(value=False)
        self.use_llm = BooleanVar(value=False)
        self.llm_service = StringVar(value="gemini")
        self.gemini_api_key = StringVar(
            value=(
                os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or ""
            ).strip()
        )
        self.skip_existing = BooleanVar(value=False)
        self.one_by_one = BooleanVar(value=False)
        self.running = False
        self.proc: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_log()

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Marker 文档解析", font=("", 16, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(
            main,
            text="选择 PDF 文件或文件夹，点击「开始处理」自动转为 Markdown / JSON / HTML",
            foreground="#555",
        ).pack(anchor="w", pady=(0, 10))

        # —— 输入 ——
        row_in = ttk.LabelFrame(main, text="输入（文件或文件夹）", padding=8)
        row_in.pack(fill="x", **pad)
        ttk.Entry(row_in, textvariable=self.input_path).pack(
            side="left", fill="x", expand=True, padx=(0, 6)
        )
        ttk.Button(row_in, text="选择文件…", command=self._pick_file).pack(side="left", padx=2)
        ttk.Button(row_in, text="选择文件夹…", command=self._pick_folder).pack(
            side="left", padx=2
        )

        # —— 输出 ——
        row_out = ttk.LabelFrame(main, text="输出目录", padding=8)
        row_out.pack(fill="x", **pad)
        ttk.Entry(row_out, textvariable=self.output_path).pack(
            side="left", fill="x", expand=True, padx=(0, 6)
        )
        ttk.Button(row_out, text="选择目录…", command=self._pick_output).pack(side="left")

        # —— 参数 ——
        opts = ttk.LabelFrame(main, text="解析参数", padding=8)
        opts.pack(fill="x", **pad)

        g = ttk.Frame(opts)
        g.pack(fill="x")

        ttk.Label(g, text="模式:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        mode_combo = ttk.Combobox(
            g,
            textvariable=self.mode,
            values=[m[0] for m in MODE_OPTIONS],
            state="readonly",
            width=18,
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        self.mode_hint = ttk.Label(g, text=MODE_OPTIONS[0][1], foreground="#666")
        self.mode_hint.grid(row=0, column=2, sticky="w", padx=4, pady=4)
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        ttk.Label(g, text="输出格式:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            g,
            textvariable=self.output_format,
            values=FORMAT_OPTIONS,
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(g, text="一般选 markdown", foreground="#666").grid(
            row=1, column=2, sticky="w", padx=4, pady=4
        )

        flags = ttk.Frame(opts)
        flags.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(
            flags, text="禁用 OCR（最快，纯文本层）", variable=self.disable_ocr
        ).pack(side="left", padx=8)
        ttk.Checkbutton(flags, text="强制 OCR", variable=self.force_ocr).pack(
            side="left", padx=8
        )
        ttk.Checkbutton(
            flags, text="使用 LLM 增强（需 API Key）", variable=self.use_llm
        ).pack(side="left", padx=8)
        ttk.Checkbutton(flags, text="跳过已有输出", variable=self.skip_existing).pack(
            side="left", padx=8
        )
        ttk.Checkbutton(flags, text="逐个处理（省内存）", variable=self.one_by_one).pack(
            side="left", padx=8
        )

        # —— LLM（可选）——
        llm_row = ttk.LabelFrame(
            main,
            text="LLM 增强（可选；不勾选也能正常转换，质量已经很好）",
            padding=8,
        )
        llm_row.pack(fill="x", **pad)
        g2 = ttk.Frame(llm_row)
        g2.pack(fill="x")
        ttk.Label(g2, text="服务:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(
            g2,
            textvariable=self.llm_service,
            values=[s[0] for s in LLM_SERVICE_OPTIONS],
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(
            g2,
            text="默认 gemini；无 Key 请勿勾选「使用 LLM 增强」",
            foreground="#666",
        ).grid(row=0, column=2, sticky="w", padx=4, pady=4)

        ttk.Label(g2, text="Gemini Key:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(g2, textvariable=self.gemini_api_key, show="*", width=48).grid(
            row=1, column=1, columnspan=2, sticky="we", padx=4, pady=4
        )
        ttk.Label(
            g2,
            text="也可写到 local.env: GOOGLE_API_KEY=...  申请: aistudio.google.com/apikey",
            foreground="#666",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=4, pady=2)

        # —— 操作 ——
        actions = ttk.Frame(main)
        actions.pack(fill="x", **pad)
        self.start_btn = ttk.Button(actions, text="▶  开始处理", command=self._start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(
            actions, text="■  停止", command=self._stop, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(actions, text="打开输出目录", command=self._open_output).pack(
            side="left", padx=4
        )
        ttk.Button(actions, text="清空日志", command=self._clear_log).pack(side="left", padx=4)

        self.status_var = StringVar(value="就绪")
        ttk.Label(actions, textvariable=self.status_var, foreground="#0a7").pack(
            side="right", padx=8
        )

        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=(0, 6))

        log_frame = ttk.LabelFrame(main, text="运行日志", padding=6)
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = scrolledtext.ScrolledText(
            log_frame, height=18, wrap="word", font=("Menlo", 11)
        )
        self.log.pack(fill="both", expand=True)
        self._log("Marker GUI 已启动。")
        self._log(f"Python: {find_python_bin()}")
        self._log(f"处理脚本: {PROCESS_SCRIPT}")
        self._log("请选择文件或文件夹，然后点击「开始处理」。")
        self._log("Mac 推荐：模式 fast；扫描件可开「强制 OCR」；纯电子档可开「禁用 OCR」提速。")
        if has_gemini_key() or self.gemini_api_key.get().strip():
            self._log("已检测到 Gemini API Key（可用 LLM 增强）。")
        else:
            self._log(
                "未配置 Gemini API Key：请勿勾选「使用 LLM 增强」，"
                "或不增强也能正常转 Markdown。\n"
            )

    def _on_mode_change(self, _event=None) -> None:
        mapping = {m[0]: m[1] for m in MODE_OPTIONS}
        self.mode_hint.config(text=mapping.get(self.mode.get(), ""))

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择要解析的文件",
            filetypes=[
                (
                    "支持的文档",
                    "*.pdf *.png *.jpg *.jpeg *.webp *.docx *.pptx *.xlsx *.html *.epub",
                ),
                ("PDF", "*.pdf"),
                ("所有文件", "*.*"),
            ],
        )
        if path:
            self.input_path.set(path)

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory(title="选择包含文档的文件夹")
        if path:
            self.input_path.set(path)

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_path.set(path)

    def _open_output(self) -> None:
        out = Path(self.output_path.get().strip() or str(DEFAULT_OUTPUT))
        out.mkdir(parents=True, exist_ok=True)
        webbrowser.open(out.as_uri())

    def _clear_log(self) -> None:
        self.log.delete("1.0", END)

    def _log(self, msg: str) -> None:
        self.log.insert(END, msg + "\n")
        self.log.see(END)

    def _poll_log(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._log(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    def _validate(self) -> tuple[Path, Path] | None:
        raw_in = self.input_path.get().strip()
        raw_out = self.output_path.get().strip()
        if not raw_in:
            messagebox.showwarning("提示", "请先选择文件或文件夹。")
            return None
        in_path = Path(raw_in).expanduser().resolve()
        if not in_path.exists():
            messagebox.showerror("错误", f"输入路径不存在：\n{in_path}")
            return None
        if in_path.is_file():
            if in_path.suffix.lower() not in SUPPORTED_EXTS:
                messagebox.showerror(
                    "错误",
                    f"不支持的文件类型：{in_path.suffix}\n"
                    f"支持: {', '.join(sorted(SUPPORTED_EXTS))}",
                )
                return None
        elif in_path.is_dir():
            found = [
                p
                for p in in_path.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ]
            if not found:
                messagebox.showwarning(
                    "提示",
                    f"文件夹中未找到可处理文件：\n{in_path}\n"
                    f"支持 PDF / 图片 / DOCX / PPTX / XLSX / HTML / EPUB",
                )
                return None
        else:
            messagebox.showerror("错误", "输入路径无效。")
            return None

        if self.disable_ocr.get() and self.force_ocr.get():
            messagebox.showwarning("提示", "「禁用 OCR」与「强制 OCR」不能同时勾选。")
            return None

        if self.use_llm.get():
            svc = self.llm_service.get()
            key = self.gemini_api_key.get().strip() or (
                os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or ""
            ).strip()
            if svc == "gemini" and not key:
                messagebox.showerror(
                    "需要 Gemini API Key",
                    "已勾选「使用 LLM 增强」，默认使用 Google Gemini，但未配置 API Key。\n\n"
                    "可选方案：\n"
                    "1. 取消勾选「使用 LLM 增强」（推荐，本地即可转 Markdown）\n"
                    "2. 在下方「Gemini Key」填入密钥\n"
                    "3. 在 local.env 写入: GOOGLE_API_KEY=你的密钥\n\n"
                    "申请: https://aistudio.google.com/apikey",
                )
                return None
            if svc == "openai" and not (
                os.environ.get("OPENAI_API_KEY") or ""
            ).strip():
                messagebox.showerror(
                    "需要 OpenAI API Key",
                    "请设置环境变量 OPENAI_API_KEY，或在 local.env 中配置。",
                )
                return None
            if svc == "openrouter" and not (
                os.environ.get("OPENROUTER_API_KEY") or ""
            ).strip():
                messagebox.showerror(
                    "需要 OpenRouter API Key",
                    "请设置环境变量 OPENROUTER_API_KEY，或在 local.env 中配置。",
                )
                return None

        out_path = Path(raw_out or str(DEFAULT_OUTPUT)).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        return in_path, out_path

    def _build_cmd(self, in_path: Path, out_path: Path) -> list[str]:
        py = find_python_bin()
        if not PROCESS_SCRIPT.is_file():
            raise FileNotFoundError(f"缺少处理脚本: {PROCESS_SCRIPT}")
        cmd = [
            py,
            str(PROCESS_SCRIPT),
            "-p",
            str(in_path),
            "-o",
            str(out_path),
            "--mode",
            self.mode.get(),
            "--output_format",
            self.output_format.get(),
        ]
        if self.disable_ocr.get():
            cmd.append("--disable_ocr")
        if self.force_ocr.get():
            cmd.append("--force_ocr")
        if self.use_llm.get():
            cmd.append("--use_llm")
            cmd.extend(["--llm_service", self.llm_service.get()])
            key = self.gemini_api_key.get().strip()
            if key:
                cmd.extend(["--gemini_api_key", key])
        if self.skip_existing.get():
            cmd.append("--skip_existing")
        if self.one_by_one.get():
            cmd.append("--one_by_one")
        # Mac 批量默认 1 worker，避免 OOM
        cmd.extend(["--workers", "1"])
        return cmd

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        key = self.gemini_api_key.get().strip()
        if key:
            env["GOOGLE_API_KEY"] = key
        py = Path(find_python_bin())
        if py.parent.name == "bin":
            env["PATH"] = f"{py.parent}:{env.get('PATH', '')}"
        # 确保 brew 的 llama-server 在 PATH 里
        brew_bin = "/opt/homebrew/bin"
        if Path(brew_bin).is_dir() and brew_bin not in env.get("PATH", ""):
            env["PATH"] = f"{brew_bin}:{env.get('PATH', '')}"
        return env

    def _run_process(self, cmd: list[str], env: dict[str, str], log_file: Path) -> int:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write("\n" + "=" * 60 + "\n")
            lf.write(f"CMD: {' '.join(cmd)}\n")
            lf.write("=" * 60 + "\n")

        log_fh = open(log_file, "a", encoding="utf-8", errors="replace")
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(APP_DIR),
            )
            with open(log_file, "r", encoding="utf-8", errors="replace") as reader:
                reader.seek(0, os.SEEK_END)
                while True:
                    line = reader.readline()
                    if line:
                        self.log_queue.put(line.rstrip("\n"))
                        continue
                    if self.proc.poll() is not None:
                        rest = reader.read()
                        if rest:
                            for part in rest.splitlines():
                                self.log_queue.put(part)
                        break
                    threading.Event().wait(0.15)
            return self.proc.returncode or 0
        finally:
            try:
                log_fh.close()
            except Exception:
                pass

    def _start(self) -> None:
        if self.running:
            return
        validated = self._validate()
        if not validated:
            return
        in_path, out_path = validated
        try:
            cmd = self._build_cmd(in_path, out_path)
        except Exception as e:
            messagebox.showerror("错误", str(e))
            return

        env = self._build_env()
        log_file = APP_DIR / "logs" / "marker_gui_last.log"

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("处理中…")
        self.progress.start(12)
        self._log(f"\n开始: {in_path}")
        self._log(f"输出: {out_path}")
        self._log(f"命令: {' '.join(cmd)}\n")

        def worker() -> None:
            code = 1
            try:
                code = self._run_process(cmd, env, log_file)
            except Exception as e:
                self.log_queue.put(f"[ERROR] {e}")
            finally:
                self.root.after(0, lambda: self._on_done(code, out_path))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, code: int, out_path: Path) -> None:
        self.running = False
        self.proc = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress.stop()
        if code == 0:
            self.status_var.set("完成")
            self._log(f"\n✓ 全部完成。输出目录: {out_path}")
            messagebox.showinfo("完成", f"处理完成。\n输出目录:\n{out_path}")
        else:
            self.status_var.set(f"失败 (code={code})")
            self._log(f"\n✗ 处理失败，退出码 {code}。详见日志。")
            messagebox.showerror(
                "失败",
                f"处理失败（退出码 {code}）。\n请查看界面日志或 logs/marker_gui_last.log",
            )

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self._log("正在停止…")
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.status_var.set("已停止")
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress.stop()

    def _on_close(self) -> None:
        if self.running:
            if not messagebox.askyesno("退出", "任务仍在运行，确定退出并停止？"):
                return
            self._stop()
        self.root.destroy()


def main() -> None:
    root = Tk()
    # macOS 更清晰的默认字体
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    MarkerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
