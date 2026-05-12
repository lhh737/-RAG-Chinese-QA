"""
模型下载工具：BGE-M3 嵌入 + BGE-Reranker 重排序
支持三种下载源（按优先级）：
  1. ModelScope（阿里，中国大陆极快）
  2. HuggingFace 镜像站 hf-mirror.com
  3. HuggingFace 直连（国际网络）

用法：
  python scripts/download_models.py          # 下载全部模型
  python scripts/download_models.py --embed  # 仅下载嵌入模型
  python scripts/download_models.py --rerank # 仅下载重排序模型
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 模型清单 ──────────────────────────────────────────────

MODELS = {
    "bge-m3": {
        "dir": "models/bge-m3",
        "modelscope": "BAAI/bge-m3",
        "huggingface": "BAAI/bge-m3",
        "description": "BGE-M3 嵌入模型（中文语义检索，1024维）",
    },
    "bge-reranker-base": {
        "dir": "models/bge-reranker-base",
        "modelscope": "BAAI/bge-reranker-base",
        "huggingface": "BAAI/bge-reranker-base",
        "description": "BGE-Reranker 重排序模型（Cross-Encoder）",
    },
}

# ── 下载源 ──────────────────────────────────────────────

HF_MIRROR = "https://hf-mirror.com"


def _pip_install(package: str):
    """确保依赖已安装。"""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", package]
    )


def download_modelscope(model_name: str, target_dir: str) -> bool:
    """通过 ModelScope 下载（推荐，中国大陆最快）。"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("  [modelscope] 未安装 modelscope，尝试安装...")
        _pip_install("modelscope")
        from modelscope.hub.snapshot_download import snapshot_download

    try:
        print(f"  [modelscope] 正在下载 {model_name} → {target_dir} ...")
        os.makedirs(target_dir, exist_ok=True)
        snapshot_download(model_name, cache_dir=target_dir)
        print(f"  [modelscope] 下载完成: {target_dir}")
        return True
    except Exception as e:
        print(f"  [modelscope] 下载失败: {e}")
        return False


def download_hf_mirror(model_name: str, target_dir: str) -> bool:
    """通过 HuggingFace 镜像站下载。"""
    old_mirror = os.environ.get("HF_ENDPOINT", "")
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    try:
        return _download_hf_direct(model_name, target_dir)
    finally:
        if old_mirror:
            os.environ["HF_ENDPOINT"] = old_mirror
        else:
            os.environ.pop("HF_ENDPOINT", None)


def _download_hf_direct(model_name: str, target_dir: str) -> bool:
    """通过 HuggingFace hub 直接下载。"""
    from huggingface_hub import snapshot_download as hf_snapshot

    try:
        print(f"  [huggingface] 正在下载 {model_name} → {target_dir} ...")
        os.makedirs(target_dir, exist_ok=True)
        hf_snapshot(
            repo_id=model_name,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  [huggingface] 下载完成: {target_dir}")
        return True
    except Exception as e:
        print(f"  [huggingface] 下载失败: {e}")
        return False


def download_model(key: str, config: dict):
    """尝试所有下载源，直到成功。"""
    print(f"\n{'='*60}")
    print(f"模型: {config['description']}")
    print(f"目录: {config['dir']}")
    print(f"{'='*60}")

    target_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        config["dir"],
    )

    # 检查是否已存在
    if os.path.isfile(os.path.join(target_dir, "config.json")):
        print("  ✓ 模型已存在，跳过下载")
        return True

    # 按优先级尝试下载源
    sources = [
        ("ModelScope（推荐）", lambda: download_modelscope(config["modelscope"], target_dir)),
        ("HF 镜像站 hf-mirror.com", lambda: download_hf_mirror(config["huggingface"], target_dir)),
        ("HF 直连", lambda: _download_hf_direct(config["huggingface"], target_dir)),
    ]

    for name, fn in sources:
        print(f"\n  尝试 {name} ...")
        if fn():
            # 验证下载完整性
            if os.path.isfile(os.path.join(target_dir, "config.json")):
                print(f"  ✓ 下载验证通过")
                return True
            else:
                print(f"  ⚠ 下载可能不完整（缺少 config.json）")

    print(f"  ✗ 所有下载源均失败")
    return False


def main():
    parser = argparse.ArgumentParser(description="下载 RAG 系统所需模型")
    parser.add_argument("--embed", action="store_true", help="仅下载嵌入模型")
    parser.add_argument("--rerank", action="store_true", help="仅下载重排序模型")
    args = parser.parse_args()

    # 前置依赖
    print("检查依赖...")
    _pip_install("huggingface_hub")

    to_download = []
    if args.embed:
        to_download.append(("bge-m3", MODELS["bge-m3"]))
    elif args.rerank:
        to_download.append(("bge-reranker-base", MODELS["bge-reranker-base"]))
    else:
        to_download = list(MODELS.items())

    success = True
    for key, config in to_download:
        if not download_model(key, config):
            success = False

    print(f"\n{'='*60}")
    if success:
        print("所有模型下载完成！")
        print("\n下一步：运行 python scripts/verify_models.py 验证模型加载")
    else:
        print("部分模型下载失败，请检查网络后重试")


if __name__ == "__main__":
    main()
