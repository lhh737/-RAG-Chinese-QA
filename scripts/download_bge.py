"""
下载 BGE-M3 嵌入模型 + BGE-Reranker 重排序模型
源优先级：hf-mirror.com → huggingface.co
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 两个模型
MODELS = {
    "bge-m3": {
        "repo": "BAAI/bge-m3",
        "dir": os.path.join(MODELS_DIR, "bge-m3"),
        "desc": "BGE-M3 嵌入模型（1024维，中文语义检索）",
        "size": "~2.2 GB",
    },
    "bge-reranker-base": {
        "repo": "BAAI/bge-reranker-base",
        "dir": os.path.join(MODELS_DIR, "bge-reranker-base"),
        "desc": "BGE-Reranker 重排序模型（Cross-Encoder）",
        "size": "~1.0 GB",
    },
}

SOURCES = [
    ("hf-mirror.com（国内镜像）", "https://hf-mirror.com"),
    ("huggingface.co（直连）", "https://huggingface.co"),
]


def download_model(repo_id: str, target_dir: str) -> bool:
    for name, endpoint in SOURCES:
        print(f"  尝试 {name} ...")
        os.environ["HF_ENDPOINT"] = endpoint
        try:
            os.makedirs(target_dir, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            # 验证
            if os.path.isfile(os.path.join(target_dir, "config.json")):
                print(f"  下载完成 ✓")
                return True
            print(f"  缺少 config.json，可能不完整")
        except Exception as e:
            print(f"  失败: {e}")
    return False


def main():
    for key, cfg in MODELS.items():
        print(f"\n{'='*55}")
        print(f"📦 {cfg['desc']}")
        print(f"   大小: {cfg['size']}")
        print(f"   目录: {cfg['dir']}")
        print(f"{'='*55}")

        if os.path.isfile(os.path.join(cfg["dir"], "config.json")):
            print("  已存在，跳过")
            continue

        ok = download_model(cfg["repo"], cfg["dir"])
        if not ok:
            print(f"  ✗ {key} 下载失败")
            return

    print(f"\n{'='*55}")
    print("全部下载完成！")
    print(f"  models/bge-m3/            → BGE-M3 嵌入")
    print(f"  models/bge-reranker-base/ → BGE-Reranker 重排序")
    print(f"{'='*55}")
    print("\n⚠ 模型已下载到本地，但尚未接入系统。")
    print("  当前系统仍使用 all-MiniLM-L6-v2 + 无重排序。")


if __name__ == "__main__":
    main()
