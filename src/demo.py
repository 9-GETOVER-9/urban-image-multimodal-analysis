# -*- coding: utf-8 -*-
"""
离线 Demo 脚本：无需 API Key 也能展示项目效果。
读取预先准备的样例输出，展示结构化分析结果。

用法:
    python -m src.demo
    python -m src.demo --sample-file examples/results/sample_predictions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_SAMPLE_FILE = "examples/results/sample_predictions.jsonl"


def load_sample_results(sample_file: str) -> list:
    """加载样例输出文件。"""
    results = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def display_results(results: list):
    """在终端美观地展示分析结果。"""
    print("=" * 70)
    print("  城市图像多模态结构化分析 - Demo 展示")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        print(f"\n{'─' * 70}")
        print(f"  📷 图片 [{i}/{len(results)}]: {r.get('image', 'N/A')}")
        print(f"{'─' * 70}")
        print(f"  🔧 设备类型: {r.get('device_type', 'N/A')}")

        scene_tags = r.get("scene_tags", [])
        print(f"  🏙️  场景标签: {', '.join(scene_tags) if scene_tags else '无'}")

        element_tags = r.get("element_tags", [])
        print(f"  🏷️  画面要素: {', '.join(element_tags) if element_tags else '无'}")

        if r.get("parse_error"):
            print(f"  ⚠️  解析警告: {r['parse_error']}")

    print(f"\n{'=' * 70}")
    print(f"  共展示 {len(results)} 条分析结果")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="城市图像多模态结构化分析 - 离线 Demo")
    parser.add_argument(
        "--sample-file",
        type=str,
        default=DEFAULT_SAMPLE_FILE,
        help="样例输出文件路径（默认: examples/results/sample_predictions.jsonl）",
    )
    args = parser.parse_args()

    sample_file = Path(args.sample_file)
    if not sample_file.exists():
        print(f"[ERROR] 样例文件不存在: {sample_file}", file=sys.stderr)
        sys.exit(1)

    results = load_sample_results(str(sample_file))
    display_results(results)


if __name__ == "__main__":
    main()
