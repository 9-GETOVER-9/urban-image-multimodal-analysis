# -*- coding: utf-8 -*-
"""
批量推理入口：对一个目录下的所有图片进行多模态结构化分析。

用法:
    python -m src.batch_infer --input-dir examples/images --output examples/results/predictions.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

from src.prompts import build_analysis_prompt
from src.parser import parse_model_response


def infer_batch(input_dir: str, output_path: str, api_key: str = None) -> list:
    """
    批量推理。

    Args:
        input_dir: 输入图片目录
        output_path: 输出 JSONL 文件路径
        api_key: API Key（可选）

    Returns:
        所有结果的列表
    """
    from src.api_client import call_qwen_vl

    input_dir = Path(input_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        print(f"[WARN] 目录 {input_dir} 中没有找到图片文件。", file=sys.stderr)
        return []

    prompt = build_analysis_prompt()
    results = []

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, img_file in enumerate(image_files):
            print(f"[INFO] ({i+1}/{len(image_files)}) 正在分析: {img_file.name} ...")
            try:
                raw_response = call_qwen_vl(str(img_file), prompt, api_key=api_key)
                result = parse_model_response(raw_response)
                result["image"] = img_file.name
            except Exception as e:
                result = {
                    "image": img_file.name,
                    "device_type": None,
                    "scene_tags": [],
                    "element_tags": [],
                    "raw_response": "",
                    "parse_error": str(e),
                }

            results.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            # 简单限速，避免 API 频率限制
            if i < len(image_files) - 1:
                time.sleep(1)

    print(f"\n[INFO] 批量推理完成，共处理 {len(results)} 张图片。")
    print(f"[INFO] 结果已保存到: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="城市图像多模态结构化分析 - 批量推理")
    parser.add_argument("--input-dir", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--api-key", type=str, default=None, help="DashScope API Key（可选）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # 前置检查 API Key
    import os
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] 未找到 API Key，请设置环境变量 DASHSCOPE_API_KEY 或通过 --api-key 参数传入。", file=sys.stderr)
        print("[HINT] export DASHSCOPE_API_KEY=your_api_key_here", file=sys.stderr)
        sys.exit(1)

    try:
        results = infer_batch(str(input_dir), args.output, api_key=api_key)
    except ImportError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("[HINT] 请安装依赖: pip install openai", file=sys.stderr)
        sys.exit(1)

    # 打印汇总
    print("\n[SUMMARY]")
    for r in results:
        status = "OK" if r.get("parse_error") is None else "WARN"
        print(f"  [{status}] {r['image']}: {r.get('device_type', 'N/A')} | {r.get('scene_tags', [])} | {r.get('element_tags', [])}")


if __name__ == "__main__":
    main()
