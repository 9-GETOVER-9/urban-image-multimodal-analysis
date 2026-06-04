# -*- coding: utf-8 -*-
"""
单图推理入口：对一张城市街景图像进行多模态结构化分析。

用法:
    python -m src.infer --image examples/images/000000000000.png
    python -m src.infer --image examples/images/000000000000.png --output result.json
"""

import argparse
import json
import sys
from pathlib import Path

from src.prompts import build_analysis_prompt
from src.parser import parse_model_response


def infer_single_image(image_path: str, api_key: str = None) -> dict:
    """
    对单张图片执行多模态结构化分析。

    Args:
        image_path: 图片路径
        api_key: API Key（可选）

    Returns:
        结构化分析结果字典
    """
    from src.api_client import call_qwen_vl

    prompt = build_analysis_prompt()
    raw_response = call_qwen_vl(image_path, prompt, api_key=api_key)
    result = parse_model_response(raw_response)
    result["image"] = Path(image_path).name
    return result


def main():
    parser = argparse.ArgumentParser(description="城市图像多模态结构化分析 - 单图推理")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径（可选）")
    parser.add_argument("--api-key", type=str, default=None, help="DashScope API Key（可选，默认读取环境变量）")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] 图片文件不存在: {image_path}", file=sys.stderr)
        sys.exit(1)

    # 前置检查 API Key
    import os
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[ERROR] 未找到 API Key，请设置环境变量 DASHSCOPE_API_KEY 或通过 --api-key 参数传入。", file=sys.stderr)
        print("[HINT] export DASHSCOPE_API_KEY=your_api_key_here", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 正在分析图片: {image_path.name} ...")
    try:
        result = infer_single_image(str(image_path), api_key=api_key)
    except ImportError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("[HINT] 请安装依赖: pip install openai", file=sys.stderr)
        sys.exit(1)

    # 输出结果
    output_json = json.dumps(result, ensure_ascii=False, indent=2)
    print("\n[RESULT]")
    print(output_json)

    # 保存到文件
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\n[INFO] 结果已保存到: {output_path}")

    return result


if __name__ == "__main__":
    main()
