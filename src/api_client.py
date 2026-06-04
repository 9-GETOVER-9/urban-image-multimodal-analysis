# -*- coding: utf-8 -*-
"""
通义千问多模态 API 客户端。
使用 DashScope OpenAI 兼容接口调用 Qwen-VL 系列模型。
"""

import os
import base64
from pathlib import Path

# 自动加载 .env 文件（如果存在且已安装 python-dotenv）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def call_qwen_vl(image_path: str, prompt: str, api_key: str = None) -> str:
    """
    调用通义千问多模态模型进行图像分析。

    Args:
        image_path: 图片路径（本地文件）
        prompt: 分析 Prompt
        api_key: DashScope API Key，如不传则从环境变量 DASHSCOPE_API_KEY 读取

    Returns:
        模型返回的文本内容
    """
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "未找到 API Key，请设置环境变量 DASHSCOPE_API_KEY 或通过 --api-key 参数传入。"
        )

    # 将本地图片编码为 base64
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # 根据文件后缀确定 MIME 类型
    suffix = image_path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".bmp": "image/bmp"}
    mime_type = mime_map.get(suffix, "image/png")

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请先安装 openai 库: pip install openai")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    response = client.chat.completions.create(
        model="qwen-vl-max",
        messages=messages,
        temperature=0.1,
    )

    return response.choices[0].message.content
