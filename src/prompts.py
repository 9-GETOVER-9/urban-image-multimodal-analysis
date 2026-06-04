# -*- coding: utf-8 -*-
"""
Prompt 模板：面向通义千问多模态 API 的结构化分析 Prompt。
"""

from src.tags import DEVICE_TYPES, SCENE_TAGS, ELEMENT_TAGS


def build_analysis_prompt() -> str:
    """
    构建统一的多任务 Prompt，要求模型一次性输出设备类型、场景标签和画面要素标签的 JSON。
    """
    device_str = "、".join(DEVICE_TYPES)
    scene_str = "、".join(SCENE_TAGS)
    element_str = "、".join(ELEMENT_TAGS)

    prompt = f"""你是一位城市图像分析专家，擅长对城市监控画面进行结构化分析。
请仔细观察这张图像，并严格按照以下要求完成三项分析任务：

**任务1 - 设备类型判断：**
根据画面的拍摄角度、视野范围等信息，判断拍摄该图像的设备属于以下哪一类：{device_str}。
只能选择其中一种。

**任务2 - 场景标签识别：**
判断画面中主要展现的场景属于以下哪些类别：{scene_str}。
可以选择一个或多个。

**任务3 - 画面要素识别：**
识别画面中是否出现以下要素：{element_str}。
可以选择零个、一个或多个。

请严格按照以下JSON格式输出结果，不要输出其他任何内容：
```json
{{
    "device_type": "从设备类型中选择一种",
    "scene_tags": ["从场景标签中选择一个或多个"],
    "element_tags": ["从画面要素中选择零个、一个或多个"]
}}
```"""

    return prompt
