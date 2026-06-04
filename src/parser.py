# -*- coding: utf-8 -*-
"""
解析模型返回的 JSON 结果，进行标签校验和容错处理。
"""

import json
import re
from typing import Dict, List, Optional

from src.tags import DEVICE_TYPES, SCENE_TAGS, ELEMENT_TAGS


def parse_model_response(raw_response: str) -> Dict:
    """
    解析模型返回的文本，提取 JSON 结构化结果。

    Args:
        raw_response: 模型原始输出文本

    Returns:
        解析后的结构化字典，包含:
        - device_type: 设备类型
        - scene_tags: 场景标签列表
        - element_tags: 画面要素标签列表
        - raw_response: 原始响应
        - parse_error: 解析错误信息（如果有）
    """
    result = {
        "device_type": None,
        "scene_tags": [],
        "element_tags": [],
        "raw_response": raw_response,
        "parse_error": None,
    }

    # 尝试从响应中提取 JSON
    json_obj = _extract_json(raw_response)
    if json_obj is None:
        result["parse_error"] = "无法从模型输出中提取有效的 JSON 格式。"
        return result

    # 解析设备类型
    device_type = json_obj.get("device_type", "")
    if device_type in DEVICE_TYPES:
        result["device_type"] = device_type
    else:
        result["parse_error"] = f"设备类型 '{device_type}' 不在允许的标签列表中。"
        result["device_type"] = device_type  # 仍然保留原始值

    # 解析场景标签
    scene_tags = json_obj.get("scene_tags", [])
    if isinstance(scene_tags, str):
        scene_tags = [s.strip() for s in scene_tags.split("、") if s.strip()]
    result["scene_tags"] = _validate_tags(scene_tags, SCENE_TAGS, "scene_tags")

    # 解析画面要素
    element_tags = json_obj.get("element_tags", [])
    if isinstance(element_tags, str):
        element_tags = [s.strip() for s in element_tags.split("、") if s.strip()]
    result["element_tags"] = _validate_tags(element_tags, ELEMENT_TAGS, "element_tags")

    return result


def _extract_json(text: str) -> Optional[Dict]:
    """从文本中提取 JSON 对象。"""
    # 方法1：尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 方法2：从 markdown 代码块中提取
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 方法3：查找第一个 { 到最后一个 } 之间的内容
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _validate_tags(tags: List[str], valid_tags: List[str], tag_type: str) -> List[str]:
    """校验标签是否在允许的范围内，过滤掉非法标签。"""
    valid_set = set(valid_tags)
    filtered = []
    for tag in tags:
        if isinstance(tag, str) and tag.strip() in valid_set:
            filtered.append(tag.strip())
    return filtered
