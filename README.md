# 🏙️ 城市图像多模态结构化分析

> 基于通义千问多模态大模型（Qwen-VL）的城市街景图像结构化分析系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 项目简介

本项目使用通义千问多模态大模型（Qwen-VL）对城市街景图像进行自动化结构化分析，实现：

- **设备类型识别**：判断图像由哪种摄像头拍摄（车载摄像头、固定摄像头、无人机、全景云台）
- **场景标签分类**：识别图像展现的主要场景（工厂、沿街店铺、公路、路口、停车场、小区、巷弄）
- **画面要素检测**：提取画面中的关键要素（禁停区域、停车位、消防通道标识、垃圾桶）

### 🎯 核心亮点

1. **单模型多任务**：使用一个 VLM 模型同时处理三类打标任务，通过精心设计的 Prompt 实现多任务联合推理
2. **工程闭环**：从 Prompt 设计 → API 调用 → 结果解析 → 标签校验，形成完整的工程链路
3. **低门槛运行**：无需 GPU，只需一个 API Key 即可运行；无 API Key 时可运行离线 Demo
4. **可扩展设计**：标签体系可配置，Prompt 模板可替换，支持批量推理

## 🏗️ 项目结构

```
├── README.md                    # 项目说明文档
├── LICENSE                      # MIT 开源协议
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量示例
├── .gitignore                   # Git 忽略规则
│
├── src/                         # 核心代码
│   ├── tags.py                  # 标签体系定义
│   ├── prompts.py               # Prompt 模板
│   ├── api_client.py            # 通义千问 API 客户端
│   ├── parser.py                # 结果解析与标签校验
│   ├── infer.py                 # 单图推理入口
│   ├── batch_infer.py           # 批量推理入口
│   └── demo.py                  # 离线 Demo 展示
│
├── examples/                    # 可运行的样例数据（随仓库上传）
│   ├── images/                  # 6 张样例图片（CityScapes 子集）
│   └── results/
│       └── sample_predictions.jsonl  # 离线样例输出
│
├── docs/                        # 项目文档
│   ├── proposal.md              # 项目方案文档（任务定义、标注流程、技术方案）
│   └── qwen-vlm.md              # Qwen-VL 微调指南（架构、训练、推理）
│
└── imgs/                        # 文档配图素材（仅 docs/ 引用）
```

> **数据说明**：完整 CityScapes 原始数据集（约 176MB）不随仓库上传，仅保留 `examples/images/` 中的 6 张小样例用于快速体验。

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目（发布后替换为真实仓库地址）
# git clone https://github.com/your-username/urban-image-multimodal-analysis.git
# cd urban-image-multimodal-analysis

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 方式1：设置环境变量（推荐）
export DASHSCOPE_API_KEY=your_api_key_here

# 方式2：复制 .env.example 为 .env 并填写
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

> 💡 **获取 API Key**：访问 [通义千问控制台](https://dashscope.console.aliyun.com/) 注册并获取

### 3. 运行推理

#### 离线 Demo（无需 API Key，快速体验）

```bash
python -m src.demo
```

#### 单图推理

```bash
python -m src.infer --image examples/images/000000000000.png
```

输出示例：
```json
{
  "image": "000000000000.png",
  "device_type": "无人机",
  "scene_tags": ["小区"],
  "element_tags": ["消防通道标识"],
  "raw_response": "...",
  "parse_error": null
}
```

#### 批量推理

```bash
python -m src.batch_infer --input-dir examples/images --output examples/results/predictions.jsonl
```

## 🔧 标签体系

| 类别 | 标签选项 | 选择规则 |
|------|----------|----------|
| 设备类型 | 车载摄像头、固定摄像头、无人机、全景云台 | 单选 |
| 场景标签 | 工厂、沿街店铺、公路、路口、停车场、小区、巷弄 | 多选 |
| 画面要素 | 禁停区域、停车位、消防通道标识、垃圾桶 | 多选（可为空） |

## 💡 技术方案

### Prompt 设计策略

采用**单次推理多任务**策略，通过一个 Prompt 同时完成三类标签的识别：

```
你是一位城市图像分析专家...
请完成三项分析任务：
1. 设备类型判断（单选）
2. 场景标签识别（多选）
3. 画面要素识别（多选）

请严格按照 JSON 格式输出...
```

**优势**：
- 单次 API 调用完成所有任务，降低推理成本
- 模型能在任务间建立语义关联（如"停车场"与"停车位"的关联）
- 输出结构化 JSON，便于后续处理

### 微调方案（进阶）

本项目还提供了基于 Qwen-VL 的微调方案，详见 [docs/qwen-vlm.md](docs/qwen-vlm.md)：

- **多轮对话合并**：将三个任务合并为多轮对话训练样本
- **Seq2Set 策略**：穷举多标签的排列组合，避免标签依赖固化
- **LoRA 微调**：单卡 4090 即可完成微调

## 📊 效果展示

| 图片 | 设备类型 | 场景标签 | 画面要素 |
|------|----------|----------|----------|
| 000000000000.png | 无人机 | 小区 | 消防通道标识 |
| 000000000001.png | 车载摄像头 | 沿街店铺 | - |
| 000000000003.png | 车载摄像头 | 公路 | - |
| 000000000004.png | 无人机 | 巷弄 | - |

> 以上为离线样例输出，可通过 `python -m src.demo` 查看完整展示。

## 🎓 面试讲解口径

### Q: 为什么选择多模态大模型而不是传统小模型？

**A**: 传统方案需要为每个标签类别训练独立模型，当标签体系扩展时成本剧增。多模态大模型的优势在于：
1. **泛化能力强**：一个模型处理所有标签任务
2. **标注成本低**：只需少量样本即可达到较好效果
3. **扩展灵活**：新增标签只需修改 Prompt，无需重新训练

### Q: 如何处理多标签任务中的标签依赖问题？

**A**: 使用 **Seq2Set** 策略：
- 在训练数据准备阶段，穷举多标签的所有排列组合
- 例如 3 个标签生成 6 条训练样本
- 避免模型学习到错误的标签顺序依赖

### Q: 这个项目的核心工程难点是什么？

**A**:
1. **Prompt 工程**：设计能同时完成多任务的结构化 Prompt
2. **结果解析**：处理模型输出的非标准 JSON（如 markdown 包裹、多余文本）
3. **标签校验**：确保输出标签在合法范围内

## 📚 相关文档

- [项目方案文档](docs/proposal.md)：详细的任务定义、数据准备、标注流程、技术方案
- [Qwen-VL 微调指南](docs/qwen-vlm.md)：模型架构、训练流程、LoRA 微调实践

## 📄 License

本项目采用 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

- [通义千问](https://qianwen.aliyun.com/)：提供多模态大模型 API
- [CityScapes](https://www.cityscapes-dataset.com/)：提供城市街景数据集
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)：开源多模态大模型
