# SciVerse Cookbook

本项目包含了一系列用于科学文献处理、分析和信息提取的 Python 脚本。这些工具利用 OpenAI 的 API 进行各种科学研究相关的自动化任务。

## 功能列表

1.  **文献综述 (1sci_review.py)**: 自动化进行文献调研与综述生成。
2.  **增强检索生成 (2sci_rag.py)**: 使用 RAG 技术针对科学文档进行智能问答。
3.  **证据提取 (3sci_evi.py)**: 从科学文本中提取关键证据与结论。
4.  **图像处理 (4sci_img.py)**: 处理科学文献中的图像。
5.  **结构化数据提取 (5sci_Stru.py)**: 将非结构化文献内容转换为结构化数据。
6.  **专利分析 (7sci_patent.py)**: 对专利文档进行自动化分析。
7.  **引用分析 (8sci_cita.py)**: 分析文献引用关系与影响力。
8.  **图表分析 (9sci_fig_analy.py)**: 对科学图表进行深度解析。

## 环境要求

- Python 3.x
- 所需依赖已列在各脚本中（建议使用虚拟环境 `venv`）

## 配置环境变量

本项目需要配置 `OPENAI_API_KEY` 和 `SCIVERSE_API_KEY`。请参考 `.env.example` 文件在项目根目录创建一个名为 `.env` 的文件，并将您的 API 密钥填入其中。

例如，您的 `.env` 文件内容可能如下：

```dotenv
OPENAI_API_KEY=sk-xxxxYou
rOpenAIKeyxxxx
SCIVERSE_API_KEY=your_sciverse_api_key_here
```

## 使用说明

1.  克隆仓库。
2.  创建并激活虚拟环境。
3.  配置 `.env` 文件。
4.  根据需求运行相应的脚本，例如：
    ```bash
    python 1sci_review.py
    ```

## 注意事项

- `figures/` 目录下的内容已被 `.gitignore` 忽略，以避免提交生成的临时图像。
- 请确保 API 密钥的安全性，不要将其硬编码在脚本中。
