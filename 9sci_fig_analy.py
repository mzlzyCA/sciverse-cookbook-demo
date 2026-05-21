import os
import sys
import re
import httpx
import asyncio
import time
import base64
from pathlib import Path
from openai import OpenAI
from httpx import Timeout
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量，覆盖已有同名环境变量以确保 `.env` 中的值生效
load_dotenv(override=True)

def _get_env_clean(key: str):
    """读取环境变量并去除可能的外层引号与不可见空白。"""
    v = os.getenv(key)
    if v is None:
        return None
    v = v.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    return v

# 从环境变量读取 API Key
SCIVERSE_API_KEY = _get_env_clean("SCIVERSE_API_KEY")
OPENAI_API_KEY = _get_env_clean("OPENAI_API_KEY")
OPENAI_API_BASE = _get_env_clean("OPENAI_API_BASE")

# 修复编码问题
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 全局配置
MAX_RETRIES = 3
RETRY_DELAY = 2
SCRAPING_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60
OPENAI_TIMEOUT = 60
BASE_URL = "https://api.sciverse.space"

async def search_papers(query: str, top_k: int = 10):
    """检索论文（带超时和重试机制）"""
    print(f"🔍 检索论文: '{query}'...")
    
    if not isinstance(query, bytes):
        query = query.encode('utf-8', 'ignore').decode('utf-8')
    
    timeout = Timeout(SCRAPING_TIMEOUT)
    headers = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{BASE_URL}/agentic-search",
                    headers=headers,
                    json={"query": query, "top_k": top_k, "sub_queries": 2}
                )
                resp.raise_for_status()
                data = resp.json()
                hits = data.get("hits", [])
                print(f"✅ 检索到 {len(hits)} 篇相关论文")
                
                processed_hits = []
                for h in hits:
                    processed_hits.append({
                        "chunk": h.get("chunk", "").encode('utf-8', 'ignore').decode('utf-8'),
                        "doc_id": h.get("doc_id", "N/A"),
                        "title": h.get("title", "无标题").encode('utf-8', 'ignore').decode('utf-8'),
                        "score": h.get("score", 0.0),
                        "offset": h.get("offset", 0)
                    })
                return processed_hits
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', RETRY_DELAY))
                    print(f"⚠️ 请求被限流，{retry_after}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(retry_after)
                elif 500 <= status_code < 600:
                    print(f"⚠️ 服务器错误({status_code})，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"❌ HTTP错误({status_code}): {str(e)}")
                    return []
                    
            except httpx.RequestError as e:
                print(f"⚠️ 网络错误: {str(e)}，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY)
                
            except Exception as e:
                print(f"❌ 检索失败: {str(e)}")
                return []
    
    print(f"❌ 达到最大重试次数({MAX_RETRIES})，检索失败")
    return []

async def get_fulltext(doc_id: str, offset: int = 0, limit: int = 4000):
    """读取文档原文（带超时和重试机制）"""
    timeout = Timeout(SCRAPING_TIMEOUT)
    headers = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(
                    f"{BASE_URL}/content",
                    headers=headers,
                    params={"doc_id": doc_id, "offset": offset, "limit": limit}
                )
                resp.raise_for_status()
                data = resp.json()
                text = data.get("text", "").encode('utf-8', 'ignore').decode('utf-8')
                return {
                    "text": text,
                    "next_offset": data.get("next_offset", offset),
                    "more": data.get("more", False)
                }
            except Exception as e:
                print(f"⚠️ 读取全文失败 (尝试 {attempt+1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
    
    return {"text": "", "next_offset": offset, "more": False}

async def get_figures_from_doc(doc_id: str, max_chars: int = 50000):
    """读取全文并提取图表路径"""
    print(f"📖 读取论文全文: {doc_id[:30]}...")
    
    # 循环读取完整文档
    full_text = ""
    offset = 0
    chunk_count = 0
    
    while len(full_text) < max_chars:
        chunk_count += 1
        result = await get_fulltext(doc_id, offset=offset, limit=4000)
        full_text += result["text"]
        
        if not result.get("more"):
            break
        
        offset = result["next_offset"]
        
        if chunk_count > 20:
            break
    
    # 提取所有图片路径
    figure_paths = re.findall(r'!\[.*?\]\((.*?)\)', full_text)
    
    # 过滤无效路径
    valid_paths = []
    for p in figure_paths:
        if len(p) < 3:
            continue
        if p.lower() in ['image', 'figure', 'fig']:
            continue
        if any(ext in p.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', 'dt=']):
            valid_paths.append(p)
        elif '/' in p or '\\' in p:
            valid_paths.append(p)
    
    print(f"   提取到 {len(valid_paths)} 个图表")
    return valid_paths

async def download_figure(file_name: str, save_dir: str = "./figures"):
    """下载资源文件（带超时和重试机制）"""
    timeout = Timeout(DOWNLOAD_TIMEOUT)
    headers = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}
    
    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(
                    f"{BASE_URL}/resource",
                    headers=headers,
                    params={"file_name": file_name}
                )
                resp.raise_for_status()
                
                # 从路径中提取文件名
                local_name = file_name.split("/")[-1]
                if not local_name or len(local_name) < 2:
                    local_name = f"figure_{hash(file_name)}.jpg"
                if '.' not in local_name:
                    local_name = local_name + '.png'
                
                save_path = f"{save_dir}/{local_name}"
                Path(save_path).write_bytes(resp.content)
                
                file_size = len(resp.content)
                size_str = f"{file_size} B"
                if file_size >= 1024:
                    size_str = f"{file_size / 1024:.2f} KB"
                
                print(f"   ✅ 下载成功: {save_path} ({size_str})")
                return save_path
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 404:
                    print(f"   ❌ 资源不存在 (404): {file_name}")
                    return None
                elif status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', RETRY_DELAY))
                    print(f"   ⚠️ 下载被限流，{retry_after}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(retry_after)
                elif 500 <= status_code < 600:
                    print(f"   ⚠️ 服务器错误({status_code})，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"   ❌ 下载失败: HTTP {status_code}")
                    return None
                    
            except httpx.RequestError as e:
                print(f"   ⚠️ 网络错误: {str(e)}，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY)
                
            except Exception as e:
                print(f"   ❌ 下载失败: {str(e)}")
                return None
    
    return None

def analyze_figure_with_openai(image_path: str, question: str) -> str:
    """用多模态 LLM 分析图表（OpenAI GPT-4o）"""
    print(f"🤖 分析图表: {os.path.basename(image_path)}...")
    
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        timeout=Timeout(OPENAI_TIMEOUT)
    )
    
    # 读取图片并转换为 base64
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    
    # 判断图片类型
    if image_path.lower().endswith('.png'):
        media_type = "image/png"
    elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
        media_type = "image/jpeg"
    elif image_path.lower().endswith('.gif'):
        media_type = "image/gif"
    elif image_path.lower().endswith('.webp'):
        media_type = "image/webp"
    else:
        media_type = "image/png"
    
    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{img_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": question
                            }
                        ]
                    }
                ],
                temperature=0.3,
                max_tokens=1024
            )
            elapsed = time.time() - start_time
            print(f"   ✅ 分析完成 (耗时: {elapsed:.2f}秒)")
            return resp.choices[0].message.content
            
        except Exception as e:
            print(f"   ⚠️ 分析失败 (尝试 {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    
    return "❌ 图表分析失败"

async def main():
    """主执行函数"""
    start_time = time.time()
    print("="*60)
    print("🔬 Sciverse 论文图表提取与分析 Demo")
    print("="*60)
    
    try:
        # Step 1: 检索论文
        query = "AlphaFold2 protein structure prediction accuracy"
        print(f"\n📝 检索查询: {query}")
        
        hits = await search_papers(query, top_k=5)
        
        if not hits:
            print("❌ 未检索到任何论文")
            return
        
        # 显示检索结果
        print(f"\n📄 检索结果：")
        for i, h in enumerate(hits[:3]):
            print(f"  [{i+1}] 分数: {h['score']:.3f} | {h['title'][:60]}...")
            print(f"      Doc ID: {h['doc_id']}")
        
        # Step 2: 选择最高分论文提取图表
        best_paper = hits[0]
        print(f"\n📖 选择最高分论文提取图表:")
        print(f"  标题: {best_paper['title']}")
        print(f"  Doc ID: {best_paper['doc_id']}")
        
        figure_paths = await get_figures_from_doc(best_paper['doc_id'])
        
        if not figure_paths:
            print("❌ 未找到图表")
            return
        
        # 显示图表路径
        print(f"\n📊 找到的图表路径（前10个）:")
        for i, p in enumerate(figure_paths[:10]):
            print(f"  [{i+1}] {p[:80]}...")
        
        # Step 3: 下载图表
        print(f"\n📥 下载图表...")
        save_dir = f"./figures/{best_paper['doc_id'][:20]}"
        
        downloaded = []
        for i, path in enumerate(figure_paths[:5]):  # 只下载前5个
            print(f"  [{i+1}/{min(5, len(figure_paths))}] 下载: {path.split('/')[-1][:50]}...")
            saved = await download_figure(path, save_dir)
            if saved:
                downloaded.append(saved)
            await asyncio.sleep(0.5)
        
        print(f"\n✅ 成功下载 {len(downloaded)} 个图表")
        
        # Step 4: 分析图表
        if downloaded and OPENAI_API_KEY:
            print("\n" + "="*60)
            print("🔬 图表分析结果")
            print("="*60)
            
            for i, img_path in enumerate(downloaded[:3]):  # 只分析前3个
                print(f"\n📊 图表 {i+1}: {os.path.basename(img_path)}")
                print("-"*40)
                
                analysis = analyze_figure_with_openai(
                    img_path,
                    "请分析这张图表的主要发现，提取关键数值和趋势。如果是精度对比图，请找出具体的数值指标。"
                )
                print(f"\n{analysis}")
                print("-"*40)
        elif downloaded and not OPENAI_API_KEY:
            print("\n⚠️ OPENAI_API_KEY 未配置，跳过图表分析")
        else:
            print("\n⚠️ 无图表可分析")
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 执行摘要")
        print("="*60)
        print(f"  检索论文: {len(hits)} 篇")
        print(f"  提取图表: {len(figure_paths)} 个")
        print(f"  下载图表: {len(downloaded)} 个")
        print(f"  分析图表: {min(len(downloaded), 3) if downloaded else 0} 个")
        
    except Exception as e:
        print(f"❌ 发生未预期错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时: {elapsed:.2f}秒")

# 运行程序（带3分钟超时）
if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=180))
    except asyncio.TimeoutError:
        print("❌ 脚本执行超时（超过3分钟）")