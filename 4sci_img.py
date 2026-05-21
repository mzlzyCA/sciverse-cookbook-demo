import os
import sys
import re
import httpx
import asyncio
import time
from pathlib import Path
from httpx import Timeout
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量，覆盖已有同名环境变量以确保 `.env` 中的值生效
load_dotenv(override=True)

def _get_env_clean(key: str):
    """读取环境变量并去除可能的外层引号与不可见空白。"""
    v = os.getenv(key)
    if v is None:
        return None
    v = v.strip()  # 去掉前后空白与换行
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    return v

# 从环境变量读取 API Key（已在 .env 文件中定义），并清洗可能的引号或空白
SCIVERSE_API_KEY = _get_env_clean("SCIVERSE_API_KEY")

# 修复编码问题
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 全局配置
MAX_RETRIES = 3
RETRY_DELAY = 2
SCRAPING_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60
BASE_URL = "https://api.sciverse.space"

async def search_literature(query: str, top_k: int = 10):
    """从Sciverse检索文献（带超时和重试机制）"""
    print(f"🔍 检索Sciverse: '{query}'...")
    
    # 确保使用UTF-8编码处理查询
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
                print(f"✅ 检索到 {len(hits)} 条文献片段")
                
                # 确保所有文本字段使用UTF-8编码
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
                if status_code == 429:  # 限流
                    retry_after = int(e.response.headers.get('Retry-After', RETRY_DELAY))
                    print(f"⚠️ 请求被限流，{retry_after}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(retry_after)
                elif 500 <= status_code < 600:  # 服务器错误
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

async def get_content(doc_id: str, offset: int = 0, limit: int = 4000):
    """获取文档内容（Markdown格式）（带超时和重试机制）"""
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
                
                # 确保文本使用UTF-8编码
                text = data.get("text", "").encode('utf-8', 'ignore').decode('utf-8')
                
                return {
                    "text": text,
                    "next_offset": data.get("next_offset", offset),
                    "more": data.get("more", False)
                }
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', RETRY_DELAY))
                    print(f"⚠️ 获取内容被限流，{retry_after}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(retry_after)
                elif 500 <= status_code < 600:
                    print(f"⚠️ 服务器错误({status_code})，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"❌ 获取内容HTTP错误({status_code}): {str(e)}")
                    return {"text": "", "next_offset": offset, "more": False}
                    
            except httpx.RequestError as e:
                print(f"⚠️ 获取内容网络错误: {str(e)}，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY)
                
            except Exception as e:
                print(f"❌ 获取内容失败: {str(e)}")
                return {"text": "", "next_offset": offset, "more": False}
    
    print(f"❌ 达到最大重试次数({MAX_RETRIES})，获取内容失败")
    return {"text": "", "next_offset": offset, "more": False}

def extract_figure_paths(markdown_text: str):
    """从Markdown文本中提取所有图片路径，过滤无效路径"""
    # 匹配标准的 Markdown 图片语法: ![alt text](path)
    # 路径不能只是 "image" 这样的纯文本
    figure_paths = re.findall(r'!\[.*?\]\((.*?)\)', markdown_text)
    
    # 过滤无效路径（长度小于3、只包含"image"、不包含文件扩展名）
    valid_paths = []
    for p in figure_paths:
        # 跳过明显的无效路径
        if len(p) < 3:
            continue
        if p.lower() in ['image', 'figure', 'fig']:
            continue
        # 检查是否包含常见图片扩展名或包含路径分隔符
        if any(ext in p.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', 'dt=']):
            valid_paths.append(p)
        elif '/' in p or '\\' in p:
            # 包含路径分隔符的也可能是有效路径
            valid_paths.append(p)
    
    print(f"  原始提取: {len(figure_paths)} 个引用，过滤后: {len(valid_paths)} 个有效路径")
    return valid_paths

async def download_resource(file_name: str, save_dir: str = "./figures"):
    """下载资源文件。参数 file_name 为 content 中提取的相对路径（带超时和重试机制）"""
    timeout = Timeout(DOWNLOAD_TIMEOUT)
    headers = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}
    
    # 创建保存目录（确保父目录存在）
    save_path_obj = Path(save_dir)
    save_path_obj.mkdir(parents=True, exist_ok=True)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                # 注意：file_name 可能包含路径，直接作为参数传递
                resp = await client.get(
                    f"{BASE_URL}/resource",
                    headers=headers,
                    params={"file_name": file_name}
                )
                resp.raise_for_status()
                
                # 从路径中提取文件名（处理各种路径格式）
                if '/' in file_name:
                    local_name = file_name.split('/')[-1]
                elif '\\' in file_name:
                    local_name = file_name.split('\\')[-1]
                else:
                    local_name = file_name
                
                # 确保文件名有扩展名，如果没有则添加 .jpg
                if '.' not in local_name:
                    local_name = local_name + '.jpg'
                
                save_path = save_path_obj / local_name
                
                # 保存二进制内容
                save_path.write_bytes(resp.content)
                
                file_size = len(resp.content)
                size_str = f"{file_size} B"
                if file_size >= 1024:
                    size_str = f"{file_size / 1024:.2f} KB"
                if file_size >= 1024 * 1024:
                    size_str = f"{file_size / (1024 * 1024):.2f} MB"
                
                print(f"  ✅ 保存成功: {save_path} ({size_str})")
                return str(save_path)
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 404:
                    print(f"  ❌ 资源不存在 (404): {file_name}")
                    return None
                elif status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', RETRY_DELAY))
                    print(f"  ⚠️ 下载被限流，{retry_after}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(retry_after)
                elif 500 <= status_code < 600:
                    print(f"  ⚠️ 服务器错误({status_code})，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"  ❌ 下载失败 {file_name}: HTTP {status_code}")
                    return None
                    
            except httpx.RequestError as e:
                print(f"  ⚠️ 下载网络错误: {str(e)}，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY)
                
            except Exception as e:
                print(f"  ❌ 下载失败 {file_name}: {str(e)}")
                return None
    
    print(f"  ❌ 达到最大重试次数({MAX_RETRIES})，下载失败: {file_name}")
    return None

async def download_all_resources(file_paths: list, save_dir: str = "./figures"):
    """批量下载所有资源文件"""
    if not file_paths:
        print("⚠️ 没有需要下载的文件")
        return []
    
    print(f"\n📥 开始下载 {len(file_paths)} 个资源文件...")
    results = []
    
    for i, file_path in enumerate(file_paths):
        print(f"  [{i+1}/{len(file_paths)}] 下载: {file_path[:80]}..." if len(file_path) > 80 else f"  [{i+1}/{len(file_paths)}] 下载: {file_path}")
        saved_path = await download_resource(file_path, save_dir)
        if saved_path:
            results.append(saved_path)
        
        # 避免请求过快
        if i < len(file_paths) - 1:
            await asyncio.sleep(0.5)
    
    print(f"\n✅ 下载完成: 成功 {len(results)}/{len(file_paths)} 个文件")
    return results

async def read_full_document_for_figures(doc_id: str, max_chars: int = 100000):
    """循环读取完整文档以提取所有图表（带超时和重试机制）"""
    print(f"📚 开始读取完整文档以提取图表: {doc_id}")
    full_text = ""
    offset = 0
    chunk_count = 0
    
    while len(full_text) < max_chars:
        chunk_count += 1
        print(f"  读取第 {chunk_count} 块 (offset={offset}, limit=4000)...")
        
        result = await get_content(doc_id, offset=offset, limit=4000)
        full_text += result["text"]
        
        if not result.get("more"):
            print(f"  ✅ 文档读取完成，共 {len(full_text)} 字符")
            break
        
        offset = result["next_offset"]
        
        # 防止无限循环
        if chunk_count > 50:
            print(f"  ⚠️ 达到最大块数限制，停止读取")
            break
    
    if len(full_text) >= max_chars:
        print(f"  ⚠️ 已达到字符上限 {max_chars}，文档可能未完全读取")
    
    return full_text

def filter_figure_paths(paths: list, extensions: list = None):
    """过滤特定格式的图片路径"""
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp']
    
    filtered = []
    for p in paths:
        p_lower = p.lower()
        # 检查文件扩展名或路径中包含图片标识
        if any(p_lower.endswith(ext) for ext in extensions):
            filtered.append(p)
        elif any(ext in p_lower for ext in extensions):
            # 路径中包含扩展名但不以扩展名结尾
            filtered.append(p)
    
    print(f"📊 过滤后保留 {len(filtered)} 个图片文件（支持格式: {', '.join(extensions)}）")
    return filtered

async def main():
    """主执行函数（带超时控制）"""
    start_time = time.time()
    print("="*60)
    print("🔬 Sciverse 论文图表下载系统")
    print("="*60)
    
    try:
        # Step 1: 检索文献
        query = "AlphaFold2 protein structure prediction deep learning"
        print(f"\n📝 检索查询: {query}")
        
        hits = await search_literature(query, top_k=5)
        
        if not hits:
            print("❌ 未检索到任何文献")
            return
        
        # 显示检索结果
        print(f"\n📄 检索结果：")
        for i, h in enumerate(hits[:3]):
            print(f"  [{i+1}] 分数: {h['score']:.3f} | {h['title'][:60]}...")
            print(f"      Doc ID: {h['doc_id']}")
        
        # Step 2: 选择最高分的文档
        best_hit = hits[0]
        # 清理 doc_id，移除可能导致目录创建失败的字符
        safe_doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', best_hit['doc_id'])
        
        print(f"\n📖 选择最高分文档提取图表:")
        print(f"  标题: {best_hit['title']}")
        print(f"  原始 Doc ID: {best_hit['doc_id']}")
        print(f"  安全 Doc ID: {safe_doc_id}")
        
        # Step 3: 获取完整文档内容
        print(f"\n📥 获取文档全文...")
        full_text = await read_full_document_for_figures(best_hit['doc_id'], max_chars=100000)
        
        if not full_text:
            print("❌ 未能获取文档内容")
            return
        
        # Step 4: 提取图表路径
        print(f"\n🔍 从Markdown中提取图片路径...")
        figure_paths = extract_figure_paths(full_text)
        
        # 显示前10个路径示例
        if figure_paths:
            print(f"\n  图片路径示例（前10个）:")
            for i, p in enumerate(figure_paths[:10]):
                print(f"    [{i+1}] {p[:80]}..." if len(p) > 80 else f"    [{i+1}] {p}")
        else:
            print("  ⚠️ 未提取到任何图片路径")
        
        if not figure_paths:
            print("⚠️ 未找到有效的图表文件")
            return
        
        # Step 5: 过滤有效图片格式
        figure_paths = filter_figure_paths(figure_paths)
        
        if not figure_paths:
            print("⚠️ 过滤后没有有效的图片文件")
            return
        
        # Step 6: 下载所有图表（使用安全的目录名）
        save_dir = f"./figures/{safe_doc_id}"
        print(f"\n💾 保存目录: {save_dir}")
        
        saved_files = await download_all_resources(figure_paths, save_dir)
        
        # Step 7: 显示下载结果统计
        print("\n" + "="*60)
        print("📊 下载结果统计：")
        print("="*60)
        print(f"  文档ID: {best_hit['doc_id']}")
        print(f"  标题: {best_hit['title'][:80]}...")
        print(f"  发现图片: {len(figure_paths)} 个")
        print(f"  成功下载: {len(saved_files)} 个")
        print(f"  保存目录: {save_dir}")
        
        if saved_files:
            print(f"\n📁 下载的文件列表：")
            for i, f in enumerate(saved_files[:10]):
                if os.path.exists(f):
                    file_size = os.path.getsize(f)
                    if file_size < 1024:
                        size_str = f"{file_size} B"
                    elif file_size < 1024 * 1024:
                        size_str = f"{file_size / 1024:.2f} KB"
                    else:
                        size_str = f"{file_size / (1024 * 1024):.2f} MB"
                    print(f"  [{i+1}] {os.path.basename(f)} ({size_str})")
            
            if len(saved_files) > 10:
                print(f"  ... 还有 {len(saved_files) - 10} 个文件")
        
    except asyncio.TimeoutError:
        print("❌ 操作超时，请检查网络连接或API响应时间")
    except Exception as e:
        print(f"❌ 发生未预期错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时: {elapsed:.2f}秒")

# 运行程序（带3分钟超时，因为下载图片可能需要更长时间）
if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=180))
    except asyncio.TimeoutError:
        print("❌ 脚本执行超时（超过3分钟）")



        