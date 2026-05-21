import os
import sys
import httpx
import asyncio
import time
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
BASE_URL = "https://api.sciverse.space"

async def search_literature(query: str, top_k: int = 10):
    """从Sciverse获取文献证据（带超时和重试机制）"""
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

async def get_fulltext(doc_id: str, offset: int = 0, limit: int = 2000):
    """读取文档原文。返回 {text, next_offset, more}（带超时和重试机制）"""
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
                    print(f"⚠️ 读取全文被限流，{retry_after}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(retry_after)
                elif 500 <= status_code < 600:
                    print(f"⚠️ 服务器错误({status_code})，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(f"❌ 读取全文HTTP错误({status_code}): {str(e)}")
                    return {"text": "", "next_offset": offset, "more": False}
                    
            except httpx.RequestError as e:
                print(f"⚠️ 读取全文网络错误: {str(e)}，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY)
                
            except Exception as e:
                print(f"❌ 读取全文失败: {str(e)}")
                return {"text": "", "next_offset": offset, "more": False}
    
    print(f"❌ 达到最大重试次数({MAX_RETRIES})，读取全文失败")
    return {"text": "", "next_offset": offset, "more": False}

async def read_full_document(doc_id: str, max_chars: int = 16000):
    """循环读取直到全文或达到字符上限"""
    print(f"📚 开始读取完整文档: {doc_id}")
    full_text = ""
    offset = 0
    chunk_count = 0
    
    while len(full_text) < max_chars:
        chunk_count += 1
        print(f"  读取第 {chunk_count} 块 (offset={offset}, limit=4000)...")
        
        result = await get_fulltext(doc_id, offset=offset, limit=4000)
        full_text += result["text"]
        
        if not result.get("more"):
            print(f"  ✅ 文档读取完成，共 {len(full_text)} 字符")
            break
        
        offset = result["next_offset"]
    
    if len(full_text) >= max_chars:
        print(f"  ⚠️ 已达到字符上限 {max_chars}，文档可能未完全读取")
    
    return full_text

async def main():
    """主执行函数（带超时控制）"""
    start_time = time.time()
    print("="*60)
    print("🔬 Sciverse 全文证据检索系统")
    print("="*60)
    
    try:
        # Step 1: 检索文献片段
        query = "AlphaFold2 protein structure prediction deep learning"
        print(f"\n📝 检索查询: {query}")
        
        hits = await search_literature(query, top_k=10)
        
        if not hits:
            print("❌ 未检索到任何文献")
            return
        
        # 显示检索结果
        print(f"\n📄 检索结果（按分数排序）：")
        for i, h in enumerate(hits[:5]):
            print(f"  [{i+1}] 分数: {h['score']:.3f} | {h['title'][:60]}...")
            print(f"      Doc ID: {h['doc_id']}, Offset: {h['offset']}")
            print(f"      片段: {h['chunk'][:100]}...")
        
        # Step 2: 选择最高分的片段读取完整上下文
        best_hit = hits[0]
        print(f"\n📖 选择最高分片段读取完整上下文:")
        print(f"  标题: {best_hit['title']}")
        print(f"  Doc ID: {best_hit['doc_id']}, 原始偏移: {best_hit['offset']}, 分数: {best_hit['score']:.3f}")
        
        # 向前偏移 300 字符以获取前文语境
        start_offset = max(0, best_hit['offset'] - 300)
        print(f"  向前偏移 300 字符，起始偏移: {start_offset}")
        
        result = await get_fulltext(best_hit['doc_id'], offset=start_offset, limit=2000)
        
        # 显示读取结果
        print(f"\n✅ 成功读取完整上下文:")
        print(f"  文本长度: {len(result['text'])} 字符")
        print(f"  还有更多内容: {'是' if result['more'] else '否'}")
        if result.get('next_offset'):
            print(f"  下一个偏移: {result['next_offset']}")
        
        print(f"\n📄 完整上下文预览（前500字符）:")
        print("-"*60)
        print(result['text'][:500])
        print("-"*60)
        
        # Step 3: 可选 - 读取完整文档
        print(f"\n💡 是否读取完整文档？(y/n) 如需完整读取，请修改代码中的 read_full 变量")
        read_full = False  # 改为 True 可读取完整文档
        
        if read_full:
            print(f"\n📚 开始读取完整文档...")
            full_text = await read_full_document(best_hit['doc_id'], max_chars=16000)
            print(f"\n✅ 完整文档读取完成，总长度: {len(full_text)} 字符")
            print(f"\n文档开头部分:")
            print("-"*60)
            print(full_text[:800])
            print("-"*60)
        
    except asyncio.TimeoutError:
        print("❌ 操作超时，请检查网络连接或API响应时间")
    except Exception as e:
        print(f"❌ 发生未预期错误: {str(e)}")
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时: {elapsed:.2f}秒")

# 运行程序（带2分钟超时）
if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=120))
    except asyncio.TimeoutError:
        print("❌ 脚本执行超时（超过2分钟）")