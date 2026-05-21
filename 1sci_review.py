import os
import sys
import httpx
import asyncio
import time
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
OPENAI_TIMEOUT = 120
BASE_URL = "https://api.sciverse.space"

async def search_literature(query: str, top_k: int = 20):
    """从Sciverse检索文献（带超时和重试机制）"""
    print(f"🔍 检索Sciverse: '{query}'...")
    
    query = query.encode('utf-8', 'ignore').decode('utf-8')
    
    timeout = Timeout(SCRAPING_TIMEOUT)
    headers = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{BASE_URL}/agentic-search",
                    headers=headers,
                    json={"query": query, "top_k": top_k}
                )
                resp.raise_for_status()
                data = resp.json()
                hits = data.get("hits", [])
                print(f"✅ 检索到 {len(hits)} 条文献片段")
                
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

async def read_context(doc_id: str, offset: int = 0, limit: int = 2000):
    """读取指定文档的原文片段"""
    headers = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}
    timeout = Timeout(SCRAPING_TIMEOUT)
    
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
                return {
                    "text": data.get("text", "").encode('utf-8', 'ignore').decode('utf-8'),
                    "next_offset": data.get("next_offset", offset),
                    "more": data.get("more", False)
                }
            except Exception as e:
                print(f"⚠️ 读取上下文失败 (尝试 {attempt+1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
    
    print(f"❌ 读取 {doc_id} 上下文失败")
    return {"text": "", "next_offset": offset, "more": False}

async def gather_evidence(hits, top_n=5):
    """收集高分证据并补充完整上下文"""
    sorted_hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:top_n]
    evidences = []
    
    print(f"📖 正在读取 {len(sorted_hits)} 篇文献的完整上下文...")
    
    for hit in sorted_hits:
        ctx = await read_context(hit["doc_id"], hit.get("offset", 0))
        evidences.append({
            "title": hit["title"],
            "doc_id": hit["doc_id"],
            "offset": hit.get("offset", 0),
            "chunk": hit["chunk"],
            "context": ctx["text"],
            "score": hit["score"]
        })
    
    print(f"✅ 成功读取 {len(evidences)} 篇文献的完整上下文")
    return evidences

async def generate_review(query: str, evidences: list):
    """使用OpenAI生成基于证据的综述（带超时和重试机制）"""
    if not evidences:
        return "⚠️ 未找到足够的相关证据"
    
    # 构建证据上下文
    evidence_text = "\n\n".join([
        f"[{e['doc_id']}, offset={e['offset']}] {e['title']}\n{e['context'][:2000]}"
        for e in evidences
    ])
    
    # 配置OpenAI客户端
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        timeout=Timeout(OPENAI_TIMEOUT)
    )
    
    # 生成提示
    system_prompt = (
        "你是一个科学研究助理，需要基于文献证据生成综述。\n"
        "回答要求：\n"
        "1. 每个论点必须标注来源 [doc_id, offset]\n"
        "2. 不要编造任何未在证据中出现的信息\n"
        "3. 证据不足时明确说明\n"
        "4. 保持专业、准确的语言风格"
    )
    user_prompt = f"问题：{query}\n\n相关证据：\n{evidence_text}"
    
    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4096
            )
            elapsed = time.time() - start_time
            print(f"✅ 生成综述成功 (耗时: {elapsed:.2f}秒)")
            return resp.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ 生成综述失败 (尝试 {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                print(f"等待 {RETRY_DELAY}秒后重试...")
                await asyncio.sleep(RETRY_DELAY)
    
    return "❌ 生成综述失败，请稍后重试或检查API配置"

async def main():
    """主执行函数"""
    start_time = time.time()
    print("="*60)
    print("🔬 科学RAG系统启动 - 基于Sciverse和GPT-4o")
    print("="*60)
    
    try:
        # 设置问题
        query = "Transformer applications in protein structure prediction 2020-2024"
        print(f"\n📝 问题: {query}")
        
        # Step 1: 检索文献
        hits = await search_literature(query, top_k=20)
        
        if not hits:
            print("❌ 未检索到任何文献")
            return
        
        # 打印检索到的片段
        print(f"\n📄 检索结果（前3条）：")
        for h in hits[:3]:
            print(f"  [{h['score']:.2f}] {h['title'][:60]}...")
        
        # Step 2: 读取完整上下文
        evidences = await gather_evidence(hits, top_n=5)
        
        if not evidences:
            print("❌ 无法读取证据上下文")
            return
        
        # Step 3: 生成综述
        review = await generate_review(query, evidences)
        
        # 打印结果
        print("\n" + "="*60)
        print("🔬 基于科学证据的结构化综述：")
        print("="*60)
        print(review.encode('utf-8', 'ignore').decode('utf-8') if isinstance(review, str) else review)
        
        # 打印引用文献
        if evidences:
            print("\n" + "="*60)
            print("📚 引用文献：")
            print("="*60)
            for i, e in enumerate(evidences):
                title = e['title'].encode('utf-8', 'ignore').decode('utf-8')
                print(f"[{i+1}] {title}")
                print(f"    Doc ID: {e['doc_id']}, Offset: {e['offset']}, 分数: {e['score']:.2f}")
    
    except Exception as e:
        print(f"❌ 发生未预期错误: {str(e)}")
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时: {elapsed:.2f}秒")

# 运行程序
if __name__ == "__main__":
    asyncio.run(main())