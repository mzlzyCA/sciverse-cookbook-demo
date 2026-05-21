import os
import sys
import httpx
import asyncio
import time
from openai import OpenAI
from httpx import Timeout, Limits
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
OPENAI_API_KEY = _get_env_clean("OPENAI_API_KEY")
OPENAI_API_BASE = _get_env_clean("OPENAI_API_BASE")

# 修复编码问题
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 全局配置
MAX_RETRIES = 3
RETRY_DELAY = 2
SCRAPING_TIMEOUT = 30
OPENAI_TIMEOUT = 60

async def sciverse_retrieve(query: str, top_k: int = 10):
    """从Sciverse获取文献证据（带超时和重试机制）"""
    print(f"🔍 检索Sciverse: '{query}'...")
    
    # 确保使用UTF-8编码处理查询
    if not isinstance(query, bytes):
        query = query.encode('utf-8', 'ignore').decode('utf-8')
    
    timeout = Timeout(SCRAPING_TIMEOUT)
    limits = Limits(max_connections=100, max_keepalive_connections=20)
    
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    "https://api.sciverse.space/agentic-search",
                    headers={"Authorization": f"Bearer {SCIVERSE_API_KEY}"},
                    json={"query": query, "top_k": top_k, "sub_queries": 2}
                )
                resp.raise_for_status()
                data = resp.json()
                print(f"✅ 检索到 {len(data['hits'])} 条文献证据")
                
                # 确保所有文本字段使用UTF-8编码
                processed_hits = []
                for h in data["hits"]:
                    processed_hits.append({
                        "text": h.get("chunk", "").encode('utf-8', 'ignore').decode('utf-8'),
                        "doc_id": h.get("doc_id", "N/A"),
                        "title": h.get("title", "无标题").encode('utf-8', 'ignore').decode('utf-8'),
                        "score": h.get("score", 0.0)
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

def filter_and_rerank(hits, threshold=0.65):
    """过滤并排序证据"""
    filtered = [h for h in hits if h["score"] >= threshold]
    sorted_hits = sorted(filtered, key=lambda x: x["score"], reverse=True)
    print(f"📊 按分数≥{threshold}过滤后保留 {len(sorted_hits)} 条高质量证据")
    return sorted_hits

async def generate_answer(query: str, evidence: list):
    """使用GPT生成基于证据的回答（带超时和重试机制）"""
    if not evidence:
        return "⚠️ 未找到足够的相关证据"
    
    # 构建证据上下文
    context = "\n\n".join([
        f"[{i+1}] {e['title']}\n{e['text']}" 
        for i, e in enumerate(evidence[:5])
    ])
    
    # 配置OpenAI客户端
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        timeout=Timeout(OPENAI_TIMEOUT)
    )
    
    # 生成提示
    system_prompt = (
        "你是一个科学研究助理，需要基于文献证据回答问题。\n"
        "回答要求：\n"
        "1. 每句话必须标注来源编号 [1],[2]等\n"
        "2. 证据不足时明确说明\n"
        "3. 保持专业、准确的语言风格"
    )
    user_prompt = f"问题：{query}\n\n相关证据：\n{context}"
    
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
                max_tokens=1024
            )
            elapsed = time.time() - start_time
            print(f"✅ 生成回答成功 (耗时: {elapsed:.2f}秒)")
            return resp.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ 生成回答失败 (尝试 {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                print(f"等待 {RETRY_DELAY}秒后重试...")
                await asyncio.sleep(RETRY_DELAY)
    
    return "❌ 生成回答失败，请稍后重试或检查API配置"

async def main():
    """主执行函数（带超时控制）"""
    start_time = time.time()
    print("="*60)
    print("🔬 科学RAG系统启动 - 基于Sciverse和GPT-4o")
    print("="*60)
    
    try:
        # 设置问题 - 使用原始字符串避免编码问题
        query = "mRNA疫苗的脂质纳米颗粒递送系统有哪些最新改进？"
        print(f"\n📝 问题: {query}")
        
        # Step 1: 获取证据
        hits = await sciverse_retrieve(query)
        
        # Step 2: 过滤证据
        evidence = filter_and_rerank(hits) if hits else []
        
        # Step 3: 生成回答
        answer = await generate_answer(query, evidence) if evidence else "⚠️ 无可用证据"
        
        # 打印结果 - 确保使用UTF-8输出
        print("\n" + "="*60)
        print("🔬 基于科学证据的最终回答：")
        print("="*60)
        print(answer.encode('utf-8', 'ignore').decode('utf-8') if isinstance(answer, str) else answer)
        
        # 打印引用的文献
        if evidence:
            print("\n" + "="*60)
            print("📚 引用文献：")
            print("="*60)
            for i, e in enumerate(evidence[:5]):
                title = e['title'].encode('utf-8', 'ignore').decode('utf-8')
                print(f"[{i+1}] {title} (分数: {e['score']:.2f})")
    
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