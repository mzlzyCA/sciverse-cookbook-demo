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
OPENAI_TIMEOUT = 60
BASE_URL = "https://api.sciverse.space"
HEADERS = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}

async def read_full_text(doc_id: str, chunk_size: int = 4000) -> str:
    """循环读取全文，直到 more=false（带超时和重试机制）"""
    full_text = []
    offset = 0
    timeout = Timeout(SCRAPING_TIMEOUT)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await client.get(
                        f"{BASE_URL}/content",
                        headers=HEADERS,
                        params={"doc_id": doc_id, "offset": offset, "limit": chunk_size}
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("text", "").encode('utf-8', 'ignore').decode('utf-8')
                    full_text.append(text)
                    
                    if not data.get("more", False):
                        break
                    offset = data.get("next_offset", offset + len(text))
                    break
                    
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    if status_code == 404:
                        print(f"❌ 论文无全文 (404): {doc_id}")
                        return ""
                    elif status_code == 429:
                        retry_after = int(e.response.headers.get('Retry-After', RETRY_DELAY))
                        print(f"⚠️ 请求被限流，{retry_after}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                        await asyncio.sleep(retry_after)
                    elif 500 <= status_code < 600:
                        print(f"⚠️ 服务器错误({status_code})，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        print(f"❌ HTTP错误({status_code}): {str(e)}")
                        return ""
                        
                except httpx.RequestError as e:
                    print(f"⚠️ 网络错误: {str(e)}，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(RETRY_DELAY)
                    
                except Exception as e:
                    print(f"❌ 读取失败: {str(e)}")
                    return ""
            else:
                break
            
            if not data.get("more", False):
                break
    
    return "".join(full_text)

def extract_structure(full_text: str) -> str:
    """用 OpenAI 抽取论文结构化信息"""
    # 如果全文太长，取前 15000 字符
    content = full_text[:15000] if len(full_text) > 15000 else full_text
    
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        timeout=Timeout(OPENAI_TIMEOUT)
    )
    
    system_prompt = "你是一个论文阅读助手，擅长从学术论文中提取结构化信息。"
    
    user_prompt = f"""请阅读以下论文全文，提取以下四个方面的关键信息：

1. **方法**: 核心技术方法和创新点
2. **数据**: 使用的数据集、实验设置、关键数值
3. **结论**: 主要发现和贡献
4. **局限**: 已知局限和未来工作方向

论文全文:
{content}"""
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=3000
    )
    return resp.choices[0].message.content

async def main():
    start_time = time.time()
    print("="*60)
    print("🔬 Sciverse 论文阅读助手")
    print("="*60)
    
    try:
        # 先通过 agentic-search 获取真实 doc_id
        print("\n📝 步骤1: 检索论文获取 doc_id")
        timeout = Timeout(SCRAPING_TIMEOUT)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(MAX_RETRIES):
                try:
                    resp = await client.post(
                        f"{BASE_URL}/agentic-search",
                        headers=HEADERS,
                        json={"query": "AlphaFold2", "top_k": 1}
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    hits = data.get("hits", [])
                    
                    if not hits:
                        print("❌ 未找到论文")
                        return
                    
                    doc_id = hits[0]["doc_id"]
                    title = hits[0].get("title", "无标题").encode('utf-8', 'ignore').decode('utf-8')
                    print(f"✅ 找到论文: {title[:60]}...")
                    print(f"   Doc ID: {doc_id}")
                    break
                    
                except Exception as e:
                    print(f"⚠️ 检索失败 (尝试 {attempt+1}/{MAX_RETRIES}): {str(e)}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        return
        
        # Step 2: 分段读取全文
        print("\n📝 步骤2: 分段读取全文")
        text = await read_full_text(doc_id, chunk_size=4000)
        
        if not text:
            print("❌ 未能读取全文")
            return
        
        print(f"   Full text length: {len(text)} chars")
        print(f"   Preview: {text[:300]}...")
        
        # Step 3: LLM 抽取结构化信息
        print("\n📝 步骤3: 抽取结构化信息")
        report = extract_structure(text)
        
        # 输出结果
        print("\n" + "="*60)
        print("📚 论文阅读报告")
        print("="*60)
        print(report)
        
    except Exception as e:
        print(f"❌ 发生未预期错误: {str(e)}")
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时: {elapsed:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main())