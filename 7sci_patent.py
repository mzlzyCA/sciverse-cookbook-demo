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

async def search(query: str, top_k: int = 15):
    """语义检索（带超时和重试机制）"""
    print(f"🔍 检索: '{query[:60]}...'")
    
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
                print(f"✅ 检索到 {len(hits)} 条片段")
                
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

async def generate_analysis(patent_hits: list, academic_hits: list):
    """使用OpenAI生成关联分析报告（带超时和重试机制）"""
    
    # 准备摘要数据
    patent_summary = []
    for h in patent_hits[:8]:
        patent_summary.append(f"- [{h['doc_id']}] (score: {h['score']:.2f}) {h['title']}: {h['chunk'][:80]}...")
    
    academic_summary = []
    for h in academic_hits[:8]:
        academic_summary.append(f"- [{h['doc_id']}] (score: {h['score']:.2f}) {h['title']}: {h['chunk'][:80]}...")
    
    # 配置OpenAI客户端
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        timeout=Timeout(OPENAI_TIMEOUT)
    )
    
    system_prompt = (
        "你是一个专利与文献探索分析专家。基于提供的检索结果进行技术关联分析。\n"
        "要求：\n"
        "1. 所有结论必须基于检索结果\n"
        "2. 标注所有来源的 doc_id\n"
        "3. 证据不足时明确说明\n"
        "4. 保持专业、客观的语言风格"
    )
    
    user_prompt = f"""分析以下两组检索结果的技术关联：

## 专利相关片段
{chr(10).join(patent_summary) if patent_summary else "无专利相关片段"}

## 学术文献片段
{chr(10).join(academic_summary) if academic_summary else "无学术文献片段"}

请输出：
1) 两组结果中的技术主题对比
2) 可能的专利-论文关联（基于内容相似性）
3) 技术发展脉络推测"""
    
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
            print(f"✅ 生成分析报告成功 (耗时: {elapsed:.2f}秒)")
            return resp.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ 生成分析失败 (尝试 {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                print(f"等待 {RETRY_DELAY}秒后重试...")
                await asyncio.sleep(RETRY_DELAY)
    
    return "❌ 生成分析报告失败，请稍后重试或检查API配置"

async def main():
    """主执行函数"""
    start_time = time.time()
    print("="*60)
    print("🔬 Sciverse 专利与文献语义探索系统")
    print("="*60)
    
    try:
        # Step 1: 检索专利相关内容
        print("\n📊 步骤1: 检索专利相关内容")
        patent_hits = await search("CRISPR base editing patent method composition", top_k=15)
        
        # Step 2: 检索学术文献
        print("\n📊 步骤2: 检索学术文献")
        academic_hits = await search("CRISPR base editing adenine cytosine mechanism", top_k=15)
        
        if not patent_hits and not academic_hits:
            print("❌ 未检索到任何结果")
            return
        
        # Step 3: 生成关联分析报告
        print("\n📊 步骤3: 生成关联分析报告")
        
        if OPENAI_API_KEY:
            analysis = await generate_analysis(patent_hits, academic_hits)
        else:
            analysis = "⚠️ OpenAI API Key 未配置，无法生成分析报告"
        
        # 打印分析报告
        print("\n" + "="*60)
        print("🔬 专利与文献语义探索报告：")
        print("="*60)
        print(analysis.encode('utf-8', 'ignore').decode('utf-8') if isinstance(analysis, str) else analysis)
        
        # 打印原始检索结果
        if patent_hits:
            print("\n" + "="*60)
            print("📚 专利相关片段详情：")
            print("="*60)
            for i, h in enumerate(patent_hits[:5]):
                title = h['title'].encode('utf-8', 'ignore').decode('utf-8')
                chunk = h['chunk'][:150].encode('utf-8', 'ignore').decode('utf-8')
                print(f"\n[{i+1}] {title}")
                print(f"    Doc ID: {h['doc_id']}, 分数: {h['score']:.3f}")
                print(f"    片段: {chunk}...")
        
        if academic_hits:
            print("\n" + "="*60)
            print("📚 学术文献片段详情：")
            print("="*60)
            for i, h in enumerate(academic_hits[:5]):
                title = h['title'].encode('utf-8', 'ignore').decode('utf-8')
                chunk = h['chunk'][:150].encode('utf-8', 'ignore').decode('utf-8')
                print(f"\n[{i+1}] {title}")
                print(f"    Doc ID: {h['doc_id']}, 分数: {h['score']:.3f}")
                print(f"    片段: {chunk}...")
        
    except Exception as e:
        print(f"❌ 发生未预期错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时: {elapsed:.2f}秒")

# 运行程序
if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=120))
    except asyncio.TimeoutError:
        print("❌ 脚本执行超时（超过2分钟）")