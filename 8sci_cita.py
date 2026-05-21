import os
import sys
import httpx
import asyncio
import time
import re
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

def split_claims(draft: str) -> list:
    """将草稿拆分为独立论点句子"""
    # 按句号、问号、感叹号拆分
    sentences = re.split(r'[。！？]', draft)
    claims = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return claims

async def search_evidence(claim: str, top_k: int = 5):
    """对单个论点检索支持证据（带超时和重试机制）"""
    if not isinstance(claim, bytes):
        claim = claim.encode('utf-8', 'ignore').decode('utf-8')
    
    timeout = Timeout(SCRAPING_TIMEOUT)
    headers = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{BASE_URL}/agentic-search",
                    headers=headers,
                    json={"query": claim, "top_k": top_k, "sub_queries": 2}
                )
                resp.raise_for_status()
                data = resp.json()
                hits = data.get("hits", [])
                
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

async def get_fulltext(doc_id: str, offset: int = 0, limit: int = 1000):
    """读取文档原文片段（带超时和重试机制）"""
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
                print(f"⚠️ 读取上下文失败 (尝试 {attempt+1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
    
    return {"text": "", "next_offset": offset, "more": False}

async def verify_with_content(hit: dict, claim: str, score_threshold: float = 0.7) -> dict:
    """读取原文验证证据是否真正支持论点"""
    # 检查分数阈值
    if hit["score"] < score_threshold:
        return {
            "doc_id": None,
            "offset": 0,
            "quote": "",
            "match_ratio": 0,
            "verified": False,
            "reason": f"证据分数不足 ({hit['score']:.2f} < {score_threshold})"
        }
    
    # 读取原文
    start_offset = max(0, hit["offset"] - 200)
    content = await get_fulltext(hit["doc_id"], offset=start_offset, limit=1000)
    text = content["text"].lower()
    
    # 提取论点中的关键词（长度大于3的词）
    claim_keywords = [w for w in claim.lower().split() if len(w) > 3]
    
    if not claim_keywords:
        return {
            "doc_id": hit["doc_id"],
            "offset": hit.get("offset", 0),
            "quote": content["text"][:200],
            "match_ratio": 0,
            "verified": False,
            "reason": "论点无有效关键词"
        }
    
    # 计算匹配度
    match_count = sum(1 for kw in claim_keywords if kw in text)
    match_ratio = match_count / len(claim_keywords)
    
    # 提取引用片段
    quote = content["text"][:200]
    
    return {
        "doc_id": hit["doc_id"],
        "offset": hit.get("offset", 0),
        "quote": quote,
        "match_ratio": match_ratio,
        "verified": match_ratio >= 0.3 and hit["score"] >= score_threshold,
        "reason": f"匹配度: {match_ratio:.2f}" if match_ratio >= 0.3 else f"匹配度不足: {match_ratio:.2f}"
    }

async def ground_claims(claims: list, score_threshold: float = 0.7):
    """对多个论点进行验证"""
    results = []
    
    for i, claim in enumerate(claims):
        print(f"\n  [{i+1}/{len(claims)}] 验证: {claim[:50]}...")
        
        # 检索证据
        hits = await search_evidence(claim, top_k=5)
        
        if hits and hits[0]["score"] >= 0.6:
            # 验证最高分的证据
            verification = await verify_with_content(hits[0], claim, score_threshold)
            results.append({
                "claim": claim,
                **verification
            })
            status = "✓" if verification["verified"] else "✗"
            print(f"    {status} 分数: {hits[0]['score']:.2f}, {verification['reason']}")
        else:
            results.append({
                "claim": claim,
                "doc_id": None,
                "offset": 0,
                "quote": "",
                "match_ratio": 0,
                "verified": False,
                "reason": "未找到相关证据"
            })
            print(f"    ✗ 未找到相关证据")
        
        # 避免请求过快
        await asyncio.sleep(0.5)
    
    return results

def build_grounded_answer(results: list) -> dict:
    """将验证结果组装为带 citation 的最终输出"""
    citations = []
    grounded_parts = []
    unverified = []
    
    for r in results:
        if r["verified"]:
            cite_id = len(citations) + 1
            citations.append({
                "id": cite_id,
                "doc_id": r["doc_id"],
                "offset": r.get("offset", 0),
                "quote": r.get("quote", ""),
                "verified": True
            })
            grounded_parts.append(f"{r['claim']}[{cite_id}]")
        else:
            grounded_parts.append(f"{r['claim']}[unverified]")
            unverified.append(r["claim"])
    
    return {
        "grounded_answer": "。".join(grounded_parts) + "。",
        "citations": citations,
        "unverified_claims": unverified
    }

async def generate_draft_answer(query: str) -> str:
    """使用OpenAI生成草稿回答"""
    print(f"\n🤖 生成草稿回答...")
    
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        timeout=Timeout(OPENAI_TIMEOUT)
    )
    
    system_prompt = "你是一个科学研究助理，请基于你的知识回答问题。回答要简洁、准确。"
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=512
            )
            draft = resp.choices[0].message.content
            print(f"✅ 草稿生成成功")
            print(f"\n📝 草稿内容:\n{draft}")
            return draft
            
        except Exception as e:
            print(f"⚠️ 生成草稿失败 (尝试 {attempt+1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    
    return "mRNA 疫苗使用可电离脂质纳米颗粒(iLNP)包裹 mRNA。其中 MC3 是最广泛使用的可电离脂质。LNP 的粒径通常在 80-100nm。"

async def main():
    """主执行函数"""
    start_time = time.time()
    print("="*60)
    print("🔬 Sciverse Citation Grounding 科学问答系统")
    print("="*60)
    
    try:
        # 设置问题
        query = "mRNA疫苗的脂质纳米颗粒递送系统有什么特点？"
        print(f"\n📝 问题: {query}")
        
        # Step 1: 生成草稿回答
        draft = await generate_draft_answer(query)
        
        # Step 2: 拆分论点
        print(f"\n📊 步骤1: 拆分论点")
        claims = split_claims(draft)
        print(f"   拆分为 {len(claims)} 个论点:")
        for i, claim in enumerate(claims):
            print(f"   [{i+1}] {claim[:60]}...")
        
        # Step 3: 逐句检索和验证
        print(f"\n📊 步骤2: 逐句检索和验证")
        results = await ground_claims(claims, score_threshold=0.7)
        
        # Step 4: 构建带引用的回答
        print(f"\n📊 步骤3: 构建带引用的最终回答")
        final = build_grounded_answer(results)
        
        # 打印结果
        print("\n" + "="*60)
        print("🔬 带引用的最终回答：")
        print("="*60)
        print(final["grounded_answer"])
        
        # 打印引用文献
        if final["citations"]:
            print("\n" + "="*60)
            print("📚 引用文献：")
            print("="*60)
            for c in final["citations"]:
                print(f"\n[{c['id']}] Doc ID: {c['doc_id']}, Offset: {c['offset']}")
                print(f"    引用原文: {c['quote'][:150]}...")
        
        # 打印未验证的论点
        if final["unverified_claims"]:
            print("\n" + "="*60)
            print("⚠️ 未验证的论点：")
            print("="*60)
            for i, claim in enumerate(final["unverified_claims"]):
                print(f"  [{i+1}] {claim}")
        
        # 打印验证统计
        verified_count = len([r for r in results if r["verified"]])
        print(f"\n📊 验证统计: {verified_count}/{len(results)} 个论点已验证")
        
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
        asyncio.run(asyncio.wait_for(main(), timeout=180))
    except asyncio.TimeoutError:
        print("❌ 脚本执行超时（超过3分钟）")