import os
import sys
import httpx
import asyncio
import time
import pandas as pd
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

# 修复编码问题
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 全局配置
MAX_RETRIES = 3
RETRY_DELAY = 2
SCRAPING_TIMEOUT = 30
BASE_URL = "https://api.sciverse.space"
HEADERS = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}

async def count_by_year(query: str, year: int) -> int:
    """统计某年发文量（带超时和重试机制）"""
    timeout = Timeout(SCRAPING_TIMEOUT)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{BASE_URL}/meta-search",
                    headers=HEADERS,
                    json={
                        "query": query,
                        "filters": [
                            {"field": "publication_published_year", "operator": "FILTER_OP_EQ", "value": year}
                        ],
                        "page": 1,
                        "page_size": 1
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("total_count", 0)
                
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
                    return 0
                    
            except httpx.RequestError as e:
                print(f"⚠️ 网络错误: {str(e)}，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY)
                
            except Exception as e:
                print(f"❌ 统计失败: {str(e)}")
                return 0
    
    return 0

async def trend_scan(query: str, start_year: int = 2020, end_year: int = 2024):
    """按年统计发文量趋势"""
    print(f"🔍 扫描趋势: '{query}', {start_year}-{end_year}")
    
    tasks = [count_by_year(query, y) for y in range(start_year, end_year + 1)]
    counts = await asyncio.gather(*tasks)
    result = list(zip(range(start_year, end_year + 1), counts))
    
    print(f"✅ 统计完成")
    return result

async def top_cited_papers(year: int, top_n: int = 5):
    """查找某年度高被引论文（按引用数排序）（带超时和重试机制）"""
    timeout = Timeout(SCRAPING_TIMEOUT)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{BASE_URL}/meta-search",
                    headers=HEADERS,
                    json={
                        "filters": [
                            {"field": "publication_published_year", "operator": "FILTER_OP_EQ", "value": year}
                        ],
                        "sort": [{"field": "citation_count", "order": "SORT_ORDER_DESC"}],
                        "page": 1,
                        "page_size": top_n
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                
                # 确保文本字段使用UTF-8编码
                for r in results:
                    if "title" in r:
                        r["title"] = r["title"].encode('utf-8', 'ignore').decode('utf-8')
                    if "publication_venue_name" in r:
                        r["publication_venue_name"] = r["publication_venue_name"].encode('utf-8', 'ignore').decode('utf-8')
                
                return results
                
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
                print(f"❌ 查询失败: {str(e)}")
                return []
    
    return []

async def main():
    start_time = time.time()
    print("="*60)
    print("🔬 Sciverse 研究方向趋势扫描")
    print("="*60)
    
    try:
        # Step 2: 按年统计发文量
        print("\n📝 步骤1: 按年统计发文量")
        trend = await trend_scan("large language model", 2020, 2024)
        
        df = pd.DataFrame(trend, columns=["year", "count"])
        print("\n📊 发文量趋势:")
        print(df.to_string(index=False))
        
        # Step 3: 查找高被引论文和头部期刊
        print("\n📝 步骤2: 查找高被引论文")
        
        for year in [2022, 2023, 2024]:
            papers = await top_cited_papers(year, top_n=5)
            print(f"\n=== {year} Top Cited ===")
            for p in papers:
                venue = p.get("publication_venue_name", "N/A")
                cites = p.get("citation_count", 0)
                title = p.get("title", "无标题")[:60]
                print(f"  [{cites} cites] {title} ({venue})")
        
        # 输出趋势报告表格
        print("\n" + "="*60)
        print("📊 趋势报告")
        print("="*60)
        print(df.to_string(index=False))
        
    except Exception as e:
        print(f"❌ 发生未预期错误: {str(e)}")
    finally:
        elapsed = time.time() - start_time
        print(f"\nⱱ️ 总耗时: {elapsed:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main())