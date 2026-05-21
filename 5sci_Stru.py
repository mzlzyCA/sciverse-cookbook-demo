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
API_TIMEOUT = 30
BASE_URL = "https://api.sciverse.space"

# 算子枚举
FILTER_OP_EQ = "FILTER_OP_EQ"
FILTER_OP_NE = "FILTER_OP_NE"
FILTER_OP_GT = "FILTER_OP_GT"
FILTER_OP_GTE = "FILTER_OP_GTE"
FILTER_OP_LT = "FILTER_OP_LT"
FILTER_OP_LTE = "FILTER_OP_LTE"
FILTER_OP_IN = "FILTER_OP_IN"
FILTER_OP_NIN = "FILTER_OP_NIN"
FILTER_OP_CONTAINS = "FILTER_OP_CONTAINS"

# 排序顺序枚举
SORT_ORDER_ASC = "SORT_ORDER_ASC"
SORT_ORDER_DESC = "SORT_ORDER_DESC"

# 可排序字段
SORTABLE_FIELDS = [
    "publication_published_year",
    "reference_count",
    "citation_count",
    "influential_citation_count",
    "fwci"
]

# 默认返回字段
DEFAULT_FIELDS = [
    "doc_id",
    "title",
    "doi",
    "language",
    "publication_published_year",
    "publication_venue_name",
    "citation_count",
    "fwci",
    "author",
    "abstract"
]

# 缓存 catalog 数据
_CATALOG_CACHE = None

async def get_catalog(use_cache: bool = True):
    """获取可用字段目录（带超时、重试和缓存机制）"""
    global _CATALOG_CACHE
    
    if use_cache and _CATALOG_CACHE is not None:
        print(f"📋 使用缓存的字段目录 ({len(_CATALOG_CACHE.get('fields', []))} 个字段)")
        return _CATALOG_CACHE
    
    print(f"🔍 获取字段目录...")
    
    timeout = Timeout(API_TIMEOUT)
    headers = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.get(
                    f"{BASE_URL}/meta-catalog",
                    headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
                
                _CATALOG_CACHE = data
                
                print(f"✅ 获取字段目录成功 ({len(data.get('fields', []))} 个字段)")
                return data
                
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
                    return None
                    
            except httpx.RequestError as e:
                print(f"⚠️ 网络错误: {str(e)}，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY)
                
            except Exception as e:
                print(f"❌ 获取目录失败: {str(e)}")
                return None
    
    print(f"❌ 达到最大重试次数({MAX_RETRIES})，获取目录失败")
    return None

async def search_papers(
    query: str = None,
    filters: list = None,
    sort: list = None,
    fields: list = None,
    page: int = 1,
    page_size: int = 25,
    cursor: str = None
):
    """调用 meta-search 进行结构化检索（带超时和重试机制）
    
    Args:
        query: 全文模糊查询（可选）。设置后 results 按相关性排序，不能与 sort 同时使用
        filters: 过滤条件列表，格式: [{"field": str, "operator": str, "value": any}]
        sort: 排序条件列表，格式: [{"field": str, "order": str}]
              注意：sort 不能与 query 同时使用
        fields: 返回字段列表，doc_id 总是返回
        page: 页码（从1开始），不能与 cursor 同时使用，且 page * page_size <= 10000
        page_size: 每页数量，默认25，范围1-200
        cursor: 游标分页令牌，用于深度分页，不能与 page > 1 同时使用
    
    Returns:
        包含 results, total_count, page, page_size, total_pages, search_time_ms, next_cursor
    """
    # 参数校验
    if query and sort:
        print(f"⚠️ 警告: query 和 sort 不能同时使用，将忽略 sort")
        sort = None
    
    if cursor and page > 1:
        print(f"⚠️ 警告: cursor 不能与 page > 1 同时使用，将使用 cursor")
        page = None
    
    # 构建请求体
    body = {}
    
    if query:
        body["query"] = query.encode('utf-8', 'ignore').decode('utf-8')
        print(f"🔍 全文检索: '{query}'...")
    else:
        print(f"🔍 结构化筛选: 使用 filters 过滤...")
    
    if filters:
        body["filters"] = filters
        print(f"   过滤条件: {len(filters)} 个")
    
    if sort and not query:
        body["sort"] = sort
        print(f"   排序条件: {len(sort)} 个")
    
    if fields:
        body["fields"] = fields
    else:
        body["fields"] = DEFAULT_FIELDS
    
    if page:
        body["page"] = page
        body["page_size"] = page_size if page_size <= 200 else 200
    elif cursor:
        body["cursor"] = cursor
        body["page_size"] = page_size if page_size <= 200 else 200
    
    timeout = Timeout(API_TIMEOUT)
    headers = {"Authorization": f"Bearer {SCIVERSE_API_KEY}"}
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{BASE_URL}/meta-search",
                    headers=headers,
                    json=body
                )
                
                if resp.status_code == 401:
                    print(f"❌ 认证失败 (401): 请检查 SCIVERSE_API_KEY 是否正确")
                    return None
                elif resp.status_code == 403:
                    print(f"❌ 权限不足 (403): 请检查 fields 参数或 token 权限")
                    return None
                elif resp.status_code == 400:
                    print(f"❌ 请求参数错误 (400): {resp.text}")
                    return None
                
                resp.raise_for_status()
                data = resp.json()
                
                results = data.get("results", [])
                total_count = data.get("total_count", 0)
                total_pages = data.get("total_pages", 0)
                search_time_ms = data.get("search_time_ms", 0)
                next_cursor = data.get("next_cursor")
                
                print(f"✅ 检索到 {len(results)} 条结果，总计 {total_count} 篇论文")
                print(f"   总页数: {total_pages}, 搜索耗时: {search_time_ms:.2f}ms")
                if next_cursor:
                    print(f"   下一页游标: {next_cursor[:30]}...")
                
                # 确保所有文本字段使用UTF-8编码
                processed_results = []
                for r in results:
                    processed = {}
                    for key, value in r.items():
                        if isinstance(value, str):
                            processed[key] = value.encode('utf-8', 'ignore').decode('utf-8')
                        elif isinstance(value, list):
                            processed[key] = [
                                v.encode('utf-8', 'ignore').decode('utf-8') if isinstance(v, str) else v
                                for v in value
                            ]
                        else:
                            processed[key] = value
                    processed_results.append(processed)
                
                return {
                    "results": processed_results,
                    "total_count": total_count,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": total_pages,
                    "search_time_ms": search_time_ms,
                    "next_cursor": next_cursor
                }
                
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
                    return None
                    
            except httpx.RequestError as e:
                print(f"⚠️ 网络错误: {str(e)}，{RETRY_DELAY}秒后重试 (尝试 {attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(RETRY_DELAY)
                
            except Exception as e:
                print(f"❌ 检索失败: {str(e)}")
                return None
    
    print(f"❌ 达到最大重试次数({MAX_RETRIES})，检索失败")
    return None

def print_catalog_info(catalog: dict):
    """格式化打印字段目录信息"""
    if not catalog:
        return
    
    print("\n" + "="*60)
    print("📋 可用字段目录：")
    print("="*60)
    
    fields = catalog.get("fields", [])
    
    # 按类型分组显示
    fields_by_type = {}
    for field in fields:
        field_type = field.get("type", "unknown")
        if field_type not in fields_by_type:
            fields_by_type[field_type] = []
        fields_by_type[field_type].append(field)
    
    for field_type, field_list in fields_by_type.items():
        print(f"\n【{field_type.upper()}】类型字段 ({len(field_list)}个):")
        for field in field_list[:15]:  # 限制显示数量
            name = field.get("name", "N/A")
            operators = field.get("operators", [])
            print(f"  • {name}")
            if operators:
                print(f"    算子: {', '.join(operators[:5])}")
        if len(field_list) > 15:
            print(f"  ... 还有 {len(field_list) - 15} 个字段")

def print_search_results(result: dict, max_display: int = 10):
    """格式化打印搜索结果"""
    if not result:
        return
    
    results = result.get("results", [])
    total_count = result.get("total_count", 0)
    page = result.get("page", 1)
    total_pages = result.get("total_pages", 0)
    search_time_ms = result.get("search_time_ms", 0)
    
    print("\n" + "="*60)
    print(f"📚 搜索结果 (第{page}/{total_pages}页，共{total_count}篇论文，本页{len(results)}篇)")
    print(f"   搜索耗时: {search_time_ms:.2f}ms")
    print("="*60)
    
    for i, paper in enumerate(results[:max_display]):
        title = paper.get("title", "无标题")
        year = paper.get("publication_published_year", "N/A")
        venue = paper.get("publication_venue_name", "N/A")
        citations = paper.get("citation_count", "N/A")
        doi = paper.get("doi", "")
        language = paper.get("language", "N/A")
        
        print(f"\n[{i+1}] {title}")
        print(f"    年份: {year} | 期刊: {venue} | 引用数: {citations} | 语言: {language}")
        if doi:
            print(f"    DOI: {doi}")
        
        authors = paper.get("author", [])
        if authors:
            print(f"    作者: {', '.join(authors[:3])}")
            if len(authors) > 3:
                print(f"          ... 还有 {len(authors) - 3} 位作者")
    
    if len(results) > max_display:
        print(f"\n... 还有 {len(results) - max_display} 篇未显示")
    
    next_cursor = result.get("next_cursor")
    if next_cursor:
        print(f"\n💡 使用游标继续分页: cursor={next_cursor[:50]}...")

async def search_with_cursor_pagination(query: str = None, filters: list = None, 
                                         page_size: int = 25, max_pages: int = 5):
    """使用游标进行深度分页"""
    all_results = []
    cursor = None
    
    print(f"\n📄 使用游标分页获取数据...")
    
    for page_num in range(max_pages):
        print(f"\n  获取第 {page_num + 1} 批数据...")
        
        result = await search_papers(
            query=query,
            filters=filters,
            cursor=cursor,
            page_size=page_size
        )
        
        if not result:
            break
        
        papers = result.get("results", [])
        if not papers:
            break
        
        all_results.extend(papers)
        
        cursor = result.get("next_cursor")
        if not cursor:
            print(f"  ✅ 已获取全部数据")
            break
    
    print(f"\n✅ 共获取 {len(all_results)} 篇论文")
    return all_results

async def main():
    """主执行函数"""
    start_time = time.time()
    print("="*60)
    print("🔬 Sciverse 结构化论文筛选系统")
    print("="*60)
    
    try:
        # Step 1: 获取可用字段目录
        catalog = await get_catalog(use_cache=True)
        
        if catalog:
            print_catalog_info(catalog)
        
        # Step 2: 示例1 - 按年份范围筛选（无query，使用filters）
        print("\n" + "="*60)
        print("示例1: 按年份范围筛选 (2022-2024年论文)")
        print("="*60)
        
        results1 = await search_papers(
            query=None,  # 不使用query，只用filters
            filters=[
                {"field": "publication_published_year", "operator": FILTER_OP_GTE, "value": 2022},
                {"field": "publication_published_year", "operator": FILTER_OP_LTE, "value": 2024}
            ],
            sort=[{"field": "citation_count", "order": SORT_ORDER_DESC}],
            fields=["title", "doi", "publication_published_year", "citation_count", "author"],
            page=1,
            page_size=10
        )
        
        if results1:
            print_search_results(results1, max_display=5)
        
        # Step 3: 示例2 - 全文检索（使用query，不能用sort）
        print("\n" + "="*60)
        print("示例2: 全文检索 (CRISPR gene editing)")
        print("="*60)
        
        results2 = await search_papers(
            query="CRISPR gene editing",  # 使用query，自动按相关性排序
            filters=[
                {"field": "publication_published_year", "operator": FILTER_OP_GTE, "value": 2020}
            ],
            page=1,
            page_size=10
        )
        
        if results2:
            print_search_results(results2, max_display=5)
        
        # Step 4: 示例3 - 按期刊和语言筛选
        print("\n" + "="*60)
        print("示例3: 按期刊筛选 (Nature/Science + 英文)")
        print("="*60)
        
        results3 = await search_papers(
            query=None,
            filters=[
                {"field": "publication_venue_name", "operator": FILTER_OP_IN, "value": ["Nature", "Science"]},
                {"field": "language", "operator": FILTER_OP_EQ, "value": "en"}
            ],
            sort=[{"field": "citation_count", "order": SORT_ORDER_DESC}],
            page=1,
            page_size=10
        )
        
        if results3:
            print_search_results(results3, max_display=5)
        
        # Step 5: 示例4 - 按引用数筛选
        print("\n" + "="*60)
        print("示例4: 按引用数筛选 (引用数 >= 100)")
        print("="*60)
        
        results4 = await search_papers(
            query=None,
            filters=[
                {"field": "citation_count", "operator": FILTER_OP_GTE, "value": 100},
                {"field": "publication_published_year", "operator": FILTER_OP_GTE, "value": 2021}
            ],
            sort=[{"field": "citation_count", "order": SORT_ORDER_DESC}],
            fields=["title", "doi", "citation_count", "publication_published_year", "publication_venue_name"],
            page=1,
            page_size=10
        )
        
        if results4:
            print_search_results(results4, max_display=5)
        
        # Step 6: 示例5 - 使用游标分页
        print("\n" + "="*60)
        print("示例5: 游标分页获取更多数据")
        print("="*60)
        
        all_papers = await search_with_cursor_pagination(
            query="machine learning",
            filters=[
                {"field": "publication_published_year", "operator": FILTER_OP_GTE, "value": 2023}
            ],
            page_size=10,
            max_pages=3
        )
        
        print(f"\n📊 共获取 {len(all_papers)} 篇论文")
        for i, paper in enumerate(all_papers[:5]):
            title = paper.get("title", "无标题")[:70]
            year = paper.get("publication_published_year", "N/A")
            citations = paper.get("citation_count", "N/A")
            print(f"  [{i+1}] {title}... ({year}, 引用: {citations})")
        
    except asyncio.TimeoutError:
        print("❌ 操作超时")
    except Exception as e:
        print(f"❌ 发生未预期错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️ 总耗时: {elapsed:.2f}秒")

if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=120))
    except asyncio.TimeoutError:
        print("❌ 脚本执行超时（超过2分钟）")