from fastapi import FastAPI, Query
from typing import List, Optional
from trendspy import Trends, BatchPeriod

app = FastAPI()
tr = Trends()

@app.get("/interest_over_time")
def interest_over_time(
    keywords: List[str] = Query(..., description="List of keywords"),
    geo: Optional[str] = Query(None, description="Geographical location code"),
    cat: Optional[str] = Query(None, description="Category ID"),
    timeframe: Optional[str] = Query(None, description="Timeframe for the data")
):
    df = tr.interest_over_time(keywords, geo=geo, cat=cat, timeframe=timeframe)
    return df.to_dict()

@app.get("/interest_by_region")
def interest_by_region(
    keyword: str,
    geo: Optional[str] = Query(None, description="Geographical location code"),
    resolution: Optional[str] = Query(None, description="Resolution level: COUNTRY, REGION, CITY")
):
    df = tr.interest_by_region(keyword, geo=geo, resolution=resolution)
    return df.to_dict()

@app.get("/related_queries")
def related_queries(keyword: str):
    data = tr.related_queries(keyword)
    return data

@app.get("/related_topics")
def related_topics(keyword: str):
    data = tr.related_topics(keyword)
    return data

@app.get("/trending_now")
def trending_now(geo: Optional[str] = Query(None, description="Geographical location code")):
    try:
        data = tr.trending_now(geo=geo)

        # Defensive dict conversion
        result = []
        for item in data:
            # Try .__dict__ or fallback to str
            if hasattr(item, '__dict__'):
                result.append(item.__dict__)
            else:
                result.append(str(item))
                
        return result

    except Exception as e:
        logging.exception("🔥 trending_now failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trending_now_by_rss")
def trending_now_by_rss(geo: Optional[str] = Query(None, description="Geographical location code")):
    data = tr.trending_now_by_rss(geo=geo)
    return [item.dict() for item in data]

@app.get("/trending_now_news_by_ids")
def trending_now_news_by_ids(
    news_tokens: List[str] = Query(..., description="List of news tokens"),
    max_news: Optional[int] = Query(3, description="Maximum number of news articles to retrieve")
):
    data = tr.trending_now_news_by_ids(news_tokens, max_news=max_news)
    return [article.dict() for article in data]

@app.get("/trending_now_showcase_timeline")
def trending_now_showcase_timeline(
    keywords: List[str] = Query(..., description="List of keywords"),
    timeframe: BatchPeriod = Query(..., description="Timeframe for the data")
):
    df = tr.trending_now_showcase_timeline(keywords, timeframe=timeframe)
    return df.to_dict()

@app.get("/categories")
def categories(find: Optional[str] = Query(None, description="Search term for categories")):
    data = tr.categories(find=find)
    return data

@app.get("/geo")
def geo(find: Optional[str] = Query(None, description="Search term for geographical locations")):
    data = tr.geo(find=find)
    return data
