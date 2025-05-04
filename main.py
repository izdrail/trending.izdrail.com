from fastapi import FastAPI, Query, Depends, HTTPException, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any, Union
from trendspy import Trends, BatchPeriod
from sqlalchemy import Column, Integer, String, DateTime, JSON, create_engine, Float, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import json
import os
import time
from functools import lru_cache
import pandas as pd
import uuid
from pathlib import Path
import asyncio
from fastapi_versioning import VersionedFastAPI, version
from cachetools import TTLCache

# ----- CONFIGURATION -----
class Settings:
    APP_TITLE = "TrendsPy API"
    APP_DESCRIPTION = "An API wrapper for Google Trends analysis using TrendsPy"
    APP_VERSION = "1.0.0"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trends.db")
    CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))  # Cache time-to-live in seconds
    CACHE_MAXSIZE = int(os.getenv("CACHE_MAXSIZE", 100))  # Maximum cache size
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", 60))  # Period in seconds
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 30))  # Max requests per period
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DATA_DIR = os.getenv("DATA_DIR", "./data")

settings = Settings()

# ----- LOGGING SETUP -----
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trendspy_api")

# ----- DATABASE SETUP -----
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ----- DATABASE MODELS -----
class ApiRequest(Base):
    __tablename__ = "api_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String, index=True)
    ip_address = Column(String)
    params = Column(JSON)
    status_code = Column(Integer)
    response_time = Column(Float)  # in seconds
    timestamp = Column(DateTime, default=datetime.utcnow)

class InterestOverTime(Base):
    __tablename__ = "interest_over_time"
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True, unique=True)
    keyword = Column(String, index=True)
    geo = Column(String, nullable=True, index=True)
    cat = Column(String, nullable=True, index=True)
    timeframe = Column(String, nullable=True)
    data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class DataExport(Base):
    __tablename__ = "data_exports"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    query_type = Column(String, index=True)
    params = Column(JSON)
    format = Column(String)
    status = Column(String, index=True)  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

class Comparison(Base):
    __tablename__ = "comparisons"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(Text, nullable=True)
    keywords = Column(JSON)
    geo = Column(String, nullable=True)
    timeframe = Column(String, nullable=True)
    cat = Column(String, nullable=True)
    data = Column(JSON, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# ----- SCHEMA MODELS -----
class InterestOverTimeRequest(BaseModel):
    keywords: List[str] = Field(..., description="List of keywords to analyze")
    geo: Optional[str] = Field(None, description="Geographical location code")
    cat: Optional[str] = Field(None, description="Category ID")
    timeframe: str = Field("now 7-d", description="Timeframe for the data")

class ComparisonCreate(BaseModel):
    title: str = Field(..., description="Title of the comparison")
    description: Optional[str] = Field(None, description="Description of the comparison")
    keywords: List[str] = Field(..., description="Keywords to compare")
    geo: Optional[str] = Field(None, description="Geographical location")
    timeframe: str = Field("now 7-d", description="Timeframe for comparison")
    cat: Optional[str] = Field(None, description="Category ID")

class ExportRequest(BaseModel):
    query_type: str = Field(..., description="Type of query to export (interest_over_time, interest_by_region, etc.)")
    params: Dict[str, Any] = Field(..., description="Parameters for the query")
    format: str = Field("csv", description="Export format (csv, json, excel)")

class ApiVersionInfo(BaseModel):
    version: str
    release_date: str
    changes: List[str]

# ----- DEPENDENCIES -----
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Global cache setup
request_cache = TTLCache(maxsize=settings.CACHE_MAXSIZE, ttl=settings.CACHE_TTL)

@lru_cache(maxsize=1)
def get_trends_client():
    return Trends()

# Rate limiting tracker
rate_limit_tracker = {}

async def check_rate_limit(request: Request):
    client_ip = request.client.host
    current_time = time.time()
    
    # Initialize or cleanup old entries
    if client_ip not in rate_limit_tracker:
        rate_limit_tracker[client_ip] = []
    
    # Remove requests older than the rate limit period
    rate_limit_tracker[client_ip] = [
        timestamp for timestamp in rate_limit_tracker[client_ip]
        if current_time - timestamp < settings.RATE_LIMIT_PERIOD
    ]
    
    # Check if too many requests
    if len(rate_limit_tracker[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_PERIOD} seconds."
        )
    
    # Add current request timestamp
    rate_limit_tracker[client_ip].append(current_time)

# ----- MIDDLEWARE -----
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Process the request
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        logger.exception("Request failed")
        status_code = 500
        raise e
    finally:
        process_time = time.time() - start_time
        
    # Log request details
    try:
        db = SessionLocal()
        log_entry = ApiRequest(
            endpoint=request.url.path,
            ip_address=request.client.host,
            params=dict(request.query_params),
            status_code=status_code,
            response_time=process_time
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log request: {e}")
    finally:
        db.close()
    
    return response

# ----- HELPER FUNCTIONS -----
def cache_key(func_name, *args, **kwargs):
    """Generate a cache key for the function with given args and kwargs"""
    key = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
    return key

async def process_export(db: Session, export_id: int, query_type: str, params: dict, format: str):
    """Background task to process data exports"""
    try:
        # Get export record
        export = db.query(DataExport).filter(DataExport.id == export_id).first()
        if not export:
            logger.error(f"Export ID {export_id} not found")
            return
            
        # Update status to processing
        export.status = "processing"
        db.commit()
        
        # Create data directory if it doesn't exist
        data_dir = Path(settings.DATA_DIR)
        data_dir.mkdir(exist_ok=True)
        
        # Initialize trends client
        tr = get_trends_client()
        
        # Execute the appropriate query based on query_type
        if query_type == "interest_over_time":
            data = tr.interest_over_time(
                params.get("keywords", []), 
                geo=params.get("geo"), 
                cat=params.get("cat"),
                timeframe=params.get("timeframe", "now 7-d")
            )
        elif query_type == "interest_by_region":
            data = tr.interest_by_region(
                params.get("keyword", ""), 
                geo=params.get("geo"),
                resolution=params.get("resolution")
            )
        elif query_type == "related_queries":
            data = tr.related_queries(params.get("keyword", ""))
            # Convert to DataFrame for easier export
            data = pd.DataFrame(data)
        elif query_type == "related_topics":
            data = tr.related_topics(params.get("keyword", ""))
            # Convert to DataFrame for easier export
            data = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        # Save to file in specified format
        filename = f"{query_type}_{int(time.time())}"
        filepath = data_dir / f"{filename}.{format}"
        
        if format == "csv":
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=True)
            else:
                pd.DataFrame(data).to_csv(filepath, index=True)
        elif format == "json":
            if isinstance(data, pd.DataFrame):
                data.to_json(filepath, orient="records")
            else:
                with open(filepath, "w") as f:
                    json.dump(data, f)
        elif format == "excel":
            if isinstance(data, pd.DataFrame):
                data.to_excel(filepath, index=True)
            else:
                pd.DataFrame(data).to_excel(filepath, index=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        # Update export record
        export.filename = str(filepath)
        export.status = "completed"
        export.completed_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Export {export_id} completed: {filepath}")
        
    except Exception as e:
        logger.exception(f"Export {export_id} failed")
        try:
            export.status = "failed"
            db.commit()
        except:
            pass

# ----- API ROUTES -----
@app.get("/", include_in_schema=False)
def read_root():
    return {
        "message": "Welcome to TrendsPy API",
        "docs_url": "/docs",
        "version": settings.APP_VERSION
    }

@app.get("/api/v1/version")
@version(1)
def get_version():
    """Get API version information"""
    return {
        "version": settings.APP_VERSION,
        "release_date": "2025-05-04",
        "changes": [
            "Initial API release",
            "Support for all TrendsPy features",
            "Database persistence",
            "Caching",
            "Rate limiting"
        ]
    }

@app.get("/api/v1/health")
@version(1)
def health_check():
    """API health check endpoint"""
    try:
        # Check database connection
        db = SessionLocal()
        db.execute("SELECT 1").fetchone()
        db.close()
        
        # Check TrendsPy client
        tr = get_trends_client()
        
        return {
            "status": "healthy",
            "database": "connected",
            "trendspy_client": "initialized",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.post("/api/v1/interest_over_time")
@version(1)
async def interest_over_time_v1(
    request: Request,
    data: InterestOverTimeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Get interest over time data for keywords"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    request_id = str(uuid.uuid4())
    
    # Check cache
    cache_key_str = cache_key(
        "interest_over_time", 
        data.keywords, 
        geo=data.geo, 
        cat=data.cat, 
        timeframe=data.timeframe
    )
    
    if cache_key_str in request_cache:
        logger.info(f"Cache hit for {cache_key_str}")
        return request_cache[cache_key_str]
    
    try:
        df = tr.interest_over_time(
            data.keywords, 
            geo=data.geo, 
            cat=data.cat, 
            timeframe=data.timeframe
        )
        
        result = df.to_dict()
        request_cache[cache_key_str] = result
        
        # Store data asynchronously
        def save_to_db():
            try:
                for keyword in data.keywords:
                    if keyword in df:
                        entry = InterestOverTime(
                            request_id=request_id,
                            keyword=keyword,
                            geo=data.geo,
                            cat=data.cat,
                            timeframe=data.timeframe,
                            data=json.loads(df[keyword].to_json())
                        )
                        db.add(entry)
                db.commit()
            except Exception as e:
                logger.error(f"Failed to save interest_over_time data: {e}")
                db.rollback()
        
        background_tasks.add_task(save_to_db)
        return result
        
    except Exception as e:
        logger.exception("interest_over_time_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/interest_by_region")
@version(1)
async def interest_by_region_v1(
    request: Request,
    keyword: str = Query(..., description="Keyword to analyze"),
    geo: Optional[str] = Query(None, description="Geographical location code"),
    resolution: Optional[str] = Query(None, description="Geographic resolution (e.g. COUNTRY, REGION)")
):
    """Get interest by region data for a keyword"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    
    # Check cache
    cache_key_str = cache_key("interest_by_region", keyword, geo=geo, resolution=resolution)
    
    if cache_key_str in request_cache:
        logger.info(f"Cache hit for {cache_key_str}")
        return request_cache[cache_key_str]
    
    try:
        df = tr.interest_by_region(keyword, geo=geo, resolution=resolution)
        result = df.to_dict()
        request_cache[cache_key_str] = result
        return result
    except Exception as e:
        logger.exception("interest_by_region_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/related_queries")
@version(1)
async def related_queries_v1(
    request: Request,
    keyword: str = Query(..., description="Keyword to analyze")
):
    """Get related queries for a keyword"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    
    # Check cache
    cache_key_str = cache_key("related_queries", keyword)
    
    if cache_key_str in request_cache:
        logger.info(f"Cache hit for {cache_key_str}")
        return request_cache[cache_key_str]
    
    try:
        result = tr.related_queries(keyword)
        request_cache[cache_key_str] = result
        return result
    except Exception as e:
        logger.exception("related_queries_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/related_topics")
@version(1)
async def related_topics_v1(
    request: Request,
    keyword: str = Query(..., description="Keyword to analyze")
):
    """Get related topics for a keyword"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    
    # Check cache
    cache_key_str = cache_key("related_topics", keyword)
    
    if cache_key_str in request_cache:
        logger.info(f"Cache hit for {cache_key_str}")
        return request_cache[cache_key_str]
    
    try:
        result = tr.related_topics(keyword)
        request_cache[cache_key_str] = result
        return result
    except Exception as e:
        logger.exception("related_topics_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trending_now")
@version(1)
async def trending_now_v1(
    request: Request,
    geo: Optional[str] = Query(None, description="Geographical location code")
):
    """Get trending searches right now"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    
    # Check cache but with short TTL (5 minutes for trending data)
    cache_key_str = cache_key("trending_now", geo=geo)
    
    current_time = time.time()
    if cache_key_str in request_cache:
        cache_time, cache_data = request_cache[cache_key_str]
        # Use a shorter TTL (5 minutes) for trending data
        if current_time - cache_time < 300:  
            logger.info(f"Cache hit for {cache_key_str}")
            return cache_data
    
    try:
        data = tr.trending_now(geo=geo)
        result = [item.__dict__ if hasattr(item, '__dict__') else str(item) for item in data]
        request_cache[cache_key_str] = (current_time, result)
        return result
    except Exception as e:
        logger.exception("trending_now_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trending_now_by_rss")
@version(1)
async def trending_now_by_rss_v1(
    request: Request,
    geo: Optional[str] = Query(None, description="Geographical location code")
):
    """Get trending searches by RSS feed"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    
    # Check cache but with short TTL
    cache_key_str = cache_key("trending_now_by_rss", geo=geo)
    
    current_time = time.time()
    if cache_key_str in request_cache:
        cache_time, cache_data = request_cache[cache_key_str]
        # Use a shorter TTL (5 minutes) for trending data
        if current_time - cache_time < 300:  
            logger.info(f"Cache hit for {cache_key_str}")
            return cache_data
    
    try:
        data = tr.trending_now_by_rss(geo=geo)
        result = [vars(item) if hasattr(item, "__dict__") else str(item) for item in data]
        request_cache[cache_key_str] = (current_time, result)
        return result
    except Exception as e:
        logger.exception("trending_now_by_rss_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trending_now_news_by_ids")
@version(1)
async def trending_now_news_by_ids_v1(
    request: Request,
    news_tokens: List[str] = Query(..., description="News tokens to fetch"),
    max_news: Optional[int] = Query(3, description="Maximum number of news items to return")
):
    """Get trending news by IDs"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    
    try:
        data = tr.trending_now_news_by_ids(news_tokens, max_news=max_news)
        return [vars(article) if hasattr(article, '__dict__') else str(article) for article in data]
    except Exception as e:
        logger.exception("trending_now_news_by_ids_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/trending_now_showcase_timeline")
@version(1)
async def trending_now_showcase_timeline_v1(
    request: Request,
    keywords: List[str] = Query(..., description="Keywords to analyze"),
    timeframe: str = Query(..., description="Timeframe for analysis")
):
    """Get trending showcase timeline"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    
    try:
        df = tr.trending_now_showcase_timeline(keywords, timeframe=timeframe)
        return df.to_dict()
    except Exception as e:
        logger.exception("trending_now_showcase_timeline_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/categories")
@version(1)
async def categories_v1(
    request: Request,
    find: Optional[str] = Query(None, description="Text to search for in categories")
):
    """Get available categories, optionally filtered by search text"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    
    # This data rarely changes, so we can cache it longer
    cache_key_str = cache_key("categories", find=find)
    
    if cache_key_str in request_cache:
        logger.info(f"Cache hit for {cache_key_str}")
        return request_cache[cache_key_str]
    
    try:
        result = tr.categories(find=find)
        request_cache[cache_key_str] = result
        return result
    except Exception as e:
        logger.exception("categories_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/geo")
@version(1)
async def geo_v1(
    request: Request,
    find: Optional[str] = Query(None, description="Text to search for in geographic locations")
):
    """Get available geographic locations, optionally filtered by search text"""
    await check_rate_limit(request)
    
    tr = get_trends_client()
    
    # This data rarely changes, so we can cache it longer
    cache_key_str = cache_key("geo", find=find)
    
    if cache_key_str in request_cache:
        logger.info(f"Cache hit for {cache_key_str}")
        return request_cache[cache_key_str]
    
    try:
        result = tr.geo(find=find)
        request_cache[cache_key_str] = result
        return result
    except Exception as e:
        logger.exception("geo_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history")
@version(1)
async def history_v1(
    request: Request,
    keyword: Optional[str] = Query(None, description="Filter by keyword"),
    geo: Optional[str] = Query(None, description="Filter by geographical location"),
    cat: Optional[str] = Query(None, description="Filter by category"),
    days: Optional[int] = Query(30, description="Number of days to look back"),
    limit: Optional[int] = Query(100, description="Maximum number of records to return"),
    offset: Optional[int] = Query(0, description="Number of records to skip"),
    db: Session = Depends(get_db)
):
    """Get historical interest over time data"""
    await check_rate_limit(request)
    
    query = db.query(InterestOverTime)
    
    if keyword:
        query = query.filter(InterestOverTime.keyword == keyword)
    if geo:
        query = query.filter(InterestOverTime.geo == geo)
    if cat:
        query = query.filter(InterestOverTime.cat == cat)
    if days:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(InterestOverTime.timestamp >= cutoff_date)
    
    total = query.count()
    
    query = query.order_by(InterestOverTime.timestamp.desc())
    query = query.offset(offset).limit(limit)
    
    results = [
        {
            "id": row.id,
            "request_id": row.request_id,
            "keyword": row.keyword,
            "geo": row.geo,
            "cat": row.cat,
            "timeframe": row.timeframe,
            "data": row.data,
            "timestamp": row.timestamp.isoformat()
        } for row in query.all()
    ]
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results
    }

@app.post("/api/v1/export")
@version(1)
async def create_export_v1(
    request: Request,
    data: ExportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a data export job"""
    await check_rate_limit(request)
    
    # Validate export format
    if data.format not in ["csv", "json", "excel"]:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {data.format}")
    
    # Validate query type
    if data.query_type not in ["interest_over_time", "interest_by_region", "related_queries", "related_topics"]:
        raise HTTPException(status_code=400, detail=f"Unsupported query type: {data.query_type}")
    
    try:
        # Create export record
        export = DataExport(
            query_type=data.query_type,
            params=data.params,
            format=data.format,
            status="pending"
        )
        db.add(export)
        db.commit()
        db.refresh(export)
        
        # Start background task to process export
        background_tasks.add_task(
            process_export, 
            db=SessionLocal(), 
            export_id=export.id,
            query_type=data.query_type,
            params=data.params,
            format=data.format
        )
        
        return {
            "id": export.id,
            "status": export.status,
            "created_at": export.created_at.isoformat()
        }
        
    except Exception as e:
        logger.exception("create_export_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/export/{export_id}")
@version(1)
async def get_export_status_v1(
    request: Request,
    export_id: int,
    db: Session = Depends(get_db)
):
    """Get status of a data export job"""
    await check_rate_limit(request)
    
    export = db.query(DataExport).filter(DataExport.id == export_id).first()
    if not export:
        raise HTTPException(status_code=404, detail=f"Export with ID {export_id} not found")
    
    return {
        "id": export.id,
        "query_type": export.query_type,
        "format": export.format,
        "status": export.status,
        "created_at": export.created_at.isoformat(),
        "completed_at": export.completed_at.isoformat() if export.completed_at else None,
        "filename": export.filename if export.status == "completed" else None
    }

@app.post("/api/v1/comparisons")
@version(1)
async def create_comparison_v1(
    request: Request,
    data: ComparisonCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new keyword comparison"""
    await check_rate_limit(request)
    
    try:
        # Create comparison record
        comparison = Comparison(
            title=data.title,
            description=data.description,
            keywords=data.keywords,
            geo=data.geo,
            timeframe=data.timeframe,
            cat=data.cat
        )
        db.add(comparison)
        db.commit()
        db.refresh(comparison)
        
        # Start background task to fetch comparison data
        def fetch_comparison_data():
            try:
                tr = get_trends_client()
                df = tr.interest_over_time(
                    data.keywords,
                    geo=data.geo,
                    cat=data.cat,
                    timeframe=data.timeframe
                )
                
                # Update comparison with data
                session = SessionLocal()
                comp = session.query(Comparison).filter(Comparison.id == comparison.id).first()
                if comp:
                    comp.data = json.loads(df.to_json())
                    comp.last_updated = datetime.utcnow()
                    session.commit()
            except Exception as e:
                logger.exception(f"Failed to fetch comparison data for ID {comparison.id}")
            finally:
                session.close()
        
        background_tasks.add_task(fetch_comparison_data)
        
        return {
            "id": comparison.id,
            "title": comparison.title,
            "keywords": comparison.keywords,
            "created_at": comparison.created_at.isoformat()
        }
        
    except Exception as e:
        logger.exception("create_comparison_v1 failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/comparisons")
@version(1)
async def list_comparisons_v1(
    request: Request,
    limit: Optional[int] = Query(10, description="Maximum number of comparisons to return"),
    offset: Optional[int] = Query(0, description="Number of comparisons to skip"),
    db: Session = Depends(get_db)
):
    """List all saved comparisons"""
    await check_rate_limit(request)
    
    total = db.query(func.count(Comparison.id)).scalar()
    
    comparisons = db.query(Comparison)\
        .order_by(Comparison.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": [
            {
                "id": comp.id,
                "title": comp.title,
                "description": comp.description,
                "keywords": comp.keywords,
                "geo": comp.geo,
                "timeframe": comp.timeframe,
                "cat": comp.cat,
                "last_updated": comp.last_updated.isoformat(),
                "created_at": comp.created_at.isoformat()
            } for comp in comparisons
        ]
    }

@app.get("/api/v1/comparisons/{comparison_id}")
@version(1)
async def get_comparison_v1(
    request: Request,
    comparison_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific comparison by ID"""
    await check_rate_limit(request)
    
    comparison = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    if not comparison:
        raise HTTPException(status_code=404, detail=f"Comparison with ID {comparison_id} not found")
    
    return {
        "id": comparison.id,
        "title": comparison.title,
        "description": comparison.description,
        "keywords": comparison.keywords,
        "geo": comparison.geo,
        "timeframe": comparison.timeframe,
        "cat": comparison.cat,
        "data": comparison.data,
        "last_updated": comparison.last_updated.isoformat(),
        "created_at": comparison.created_at.isoformat()
    }

@app.delete("/api/v1/comparisons/{comparison_id}")
@version(1)
async def delete_comparison_v1(
    request: Request,
    comparison_id: int,
    db: Session = Depends(get_db)
):
    """Delete a specific comparison by ID"""
    await check_rate_limit(request)
    
    comparison = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    if not comparison:
        raise HTTPException(status_code=404, detail=f"Comparison with ID {comparison_id} not found")
    
    db.delete(comparison)
    db.commit()
    
    return {"message": f"Comparison with ID {comparison_id} successfully deleted"}

@app.put("/api/v1/comparisons/{comparison_id}")
@version(1)
async def update_comparison_v1(
    request: Request,
    comparison_id: int,
    data: ComparisonCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Update a specific comparison by ID"""
    await check_rate_limit(request)
    
    comparison = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    if not comparison:
        raise HTTPException(status_code=404, detail=f"Comparison with ID {comparison_id} not found")
    
    # Update comparison fields
    comparison.title = data.title
    comparison.description = data.description
    comparison.keywords = data.keywords
    comparison.geo = data.geo
    comparison.timeframe = data.timeframe
    comparison.cat = data.cat
    
    db.commit()
    
    # Start background task to refresh comparison data
    def refresh_comparison_data():
        try:
            tr = get_trends_client()
            df = tr.interest_over_time(
                data.keywords,
                geo=data.geo,
                cat=data.cat,
                timeframe=data.timeframe
            )
            
            # Update comparison with data
            session = SessionLocal()
            comp = session.query(Comparison).filter(Comparison.id == comparison_id).first()
            if comp:
                comp.data = json.loads(df.to_json())
                comp.last_updated = datetime.utcnow()
                session.commit()
        except Exception as e:
            logger.exception(f"Failed to refresh comparison data for ID {comparison_id}")
        finally:
            session.close()
    
    background_tasks.add_task(refresh_comparison_data)
    
    return {
        "id": comparison.id,
        "title": comparison.title,
        "description": comparison.description,
        "keywords": comparison.keywords,
        "geo": comparison.geo,
        "timeframe": comparison.timeframe,
        "cat": comparison.cat,
        "last_updated": comparison.last_updated.isoformat(),
        "created_at": comparison.created_at.isoformat()
    }

@app.get("/api/v1/stats")
@version(1)
async def get_api_stats_v1(
    request: Request,
    days: Optional[int] = Query(7, description="Number of days to look back"),
    db: Session = Depends(get_db)
):
    """Get API usage statistics"""
    await check_rate_limit(request)
    
    # Calculate time range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get total requests
    total_requests = db.query(func.count(ApiRequest.id))\
        .filter(ApiRequest.timestamp >= start_date)\
        .scalar()
    
    # Get requests by endpoint
    requests_by_endpoint = db.query(
            ApiRequest.endpoint,
            func.count(ApiRequest.id).label("count"),
            func.avg(ApiRequest.response_time).label("avg_response_time")
        )\
        .filter(ApiRequest.timestamp >= start_date)\
        .group_by(ApiRequest.endpoint)\
        .order_by(text("count DESC"))\
        .all()
    
    # Get requests by status code
    requests_by_status = db.query(
            ApiRequest.status_code,
            func.count(ApiRequest.id).label("count")
        )\
        .filter(ApiRequest.timestamp >= start_date)\
        .group_by(ApiRequest.status_code)\
        .order_by(text("count DESC"))\
        .all()
    
    # Get requests by day
    requests_by_day = db.query(
            func.date(ApiRequest.timestamp).label("date"),
            func.count(ApiRequest.id).label("count")
        )\
        .filter(ApiRequest.timestamp >= start_date)\
        .group_by(text("date"))\
        .order_by(text("date"))\
        .all()
    
    return {
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days": days
        },
        "total_requests": total_requests,
        "requests_by_endpoint": [
            {
                "endpoint": endpoint,
                "count": count,
                "avg_response_time": round(avg_response_time, 3)
            }
            for endpoint, count, avg_response_time in requests_by_endpoint
        ],
        "requests_by_status": [
            {
                "status_code": status_code,
                "count": count
            }
            for status_code, count in requests_by_status
        ],
        "requests_by_day": [
            {
                "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
                "count": count
            }
            for date, count in requests_by_day
        ]
    }

@app.delete("/api/v1/cache")
@version(1)
async def clear_cache_v1(request: Request):
    """Clear the API cache"""
    await check_rate_limit(request)
    
    # Require admin authorization (simplified for example)
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != os.getenv("ADMIN_API_KEY", "admin_secret_key"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API key required"
        )
    
    # Clear the cache
    request_cache.clear()
    
    return {"message": "Cache cleared successfully"}

# ----- API V2 ROUTES (FUTURE EXPANSION) -----
# This section would contain newer versions of endpoints when needed

# ----- VERSIONED APP -----
# Apply versioning to the FastAPI app
app = VersionedFastAPI(app,
    version_format='{major}',
    prefix_format='/api/v{major}',
    default_version=(1, 0),
    enable_latest=True
)

# Create a static directory for downloads if it doesn't exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
app.mount("/downloads", StaticFiles(directory=settings.DATA_DIR), name="downloads")

# ----- STARTUP EVENTS -----
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.APP_TITLE} v{settings.APP_VERSION}")
    
    # Initialize database if needed
    Base.metadata.create_all(bind=engine)
    
    # Pre-initialize the trends client
    _ = get_trends_client()
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API")

# If this module is run directly, start the app with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=1099, reload=True)