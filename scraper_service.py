from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional
import subprocess
import json
import os
import logging
from pathlib import Path
import spacy
from textblob import TextBlob

class ScraperService:
    def __init__(self, skraper_path: str = '/usr/local/bin/skraper'):
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set path to skraper binary
        self.SKRAPER_PATH = Path(skraper_path)
        
        # Define allowed networks
        self.ALLOWED_NETWORKS = {'twitter', 'reddit', 'instagram'}
        
        # Initialize router
        self.router = APIRouter(prefix="/api/v1", tags=["scraper"])
        self.setup_routes()
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            self.logger.error("spaCy model 'en_core_web_md' not found. Please install it using: python -m spacy download en_core_web_md")
            raise

    def setup_routes(self):
        """Configure API routes"""
        self.router.add_api_route("/run/scraper", self.run_scraper, methods=["POST"], response_model=dict)

    class ScraperAction(BaseModel):
        network: str
        query: str
        
        @validator('network')
        def validate_network(cls, value: str) -> str:
            allowed_networks = {'twitter', 'reddit', 'instagram', 'facebook', '9gag'}
            if value.lower() not in allowed_networks:
                raise ValueError(f"Network must be one of: {', '.join(allowed_networks)}")
            return value.lower()

    class EnrichedItem(BaseModel):
        original_item: Dict[str, Any]
        sentiment: float
        entities: List[Dict[str, Any]]
        keywords: List[str]

    class EnrichedData(BaseModel):
        items: List["ScraperService.EnrichedItem"]

    def enrich_data(self, data: Any) -> "ScraperService.EnrichedData":
        """
        Enrich scraped data with NLP analysis for each item in the list.
        
        Args:
            data: Original scraped data (expected to be a list of dictionaries)
            
        Returns:
            EnrichedData with a list of enriched items
        """
        if not isinstance(data, list):
            self.logger.error(f"Expected a list for scraped data, got {type(data)}")
            raise ValueError("Scraped data must be a list of items")
        
        enriched_items = []
        
        for item in data:
            if not isinstance(item, dict):
                self.logger.warning(f"Skipping invalid item: {item}")
                continue
                
            # Extract text content (adjust based on your data structure)
            text_content = item.get('content', '') or item.get('text', '') or ''
            if not text_content:
                self.logger.warning(f"No text content found in item: {item}")
                enriched_items.append(self.EnrichedItem(
                    original_item=item,
                    sentiment=0.0,
                    entities=[],
                    keywords=[]
                ))
                continue
                
            try:
                # Process with spaCy
                doc = self.nlp(text_content)
                
                # Sentiment analysis
                sentiment = TextBlob(text_content).sentiment.polarity
                
                # Named Entity Recognition
                entities = [
                    {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                    for ent in doc.ents
                ]
                
                # Keyword extraction
                keywords = list(set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 2))
                
                enriched_items.append(self.EnrichedItem(
                    original_item=item,
                    sentiment=sentiment,
                    entities=entities,
                    keywords=keywords[:10]  # Limit to top 10 keywords
                ))
            except Exception as e:
                self.logger.error(f"Error processing item {item}: {str(e)}")
                enriched_items.append(self.EnrichedItem(
                    original_item=item,
                    sentiment=0.0,
                    entities=[],
                    keywords=[]
                ))
        
        if not enriched_items:
            self.logger.warning("No valid items were enriched")
        
        return self.EnrichedData(items=enriched_items)

    async def run_scraper(self, scraper: "ScraperService.ScraperAction"):
        """
        Execute scraper CLI command and return enriched results with NLP analysis.
        
        Args:
            scraper: ScraperAction model containing network and query
            
        Returns:
            Dictionary containing execution log and enriched scraped data
            
        Raises:
            HTTPException: For various error conditions
        """
        # Verify skraper binary
        if not self.SKRAPER_PATH.exists():
            self.logger.error(f"Skraper binary not found at {self.SKRAPER_PATH}")
            raise HTTPException(status_code=500, detail="Skraper binary not found")
        
        if not self.SKRAPER_PATH.is_file() or not os.access(self.SKRAPER_PATH, os.X_OK):
            self.logger.error(f"Skraper binary at {self.SKRAPER_PATH} is not executable")
            raise HTTPException(status_code=500, detail="Skraper binary not executable")
            
        # Sanitize input
        safe_network = scraper.network.replace(' ', '')
        safe_query = scraper.query.replace(' ', '')
        
        try:
            # Execute command
            cmd = [str(self.SKRAPER_PATH), safe_network, safe_query, '-t', 'json']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            # Parse output for file path
            file_path = None
            for line in result.stdout.splitlines():
                if "has been written to" in line:
                    file_path = line.split("has been written to ")[-1].strip()
                    break
                    
            if not file_path:
                self.logger.warning("Could not find generated file path in output")
                raise HTTPException(status_code=500, detail="Could not find generated file path")
                
            # Read and parse JSON file
            try:
                with open(file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    print(json_data)
                
                # Enrich the data
                enriched_data = self.enrich_data(json_data).dict()
                
                # Clean up temporary file
                try:
                    os.remove(file_path)
                    self.logger.info(f"Cleaned up temporary file: {file_path}")
                except OSError as e:
                    self.logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
                    
                return {
                    "execution_log": result.stdout,
                    "scraped_data": enriched_data
                }
            except FileNotFoundError:
                self.logger.error(f"Generated file not found at {file_path}")
                raise HTTPException(status_code=500, detail=f"Generated file not found at {file_path}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in file {file_path}: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to parse generated JSON file")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Scraper command failed: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Command execution failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            self.logger.error("Scraper command timed out")
            raise HTTPException(status_code=504, detail="Scraper command timed out")

# Usage example:
# scraper_service = ScraperService()
# app = FastAPI()
# app.include_router(scraper_service.router)