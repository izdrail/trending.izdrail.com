import asyncio

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from endpoints import feeds

app = FastAPI(
    title="Trending API",
    description="This is a collection of endpoints that powers some of my hobby projects",
    version="0.0.1",
    terms_of_service="https://izdrail.com/terms/",

    contact={
        "name": "Stefan",
        "url": "https://izdrail.com/",
        "email": "stefan@izdrail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Endpoints
app.include_router(feeds.router)



@app.get("/")
async def root():
    return {"data": "You can try the latest API endpoint here -> https://trending.izdrail.com/docs"}

# This line is removed as it was causing the issue
# loop = asyncio.get_event_loop()
# loop.run_until_complete()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=True)