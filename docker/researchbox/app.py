from fastapi import FastAPI
import httpx
import feedparser

app = FastAPI()
RSS_API = "http://aichat-rssfeed:8091"


@app.get("/search-feeds")
def search_feeds(topic: str):
    return {
        "topic": topic,
        "feeds": [
            f"https://news.google.com/rss/search?q={topic}",
            f"https://hnrss.org/newest?q={topic}",
        ],
    }


@app.post("/push-feed")
async def push_feed(payload: dict):
    topic = payload["topic"]
    feed_url = payload["feed_url"]
    parsed = feedparser.parse(feed_url)
    items = [{"title": e.get("title", "untitled"), "url": e.get("link", feed_url)} for e in parsed.entries[:20]]
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{RSS_API}/ingest", json={"topic": topic, "items": items})
        r.raise_for_status()
    return {"topic": topic, "feed_url": feed_url, "inserted": len(items)}
