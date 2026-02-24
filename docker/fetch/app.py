"""aichat-fetch: web page fetching service.

Fetches a URL and returns clean, readable text. Useful for reading
documentation, articles, and any content the AI needs to reference.
"""
from __future__ import annotations

import html2text
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="aichat-fetch")

_converter = html2text.HTML2Text()
_converter.ignore_links = False
_converter.ignore_images = True
_converter.body_width = 0  # no wrapping

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}


class FetchRequest(BaseModel):
    url: str
    max_chars: int = 4000


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/fetch")
async def fetch(req: FetchRequest) -> dict:
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=20,
            verify=False,  # Docker containers lack system CA bundle; safe for a local fetch proxy
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Upstream error: {exc.response.status_code}",
        ) from exc
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Request failed: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "html" in content_type:
        text = _converter.handle(response.text)
    else:
        text = response.text

    text = text.strip()
    truncated = False
    if len(text) > req.max_chars:
        text = text[: req.max_chars]
        truncated = True

    return {
        "url": str(response.url),
        "status_code": response.status_code,
        "char_count": len(text),
        "truncated": truncated,
        "text": text,
    }
