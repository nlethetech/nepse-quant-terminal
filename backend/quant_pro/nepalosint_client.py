"""Small NepalOSINT API client for semantic and unified search."""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_OSINT_BASE_URL = "https://nepalosint.com/api/v1"
DEFAULT_TIMEOUT_SECONDS = 8


def _base_url(base_url: str | None = None) -> str:
    return str(base_url or DEFAULT_OSINT_BASE_URL).rstrip("/")


def semantic_story_search(
    query: str,
    *,
    hours: int = 720,
    top_k: int = 10,
    min_similarity: float = 0.45,
    base_url: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    payload = {
        "query": str(query or "").strip(),
        "hours": int(hours),
        "top_k": int(top_k),
        "min_similarity": float(min_similarity),
    }
    try:
        response = requests.post(
            f"{_base_url(base_url)}/embeddings/search",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = dict(response.json() or {})
    except Exception as exc:
        return {
            "query": payload["query"],
            "results": [],
            "total_found": 0,
            "error": str(exc),
        }
    data.setdefault("query", payload["query"])
    data["results"] = list(data.get("results") or [])
    data["total_found"] = int(data.get("total_found") or len(data["results"]))
    return data


def unified_search(
    query: str,
    *,
    limit: int = 10,
    election_year: int | None = None,
    base_url: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "q": str(query or "").strip(),
        "limit": int(limit),
    }
    if election_year is not None:
        params["election_year"] = int(election_year)
    try:
        response = requests.get(
            f"{_base_url(base_url)}/search/unified",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        data = dict(response.json() or {})
    except Exception as exc:
        return {
            "query": params["q"],
            "total": 0,
            "categories": {},
            "error": str(exc),
        }
    categories = dict(data.get("categories") or {})
    for name, payload in list(categories.items()):
        item_block = dict(payload or {})
        item_block["items"] = list(item_block.get("items") or [])
        item_block["total"] = int(item_block.get("total") or len(item_block["items"]))
        categories[name] = item_block
    data.setdefault("query", params["q"])
    data["total"] = int(data.get("total") or 0)
    data["categories"] = categories
    return data


def related_stories(
    story_id: str,
    *,
    top_k: int = 8,
    min_similarity: float = 0.55,
    hours: int = 8760,
    base_url: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    clean_id = str(story_id or "").strip()
    if not clean_id:
        return {"source_story_id": "", "similar_stories": [], "total_found": 0}
    try:
        response = requests.get(
            f"{_base_url(base_url)}/stories/{clean_id}/related",
            params={
                "top_k": int(top_k),
                "min_similarity": float(min_similarity),
                "hours": int(hours),
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = dict(response.json() or {})
    except Exception as exc:
        return {
            "source_story_id": clean_id,
            "similar_stories": [],
            "total_found": 0,
            "error": str(exc),
        }
    data.setdefault("source_story_id", clean_id)
    data["similar_stories"] = list(data.get("similar_stories") or [])
    data["total_found"] = int(data.get("total_found") or len(data["similar_stories"]))
    return data


def symbol_intelligence(
    query: str,
    *,
    hours: int = 720,
    top_k: int = 6,
    min_similarity: float = 0.45,
    base_url: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    semantic = semantic_story_search(
        query,
        hours=hours,
        top_k=top_k,
        min_similarity=min_similarity,
        base_url=base_url,
        timeout=timeout,
    )
    unified = unified_search(
        query,
        limit=max(int(top_k), 6),
        base_url=base_url,
        timeout=timeout,
    )
    categories = dict(unified.get("categories") or {})
    story_items = list(dict(categories.get("stories") or {}).get("items") or [])
    social_items = list(dict(categories.get("social_signals") or {}).get("items") or [])

    lead_story_id = ""
    if story_items:
        lead_story_id = str(story_items[0].get("id") or "").strip()
    elif semantic.get("results"):
        lead_story_id = str(semantic["results"][0].get("story_id") or "").strip()

    related = related_stories(
        lead_story_id,
        top_k=min(max(int(top_k), 3), 8),
        base_url=base_url,
        timeout=timeout,
    ) if lead_story_id else {"source_story_id": "", "similar_stories": [], "total_found": 0}

    return {
        "query": str(query or "").strip(),
        "semantic": semantic,
        "unified": unified,
        "related": related,
        "lead_story_id": lead_story_id,
        "story_items": story_items,
        "social_items": social_items,
        "related_items": list(related.get("similar_stories") or []),
        "story_count": len(story_items) or int(semantic.get("total_found") or 0),
        "social_count": len(social_items),
        "related_count": int(related.get("total_found") or 0),
    }
