#!/usr/bin/env python3
"""Collect global tech + AI news and post a Mongolian summary to a Facebook page."""

from __future__ import annotations

import html
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / ".state"
STATE_FILE = STATE_DIR / "posted_items.json"
ENV_FILE = ROOT / ".env"

# Direct publisher feeds for global tech and AI.
FEEDS = [
    "https://techcrunch.com/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://www.theverge.com/rss/tech/index.xml",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed/",
    "https://www.artificialintelligence-news.com/feed/",
    "https://www.wired.com/feed/tag/ai/latest/rss",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
]

POSITIVE_KEYWORDS = [
    "ai",
    "artificial intelligence",
    "machine learning",
    "llm",
    "agentic",
    "openai",
    "anthropic",
    "copilot",
    "gemini",
    "nvidia",
    "robot",
    "robotics",
    "chip",
    "semiconductor",
    "quantum",
    "cybersecurity",
    "startup",
    "data center",
    "cloud",
    "autonomous",
    "model",
    "algorithm",
    "technology",
    "tech",
]

NEGATIVE_KEYWORDS = [
    "deal",
    "discount",
    "sale",
    "coupon",
    "buy now",
    "hands-on",
    "review",
    "charger",
    "headphones",
]


def load_env_file() -> None:
    if not ENV_FILE.exists():
        return

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def clean_html_text(value: str) -> str:
    if not value:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    unescaped = html.unescape(without_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def translate_to_mongolian(text: str, timeout_sec: int = 15) -> str:
    """Translate text to Mongolian using a lightweight public endpoint."""
    if not text.strip():
        return text

    params = urllib.parse.urlencode(
        {
            "client": "gtx",
            "sl": "auto",
            "tl": "mn",
            "dt": "t",
            "q": text,
        }
    )
    url = f"https://translate.googleapis.com/translate_a/single?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FBMongoliaAutoPost/1.0)",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
        payload = json.loads(body)
        parts = payload[0] if isinstance(payload, list) and payload else []
        translated = "".join(
            p[0] for p in parts if isinstance(p, list) and p and isinstance(p[0], str)
        ).strip()
        return translated or text
    except Exception:
        return text


def try_parse_json_array(text: str) -> List[str] | None:
    raw = text.strip()
    if not raw:
        return None

    # Best case: response is already a valid JSON array.
    try:
        data = json.loads(raw)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        pass

    # Fallback: extract first JSON array-looking segment.
    left = raw.find("[")
    right = raw.rfind("]")
    if left == -1 or right == -1 or right <= left:
        return None

    snippet = raw[left : right + 1]
    try:
        data = json.loads(snippet)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        return None

    return None


def rewrite_headlines_with_ai(headlines: List[str], timeout_sec: int = 30) -> List[str] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not headlines:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    url = f"{base_url}/chat/completions"

    system_prompt = (
        "You are a Mongolian tech news editor. Rewrite each input headline into natural,"
        " concise Mongolian suitable for Facebook news posts. Keep names of people,"
        " companies, and products accurate. Do not add facts. Return only a JSON array"
        " of strings in the same order and same length as input."
    )
    user_prompt = "Headlines:\n" + "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        parsed = try_parse_json_array(content)
        if parsed and len(parsed) == len(headlines):
            return [x.strip() or headlines[i] for i, x in enumerate(parsed)]
    except Exception:
        return None

    return None


def rewrite_briefs_with_ai(items: List[Dict[str, str]], timeout_sec: int = 35) -> List[str] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or not items:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    url = f"{base_url}/chat/completions"

    compact_items: List[Dict[str, str]] = []
    for item in items:
        compact_items.append(
            {
                "title": item.get("title", "").strip(),
                "description": item.get("description", "").strip(),
                "source": item.get("source", "").strip(),
            }
        )

    system_prompt = (
        "You are a Mongolian tech news editor. For each input item, write a 2-3 sentence "
        "Mongolian brief suitable for Facebook. The brief should include: "
        "(1) what happened, (2) why it matters. Keep it factual. Use only provided "
        "headline/description. Do not invent facts. Return only a JSON array of strings "
        "in the same order and same length as input."
    )
    user_prompt = "Items JSON:\n" + json.dumps(compact_items, ensure_ascii=False)

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        parsed = try_parse_json_array(content)
        if parsed and len(parsed) == len(items):
            return [x.strip() for x in parsed]
    except Exception:
        return None

    return None


def build_mongolian_reader_link(url: str) -> str:
    quoted = urllib.parse.quote(url, safe="")
    return f"https://translate.google.com/translate?hl=mn&sl=auto&tl=mn&u={quoted}"


def resolve_final_url(url: str, timeout_sec: int = 12) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FBMongoliaAutoPost/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            final_url = (response.geturl() or "").strip()
            return final_url or url
    except Exception:
        return url


def clean_title_for_translation(title: str, source: str) -> str:
    cleaned = title.strip()
    source = source.strip()
    if source:
        suffix = f" - {source}"
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned


def parse_datetime_maybe(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None

    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Atom feeds often use ISO 8601.
    try:
        normalized = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def source_from_link(link: str) -> str:
    host = urllib.parse.urlparse(link).netloc.lower()
    host = host.removeprefix("www.")
    return host or "Unknown Source"


def normalize_source_name(source: str, link: str) -> str:
    cleaned = (source or "").strip()
    if not cleaned:
        return source_from_link(link)
    lower = cleaned.lower()
    if lower in {"latest news", "feed", "rss"}:
        return source_from_link(link)
    if lower.startswith("feed:"):
        return source_from_link(link)
    return cleaned


def fetch_rss_items(url: str, timeout_sec: int = 20) -> List[Dict[str, Any]]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FBMongoliaAutoPost/1.0)",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout_sec) as response:
        content = response.read()

    root = ET.fromstring(content)
    items: List[Dict[str, Any]] = []

    # RSS 2.0
    channel_title = (root.findtext("./channel/title") or "").strip()
    for node in root.findall(".//item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        description = clean_html_text((node.findtext("description") or "").strip())
        if not description:
            description = clean_html_text((node.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or "").strip())
        pub_date = parse_datetime_maybe(node.findtext("pubDate") or "")

        source = ""
        source_node = node.find("source")
        if source_node is not None and source_node.text:
            source = source_node.text.strip()
        source = normalize_source_name(source or channel_title, link)

        if title and link:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "source": source,
                    "pub_date": pub_date,
                }
            )

    if items:
        return items

    # Atom
    atom_ns = {"atom": "http://www.w3.org/2005/Atom"}
    feed_title = (root.findtext("atom:title", default="", namespaces=atom_ns) or "").strip()
    for entry in root.findall("atom:entry", atom_ns):
        title = (entry.findtext("atom:title", default="", namespaces=atom_ns) or "").strip()

        link = ""
        link_node = entry.find("atom:link[@rel='alternate']", atom_ns)
        if link_node is None:
            link_node = entry.find("atom:link", atom_ns)
        if link_node is not None:
            link = (link_node.attrib.get("href") or "").strip()

        summary = (entry.findtext("atom:summary", default="", namespaces=atom_ns) or "").strip()
        content_text = (entry.findtext("atom:content", default="", namespaces=atom_ns) or "").strip()
        description = clean_html_text(summary or content_text)
        pub_date = parse_datetime_maybe(
            entry.findtext("atom:published", default="", namespaces=atom_ns)
            or entry.findtext("atom:updated", default="", namespaces=atom_ns)
            or ""
        )
        source = normalize_source_name(feed_title, link)

        if title and link:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "source": source,
                    "pub_date": pub_date,
                }
            )

    return items


def normalize_link(link: str) -> str:
    # Remove URL fragments and UTM-style params for better deduplication.
    parsed = urllib.parse.urlparse(link)
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    filtered_pairs = [
        (k, v)
        for k, v in query_pairs
        if not k.lower().startswith("utm_") and k.lower() not in {"oc", "ved"}
    ]
    normalized_query = urllib.parse.urlencode(filtered_pairs)
    normalized = parsed._replace(query=normalized_query, fragment="")
    return urllib.parse.urlunparse(normalized)


def item_hash(link: str) -> str:
    return hashlib.sha256(normalize_link(link).encode("utf-8")).hexdigest()


def load_state() -> Dict[str, str]:
    if not STATE_FILE.exists():
        return {}

    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass

    return {}


def save_state(data: Dict[str, str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dedupe_and_sort(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in items:
        h = item_hash(item["link"])
        if h in seen:
            continue
        seen.add(h)
        unique.append(item)

    def sort_key(row: Dict[str, Any]) -> datetime:
        return row["pub_date"] or datetime(1970, 1, 1, tzinfo=timezone.utc)

    unique.sort(key=sort_key, reverse=True)
    return unique


def relevance_score(item: Dict[str, Any]) -> int:
    text = f"{item.get('title', '')} {item.get('description', '')}".lower()
    score = 0
    for kw in POSITIVE_KEYWORDS:
        if kw in text:
            score += 1
    for kw in NEGATIVE_KEYWORDS:
        if kw in text:
            score -= 2
    return score


def pick_items(items: List[Dict[str, Any]], posted: Dict[str, str], max_items: int, max_age_hours: int) -> List[Dict[str, Any]]:
    now_utc = datetime.now(timezone.utc)
    age_limit = now_utc - timedelta(hours=max_age_hours)

    eligible: List[Dict[str, Any]] = []
    for item in items:
        h = item_hash(item["link"])
        if h in posted:
            continue

        pub_date = item["pub_date"]
        if pub_date is not None and pub_date < age_limit:
            continue

        eligible.append(item)

    ranked = sorted(eligible, key=lambda x: (relevance_score(x), x.get("pub_date") or datetime(1970, 1, 1, tzinfo=timezone.utc)), reverse=True)
    filtered = [x for x in ranked if relevance_score(x) >= 1][:max_items]

    # If relevance filter is too strict, gracefully fallback to most recent eligible items.
    if not filtered:
        filtered = eligible[:max_items]

    # Fallback: if no fresh items, still allow top unposted items.
    if not filtered:
        for item in items:
            h = item_hash(item["link"])
            if h in posted:
                continue
            filtered.append(item)
            if len(filtered) >= max_items:
                break

    return filtered


def build_post(items: List[Dict[str, Any]]) -> str:
    now_local = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"Дэлхийн шинэ технологи ба AI мэдээ ({now_local})",
        "",
        "Сүүлийн тойм мэдээнүүд:",
        "",
    ]

    source_and_titles: List[tuple[str, str]] = []
    for item in items:
        source = item["source"].strip() if item.get("source") else "Google News"
        source_and_titles.append((source, clean_title_for_translation(item["title"], source)))

    unique_titles: List[str] = []
    for _, title in source_and_titles:
        if title not in unique_titles:
            unique_titles.append(title)

    ai_map: Dict[str, str] = {}
    ai_rewrites = rewrite_headlines_with_ai(unique_titles)
    if ai_rewrites:
        ai_map = {title: ai_rewrites[i] for i, title in enumerate(unique_titles)}

    translation_cache: Dict[str, str] = {}
    brief_inputs: List[Dict[str, str]] = []
    for idx, item in enumerate(items):
        source, original_title = source_and_titles[idx]
        brief_inputs.append(
            {
                "title": original_title,
                "description": str(item.get("description", "")).strip(),
                "source": source,
            }
        )

    ai_briefs = rewrite_briefs_with_ai(brief_inputs)

    for i, item in enumerate(items, start=1):
        source, original_title = source_and_titles[i - 1]
        if original_title in ai_map:
            translated_title = ai_map[original_title]
        else:
            if original_title not in translation_cache:
                translation_cache[original_title] = translate_to_mongolian(original_title)
            translated_title = translation_cache[original_title]

        brief = ""
        if ai_briefs and len(ai_briefs) >= i and ai_briefs[i - 1].strip():
            brief = ai_briefs[i - 1].strip()
        else:
            desc = str(item.get("description", "")).strip()
            if desc:
                brief = translate_to_mongolian(desc)
            if not brief:
                brief = "Дэлгэрэнгүй мэдээллийг доорх линкээс уншина уу."
            if len(brief) > 420:
                brief = brief[:417].rstrip() + "..."
            if len(brief) < 120:
                brief = (
                    f"{brief} Энэ сэдэв нь технологи, AI-ийн чиг хандлагад нөлөөлөх боломжтой."
                ).strip()

        final_url = str(item.get("final_url") or item.get("link") or "").strip()
        mn_reader_url = str(item.get("mn_reader_url") or "").strip() or build_mongolian_reader_link(final_url)

        lines.append(f"{i}. {translated_title}")
        lines.append(f"Дэлгэрэнгүй: {brief}")
        lines.append(f"Эх сурвалж: {source}")
        lines.append(f"Монгол хэлээр унших: {mn_reader_url}")
        lines.append(f"Эх линк (англи): {final_url}")
        lines.append("")

    lines.append("Та аль мэдээг нь илүү дэлгэрэнгүй задлуулмаар байгаагаа коммент дээр бичээрэй.")
    lines.append("#ХиймэлОюун #Технологи #ШинэМэдээ #Дэлхий")
    return "\n".join(lines).strip()


def post_to_facebook(page_id: str, page_access_token: str, message: str) -> Dict[str, Any]:
    url = f"https://graph.facebook.com/{page_id}/feed"
    payload = urllib.parse.urlencode(
        {
            "message": message,
            "access_token": page_access_token,
        }
    ).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        body = response.read().decode("utf-8")

    return json.loads(body)


def prune_state(posted: Dict[str, str], keep_days: int = 14) -> Dict[str, str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    cleaned: Dict[str, str] = {}

    for key, timestamp in posted.items():
        try:
            dt = datetime.fromisoformat(timestamp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                cleaned[key] = dt.astimezone(timezone.utc).isoformat()
        except Exception:
            continue

    return cleaned


def main() -> int:
    load_env_file()

    max_items = int(os.getenv("MAX_ITEMS", "5"))
    max_age_hours = int(os.getenv("MAX_ITEM_AGE_HOURS", "12"))
    dry_run = os.getenv("DRY_RUN", "0").lower() in {"1", "true", "yes"}

    page_id = os.getenv("FACEBOOK_PAGE_ID", "").strip()
    page_access_token = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN", "").strip()

    all_items: List[Dict[str, Any]] = []
    for feed_url in FEEDS:
        try:
            all_items.extend(fetch_rss_items(feed_url))
        except Exception as exc:
            print(f"[WARN] Failed to read feed: {feed_url}")
            print(f"       {exc}")

    if not all_items:
        print("[INFO] No feed items found. Exiting.")
        return 0

    all_items = dedupe_and_sort(all_items)
    posted = prune_state(load_state())
    selected = pick_items(all_items, posted, max_items=max_items, max_age_hours=max_age_hours)

    if not selected:
        print("[INFO] No unposted items available. Exiting.")
        save_state(posted)
        return 0

    for item in selected:
        final_url = resolve_final_url(item["link"])
        item["final_url"] = final_url
        item["mn_reader_url"] = build_mongolian_reader_link(final_url)

    message = build_post(selected)

    if dry_run:
        print("[DRY RUN] Generated message:\n")
        print(message)
    else:
        if not page_id or not page_access_token:
            print("[ERROR] Missing FACEBOOK_PAGE_ID or FACEBOOK_PAGE_ACCESS_TOKEN")
            return 1

        try:
            result = post_to_facebook(page_id, page_access_token, message)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            print("[ERROR] Facebook API error")
            print(details)
            return 1
        except Exception as exc:
            print(f"[ERROR] Failed to post to Facebook: {exc}")
            return 1

        post_id = result.get("id", "unknown")
        print(f"[OK] Posted to Facebook. post_id={post_id}")

    now_iso = datetime.now(timezone.utc).isoformat()
    for item in selected:
        posted[item_hash(item["link"])] = now_iso

    save_state(prune_state(posted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
