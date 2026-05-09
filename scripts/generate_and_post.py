#!/usr/bin/env python3
"""Collect global tech + AI news and post a Mongolian summary to a Facebook page."""

from __future__ import annotations

import html
import hashlib
import json
import os
import re
import sys
import time
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


MEDIA_NS = {"media": "http://search.yahoo.com/mrss/"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def clean_html_text(value: str) -> str:
    if not value:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    unescaped = html.unescape(without_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def looks_like_image_url(url: str) -> bool:
    if not url:
        return False
    path = urllib.parse.urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS)


def extract_image_from_html(raw: str) -> str:
    if not raw:
        return ""
    match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', raw, re.IGNORECASE)
    if not match:
        return ""
    return html.unescape(match.group(1).strip())


def extract_rss_image(node, raw_description: str = "") -> str:
    """Find a usable image URL inside an RSS 2.0 <item> or Atom <entry> node."""
    for mc in node.findall("media:content", MEDIA_NS):
        url = (mc.attrib.get("url") or "").strip()
        medium = (mc.attrib.get("medium") or "").lower()
        type_attr = (mc.attrib.get("type") or "").lower()
        if url and (medium == "image" or type_attr.startswith("image/") or looks_like_image_url(url)):
            return url

    for mt in node.findall("media:thumbnail", MEDIA_NS):
        url = (mt.attrib.get("url") or "").strip()
        if url:
            return url

    for enc in node.findall("enclosure"):
        url = (enc.attrib.get("url") or "").strip()
        type_attr = (enc.attrib.get("type") or "").lower()
        if url and (type_attr.startswith("image/") or looks_like_image_url(url)):
            return url

    # Atom enclosure variant: <link rel="enclosure" href="..." type="image/*"/>
    atom_ns = {"atom": "http://www.w3.org/2005/Atom"}
    for ln in node.findall("atom:link[@rel='enclosure']", atom_ns):
        href = (ln.attrib.get("href") or "").strip()
        type_attr = (ln.attrib.get("type") or "").lower()
        if href and (type_attr.startswith("image/") or looks_like_image_url(href)):
            return href

    return extract_image_from_html(raw_description)


def urlopen_with_retry(
    req: urllib.request.Request,
    timeout_sec: int,
    label: str,
):
    retries = max(1, int(os.getenv("HTTP_RETRIES", "3")))
    backoff_sec = max(1, int(os.getenv("HTTP_RETRY_BACKOFF_SEC", "2")))
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            return urllib.request.urlopen(req, timeout=timeout_sec)
        except Exception as exc:
            last_exc = exc
            reason = exc
            if isinstance(exc, urllib.error.URLError):
                reason = exc.reason
            print(f"[WARN] {label} failed ({attempt}/{retries}): {reason}")
            if attempt < retries:
                time.sleep(backoff_sec * attempt)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed with unknown error")


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
        with urlopen_with_retry(req, timeout_sec, "translate.googleapis.com request") as response:
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


def resolve_ai_provider() -> str:
    provider = os.getenv("AI_PROVIDER", "auto").strip().lower()
    if provider in {"openai", "gemini"}:
        return provider
    if os.getenv("GEMINI_API_KEY", "").strip():
        return "gemini"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    return "none"


def rewrite_json_array_with_openai(
    system_prompt: str,
    user_prompt: str,
    expected_len: int,
    timeout_sec: int,
    label: str,
) -> List[str] | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip() or "gpt-5-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    url = f"{base_url}/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # gpt-5* chat-completions currently accepts only default temperature.
    if not model.lower().startswith("gpt-5"):
        payload["temperature"] = 0.2

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
        with urlopen_with_retry(req, timeout_sec, f"OpenAI {label} request") as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        parsed = try_parse_json_array(content)
        if parsed and len(parsed) == expected_len:
            return [x.strip() for x in parsed]
    except Exception:
        return None

    return None


def rewrite_json_array_with_gemini(
    system_prompt: str,
    user_prompt: str,
    expected_len: int,
    timeout_sec: int,
    label: str,
) -> List[str] | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview").strip() or "gemini-3.1-flash-lite-preview"
    query = urllib.parse.urlencode({"key": api_key})
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?{query}"

    prompt = (
        f"{system_prompt}\n\n{user_prompt}\n\n"
        f"Return ONLY a JSON array of strings with exactly {expected_len} items."
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen_with_retry(req, timeout_sec, f"Gemini {label} request") as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        if isinstance(data, dict) and "error" in data:
            return None
        parts = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        content = "".join(
            str(p.get("text", "")) for p in parts if isinstance(p, dict)
        ).strip()
        parsed = try_parse_json_array(content)
        if parsed and len(parsed) == expected_len:
            return [x.strip() for x in parsed]
    except Exception:
        return None

    return None


def rewrite_json_array_with_ai(
    system_prompt: str,
    user_prompt: str,
    expected_len: int,
    timeout_sec: int,
    label: str,
) -> List[str] | None:
    provider = resolve_ai_provider()

    if provider == "gemini":
        out = rewrite_json_array_with_gemini(system_prompt, user_prompt, expected_len, timeout_sec, label)
        if out is not None:
            return out
        return rewrite_json_array_with_openai(system_prompt, user_prompt, expected_len, timeout_sec, label)

    if provider == "openai":
        out = rewrite_json_array_with_openai(system_prompt, user_prompt, expected_len, timeout_sec, label)
        if out is not None:
            return out
        return rewrite_json_array_with_gemini(system_prompt, user_prompt, expected_len, timeout_sec, label)

    return None


def rewrite_headlines_with_ai(headlines: List[str], timeout_sec: int = 30) -> List[str] | None:
    if not headlines:
        return None

    system_prompt = (
        "You are a Mongolian tech news editor. Rewrite each input headline into natural,"
        " concise Mongolian suitable for Facebook news posts. Keep names of people,"
        " companies, and products accurate. Do not add facts. Return only a JSON array"
        " of strings in the same order and same length as input."
    )
    user_prompt = "Headlines:\n" + "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))

    parsed = rewrite_json_array_with_ai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_len=len(headlines),
        timeout_sec=timeout_sec,
        label="headline rewrite",
    )
    if parsed and len(parsed) == len(headlines):
        return [x or headlines[i] for i, x in enumerate(parsed)]
    return None


def summarize_article_with_ai(
    title: str,
    article_text: str,
    source: str,
    timeout_sec: int = 60,
) -> tuple[str, str]:
    """Send full article to AI and get (short, detailed) Mongolian summaries."""
    if not article_text.strip():
        return ("", "")

    system_prompt = (
        "You are a Mongolian tech news editor writing for a Facebook audience. Read the "
        "provided English article and produce TWO Mongolian versions. "
        "First — a SHORT 1-2 sentence hook capturing the single most important point in "
        "engaging, clear Mongolian. "
        "Second — a DETAILED 7-12 sentence breakdown covering: what happened in detail, "
        "the companies, products, and people involved, relevant background and context, "
        "important specifics (numbers, dates, quotes), why it matters and broader implications, "
        "and any next steps or what to watch. "
        "Both versions must be in natural, engaging Mongolian — not stiff word-for-word translation. "
        "Stay strictly factual — use only what is in the article. Do not invent facts. "
        "Do not include the headline, source name, hashtags, or links in either version. "
        "Return ONLY a JSON array with EXACTLY two strings: [short_version, detailed_version]."
    )

    snippet = article_text.strip()
    if len(snippet) > 9000:
        snippet = snippet[:9000].rsplit(" ", 1)[0] + "..."

    user_prompt = (
        f"Title: {title}\n"
        f"Source: {source}\n\n"
        f"Article body:\n{snippet}"
    )

    parsed = rewrite_json_array_with_ai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_len=2,
        timeout_sec=timeout_sec,
        label="article short+detailed summary",
    )
    if parsed and len(parsed) == 2:
        return (parsed[0].strip(), parsed[1].strip())
    return ("", "")


def rewrite_briefs_with_ai(items: List[Dict[str, str]], timeout_sec: int = 35) -> List[str] | None:
    if not items:
        return None

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
        "You are a Mongolian tech news editor. For each input item, write a detailed "
        "5-7 sentence Mongolian brief suitable for Facebook. The brief should cover: "
        "(1) what happened in detail, (2) the key parties/companies/people involved, "
        "(3) the relevant context or background, (4) why it matters and the broader impact, "
        "(5) any notable next steps or implications. Write in natural, engaging Mongolian. "
        "Keep it factual. Use only provided headline/description — do not invent facts. "
        "Return only a JSON array of strings in the same order and same length as input."
    )
    user_prompt = "Items JSON:\n" + json.dumps(compact_items, ensure_ascii=False)

    return rewrite_json_array_with_ai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_len=len(items),
        timeout_sec=timeout_sec,
        label="brief rewrite",
    )


def build_mongolian_reader_link(url: str) -> str:
    quoted = urllib.parse.quote(url, safe="")
    return f"https://translate.google.com/translate?hl=mn&sl=auto&tl=mn&u={quoted}"


def resolve_final_url(url: str, timeout_sec: int = 12) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FBMongoliaAutoPost/1.0)"},
    )
    try:
        with urlopen_with_retry(req, timeout_sec, f"Resolve final URL: {url}") as response:
            final_url = (response.geturl() or "").strip()
            return final_url or url
    except Exception:
        return url


IMAGE_META_PATTERNS = [
    r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::secure_url)?["\']',
    r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
]


def _extract_meta_image(head: str) -> str:
    for pat in IMAGE_META_PATTERNS:
        match = re.search(pat, head, re.IGNORECASE)
        if match:
            candidate = html.unescape(match.group(1).strip())
            if candidate.startswith("//"):
                candidate = "https:" + candidate
            if candidate.startswith("http"):
                return candidate
    return ""


def _extract_article_body(html_body: str, max_chars: int = 10000) -> str:
    cleaned = re.sub(r"<(script|style|noscript)[\s\S]*?</\1>", " ", html_body, flags=re.IGNORECASE)

    scope = ""
    art_match = re.search(r"<article[\s\S]*?</article>", cleaned, re.IGNORECASE)
    if art_match:
        scope = art_match.group(0)
    if not scope:
        main_match = re.search(r"<main[\s\S]*?</main>", cleaned, re.IGNORECASE)
        if main_match:
            scope = main_match.group(0)
    if not scope:
        scope = cleaned

    paragraphs = re.findall(r"<p[^>]*>([\s\S]*?)</p>", scope, re.IGNORECASE)
    parts: List[str] = []
    for raw in paragraphs:
        text = clean_html_text(raw)
        if len(text) > 40:
            parts.append(text)

    body = "\n\n".join(parts)
    if len(body) > max_chars:
        body = body[:max_chars].rsplit(" ", 1)[0]
    return body.strip()


def fetch_article_meta(url: str, timeout_sec: int = 15) -> Dict[str, str]:
    """Fetch article HTML and return {'image_url': ..., 'text': ...}."""
    blank = {"image_url": "", "text": ""}
    if not url:
        return blank

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FBMongoliaAutoPost/1.0)"},
    )
    try:
        with urlopen_with_retry(req, timeout_sec, f"Fetch article HTML: {url}") as response:
            body = response.read(2_000_000).decode("utf-8", errors="replace")
    except Exception:
        return blank

    head = body.split("</head>", 1)[0]
    return {
        "image_url": _extract_meta_image(head),
        "text": _extract_article_body(body),
    }


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

    with urlopen_with_retry(req, timeout_sec, f"Fetch feed: {url}") as response:
        content = response.read()

    root = ET.fromstring(content)
    items: List[Dict[str, Any]] = []

    # RSS 2.0
    channel_title = (root.findtext("./channel/title") or "").strip()
    for node in root.findall(".//item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        raw_description = (node.findtext("description") or "").strip()
        raw_encoded = (node.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or "").strip()
        description = clean_html_text(raw_description)
        if not description:
            description = clean_html_text(raw_encoded)
        pub_date = parse_datetime_maybe(node.findtext("pubDate") or "")

        image_url = extract_rss_image(node, raw_encoded or raw_description)

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
                    "image_url": image_url,
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
        image_url = extract_rss_image(entry, content_text or summary)

        if title and link:
            items.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "source": source,
                    "pub_date": pub_date,
                    "image_url": image_url,
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


def build_item_post(item: Dict[str, Any]) -> str:
    source = (item.get("source") or "").strip() or source_from_link(item.get("link", ""))
    original_title = clean_title_for_translation(item.get("title", ""), source)

    translated_title = ""
    rewrites = rewrite_headlines_with_ai([original_title])
    if rewrites and rewrites[0].strip():
        translated_title = rewrites[0].strip()
    if not translated_title:
        translated_title = translate_to_mongolian(original_title) or original_title

    article_text = str(item.get("article_text") or "").strip()
    rss_description = str(item.get("description") or "").strip()

    short_brief = ""
    detailed_brief = ""

    if article_text:
        short_brief, detailed_brief = summarize_article_with_ai(original_title, article_text, source)

    if not detailed_brief:
        briefs = rewrite_briefs_with_ai(
            [
                {
                    "title": original_title,
                    "description": article_text or rss_description,
                    "source": source,
                }
            ]
        )
        if briefs and briefs[0].strip():
            detailed_brief = briefs[0].strip()

    if not detailed_brief:
        fallback_text = article_text[:1500] if article_text else rss_description
        if fallback_text:
            detailed_brief = translate_to_mongolian(fallback_text)

    # If we have a detailed but no short, derive the short from the first sentence.
    if detailed_brief and not short_brief:
        match = re.match(r"\s*(.+?[\.!\?])(\s|$)", detailed_brief)
        if match:
            short_brief = match.group(1).strip()

    if len(detailed_brief) > 1800:
        detailed_brief = detailed_brief[:1797].rstrip() + "..."
    if len(short_brief) > 280:
        short_brief = short_brief[:277].rstrip() + "..."

    lines: List[str] = [translated_title, ""]
    if short_brief:
        lines.append(short_brief)
        lines.append("")
    if detailed_brief and detailed_brief != short_brief:
        lines.append(detailed_brief)
        lines.append("")
    lines.append(f"Эх сурвалж: {source}")
    lines.append("")
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
    with urlopen_with_retry(req, 30, "Facebook post request") as response:
        body = response.read().decode("utf-8")

    return json.loads(body)


def post_photo_to_facebook(
    page_id: str,
    page_access_token: str,
    message: str,
    image_url: str,
) -> Dict[str, Any]:
    url = f"https://graph.facebook.com/{page_id}/photos"
    payload = urllib.parse.urlencode(
        {
            "url": image_url,
            "message": message,
            "published": "true",
            "access_token": page_access_token,
        }
    ).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    with urlopen_with_retry(req, 60, "Facebook photo post request") as response:
        body = response.read().decode("utf-8")

    return json.loads(body)


def resolve_page_token_from_user_token(page_id: str, token: str) -> str:
    """If token is a user token, resolve and return the matching page token."""
    if not page_id or not token:
        return token

    url = "https://graph.facebook.com/v21.0/me/accounts"
    query = urllib.parse.urlencode(
        {
            "fields": "id,name,tasks,access_token",
            "access_token": token,
        }
    )
    req = urllib.request.Request(f"{url}?{query}", method="GET")

    try:
        with urlopen_with_retry(req, 30, "Resolve page token via /me/accounts") as response:
            body = response.read().decode("utf-8")
        data = json.loads(body)
    except Exception as exc:
        print(f"[WARN] Could not resolve page token from user token: {exc}")
        return token

    if isinstance(data, dict) and "error" in data:
        return token

    for row in data.get("data", []):
        if str(row.get("id", "")).strip() != page_id:
            continue
        page_token = str(row.get("access_token", "")).strip()
        if not page_token:
            continue
        tasks = [str(x).strip() for x in row.get("tasks", []) if str(x).strip()]
        if "CREATE_CONTENT" not in tasks:
            print("[WARN] Page token resolved, but CREATE_CONTENT task is missing.")
        else:
            print("[INFO] Using page token resolved from user token.")
        return page_token

    return token


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

        meta = fetch_article_meta(final_url)
        item["article_text"] = meta.get("text", "")
        if not (item.get("image_url") or "").strip():
            item["image_url"] = meta.get("image_url", "")

    if not dry_run and (not page_id or not page_access_token):
        print("[ERROR] Missing FACEBOOK_PAGE_ID or FACEBOOK_PAGE_ACCESS_TOKEN")
        return 1

    effective_token = page_access_token
    if not dry_run:
        effective_token = resolve_page_token_from_user_token(page_id, page_access_token)

    success_count = 0
    failure_count = 0

    for item in selected:
        message = build_item_post(item)
        image_url = (item.get("image_url") or "").strip()

        if dry_run:
            print(f"[DRY RUN] image_url={image_url or '(none)'}")
            print(message)
            print("---")
            success_count += 1
            continue

        result: Dict[str, Any] | None = None
        used_photo = False
        if image_url:
            try:
                result = post_photo_to_facebook(page_id, effective_token, message, image_url)
                used_photo = True
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                print(f"[WARN] Photo post failed, falling back to text. image_url={image_url}")
                print(details)
            except Exception as exc:
                print(f"[WARN] Photo post failed, falling back to text: {exc}")

        if result is None:
            try:
                result = post_to_facebook(page_id, effective_token, message)
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                print("[ERROR] Facebook API error")
                print(details)
                failure_count += 1
                continue
            except Exception as exc:
                print(f"[ERROR] Failed to post to Facebook: {exc}")
                failure_count += 1
                continue

        post_id = result.get("id") or result.get("post_id") or "unknown"
        kind = "photo" if used_photo else "text"
        print(f"[OK] Posted ({kind}) post_id={post_id}")
        success_count += 1
        posted[item_hash(item["link"])] = datetime.now(timezone.utc).isoformat()
        save_state(prune_state(posted))

    save_state(prune_state(posted))
    print(f"[INFO] Posted {success_count} items, {failure_count} failed.")
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
