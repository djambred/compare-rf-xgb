from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from youtube_comment_downloader import SORT_BY_RECENT, YoutubeCommentDownloader


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)

    if parsed.netloc in {"youtu.be"}:
        return parsed.path.strip("/")

    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query and query["v"]:
            return query["v"][0]

        match = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", parsed.path)
        if match:
            return match.group(1)

        match = re.search(r"/live/([A-Za-z0-9_-]{6,})", parsed.path)
        if match:
            return match.group(1)

    raise ValueError("URL YouTube tidak valid atau video id tidak ditemukan.")


def scrape_comments(url: str, max_items: int = 100) -> list[str]:
    downloader = YoutubeCommentDownloader()
    generator = downloader.get_comments_from_url(url, sort_by=SORT_BY_RECENT)

    comments: list[str] = []
    for item in generator:
        text = str(item.get("text", "")).strip()
        if text:
            comments.append(text)
        if len(comments) >= max_items:
            break

    return comments


def scrape_live_chat(url: str, max_items: int = 100) -> list[str]:
    try:
        import pytchat
    except Exception as err:
        raise RuntimeError("Library pytchat belum terpasang.") from err

    video_id = extract_video_id(url)
    chat = pytchat.create(video_id=video_id)

    comments: list[str] = []
    empty_round = 0

    while chat.is_alive() and len(comments) < max_items:
        data = chat.get()
        items = data.sync_items()

        if not items:
            empty_round += 1
            if empty_round >= 12:
                break
            continue

        empty_round = 0
        for item in items:
            message = str(getattr(item, "message", "")).strip()
            if message:
                comments.append(message)
            if len(comments) >= max_items:
                break

    chat.terminate()
    return comments


def scrape_youtube(url: str, mode: str = "auto", max_items: int = 100) -> tuple[str, list[str]]:
    selected_mode = mode.lower().strip()

    if selected_mode not in {"auto", "live_chat", "comments"}:
        raise ValueError("Mode harus salah satu dari: auto, live_chat, comments")

    if selected_mode == "comments":
        return "comments", scrape_comments(url, max_items=max_items)

    if selected_mode == "live_chat":
        return "live_chat", scrape_live_chat(url, max_items=max_items)

    try:
        live_chat_comments = scrape_live_chat(url, max_items=max_items)
        if live_chat_comments:
            return "live_chat", live_chat_comments
    except Exception:
        pass

    return "comments", scrape_comments(url, max_items=max_items)
