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
    
    # Try to create chat object dengan error handling untuk NaN issue
    chat = None
    try:
        chat = pytchat.create(video_id=video_id, timeout=3)
    except (ValueError, TypeError) as err:
        if "NaN" in str(err) or "float" in str(err):
            raise ValueError(
                f"Pytchat error: {err}. "
                f"Ini adalah issue dengan pytchat library pada video tertentu. "
                f"Solusi: Coba gunakan mode COMMENTS atau mode BOTH, atau coba video lain."
            ) from err
        raise ValueError(
            f"Gagal akses live chat dari video. "
            f"Video mungkin: 1) belum live/sudah selesai, 2) live chat ditutup, 3) URL tidak valid. "
            f"Detail error: {type(err).__name__}: {err}"
        ) from err
    except Exception as err:
        raise ValueError(
            f"Gagal akses live chat dari video. "
            f"Video mungkin: 1) belum live/sudah selesai, 2) live chat ditutup, 3) URL tidak valid. "
            f"Detail error: {type(err).__name__}: {err}"
        ) from err

    try:
        # Check if chat is alive at the start
        if not chat.is_alive():
            raise ValueError(
                "Live chat tidak aktif. Video mungkin belum dimulai, sudah selesai, "
                "atau live chat disabled. Coba gunakan mode COMMENTS untuk video regular."
            )

        comments: list[str] = []
        empty_round = 0
        max_empty_rounds = 5  # Reduced untuk faster timeout
        attempt_count = 0
        max_attempts = 15  # Max attempts untuk prevent infinite loop

        # Get initial messages dengan short timeout
        try:
            initial_data = chat.get(timeout=1)  # Very short timeout for initial
            if initial_data:
                try:
                    items = initial_data.sync_items()
                    for item in items:
                        try:
                            message = str(getattr(item, "message", "")).strip()
                            if message and len(message) > 0:
                                comments.append(message)
                            if len(comments) >= max_items:
                                return comments
                        except Exception:
                            continue
                except (ValueError, TypeError) as e:
                    if "NaN" in str(e) or "float" in str(e):
                        # Don't fail here, continue to next fetch
                        empty_round += 1
                    else:
                        empty_round += 1
        except Exception:
            pass

        # Fetch more messages dengan strict timeout
        while len(comments) < max_items and attempt_count < max_attempts:
            attempt_count += 1
            
            try:
                if not chat.is_alive():
                    break

                # Short timeout to prevent hanging
                try:
                    data = chat.get(timeout=1)
                except (ValueError, TypeError) as e:
                    if "NaN" in str(e) or "float" in str(e):
                        empty_round += 1
                        if empty_round >= 2:
                            # Too many NaN errors, give up
                            break
                    empty_round += 1
                    if empty_round >= max_empty_rounds:
                        break
                    continue
                except Exception:
                    empty_round += 1
                    if empty_round >= max_empty_rounds:
                        break
                    continue

                if not data:
                    empty_round += 1
                    if empty_round >= max_empty_rounds:
                        break
                    continue

                try:
                    items = data.sync_items()
                except (ValueError, TypeError) as e:
                    if "NaN" in str(e) or "float" in str(e):
                        empty_round += 1
                        if empty_round >= 2:
                            break
                    else:
                        empty_round += 1
                    continue
                except Exception:
                    empty_round += 1
                    continue

                if not items:
                    empty_round += 1
                    if empty_round >= max_empty_rounds:
                        break
                    continue

                empty_round = 0
                for item in items:
                    try:
                        message = str(getattr(item, "message", "")).strip()
                        if message and len(message) > 0:
                            comments.append(message)
                        if len(comments) >= max_items:
                            return comments
                    except Exception:
                        continue

            except Exception:
                empty_round += 1
                if empty_round >= max_empty_rounds:
                    break
                continue

        if not comments:
            raise ValueError(
                "Tidak ada live chat message yang berhasil diambil. "
                "Kemungkinan: 1) Live stream belum/sudah berhenti, "
                "2) Sedikit aktivitas di chat, 3) Live chat disabled. "
                "Coba mode COMMENTS untuk scrape komentar regular saja."
            )

        return comments
        
    finally:
        if chat:
            try:
                chat.terminate()
            except Exception:
                pass


def scrape_youtube(url: str, mode: str = "auto", max_items: int = 100) -> tuple[str, list[str]]:
    selected_mode = mode.lower().strip()

    if selected_mode not in {"auto", "live_chat", "comments", "both"}:
        raise ValueError("Mode harus salah satu dari: auto, live_chat, comments, both")

    if selected_mode == "comments":
        return "comments", scrape_comments(url, max_items=max_items)

    if selected_mode == "live_chat":
        # For live_chat mode, try first, if fail then give clear error
        return "live_chat", scrape_live_chat(url, max_items=max_items)

    if selected_mode == "both":
        # Try to get both live chat and comments
        all_data = []
        source_info = []
        
        # Try live chat first
        try:
            live_chat_data = scrape_live_chat(url, max_items=max_items)
            all_data.extend(live_chat_data)
            source_info.append(f"live_chat({len(live_chat_data)})")
        except (ValueError, RuntimeError) as e:
            # Pytchat errors are expected for some videos - just skip
            pass
        except Exception:
            pass
        
        # Always try comments as fallback in BOTH mode
        try:
            comments_data = scrape_comments(url, max_items=max_items)
            all_data.extend(comments_data)
            source_info.append(f"comments({len(comments_data)})")
        except Exception:
            pass
        
        if not all_data:
            raise ValueError("Tidak bisa scrape live chat maupun comments dari URL ini")
        
        source = " + ".join(source_info)
        return source, all_data

    # Auto mode: try live_chat first with silent fallback to comments
    try:
        live_chat_comments = scrape_live_chat(url, max_items=max_items)
        if live_chat_comments:
            return "live_chat", live_chat_comments
    except (ValueError, RuntimeError):
        # Pytchat error or no live chat - silent fallback
        pass
    except Exception:
        pass

    # Fallback to comments
    comments_data = scrape_comments(url, max_items=max_items)
    if comments_data:
        return "comments", comments_data
    
    # If both fail, raise error
    raise ValueError("Tidak bisa scrape data dari URL ini (live chat dan comments gagal)")
