"""Parse raw JSON exports from the collection snippet into tidy tables."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VIEW_KEYS = ("play_count", "ig_play_count", "view_count", "video_view_count", "video_play_count", "clips_play_count")


def _first(d: dict, keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def _nested_count(d: dict, key: str):
    v = d.get(key)
    if isinstance(v, dict):
        return v.get("count")
    return None


def _caption(item: dict) -> str:
    cap = item.get("caption")
    if isinstance(cap, dict):
        return cap.get("text") or ""
    if isinstance(cap, str):
        return cap
    edges = (item.get("edge_media_to_caption") or {}).get("edges") or []
    if edges:
        return (edges[0].get("node") or {}).get("text") or ""
    return ""


IG_ID_EPOCH_MS = 1314220021721


def _timestamp_from_pk(item: dict):
    pk = str(item.get("pk") or item.get("id") or "").split("_")[0]
    if not pk.isdigit():
        return None
    return datetime.fromtimestamp(((int(pk) >> 23) + IG_ID_EPOCH_MS) / 1000.0, tz=timezone.utc)


def _timestamp(item: dict):
    ts = _first(item, ("taken_at", "taken_at_timestamp", "device_timestamp"))
    if ts is None:
        return _timestamp_from_pk(item)
    ts = float(ts)
    if ts > 1e12:
        ts /= 1000.0
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def parse_item(item: dict, fallback_user: str, collected_at: datetime) -> dict | None:
    code = item.get("code") or item.get("shortcode")
    if not code:
        return None
    user = item.get("user") or item.get("owner") or {}
    versions = item.get("video_versions") or []
    likes = item.get("like_count")
    if likes is None:
        likes = _nested_count(item, "edge_liked_by") or _nested_count(item, "edge_media_preview_like")
    comments = item.get("comment_count")
    if comments is None:
        comments = _nested_count(item, "edge_media_to_comment")
    is_video = item.get("media_type") == 2 or item.get("product_type") == "clips" or bool(versions) or item.get("is_video")
    return {
        "reel_id": str(code),
        "media_pk": str(item.get("pk") or item.get("id") or ""),
        "username": (user.get("username") or fallback_user or "").lower(),
        "posted_at": _timestamp(item),
        "collected_at": collected_at,
        "views": _first(item, VIEW_KEYS),
        "likes": likes,
        "comments": comments,
        "duration": item.get("video_duration"),
        "caption": _caption(item),
        "video_url": versions[0].get("url") if versions else item.get("video_url"),
        "is_video": bool(is_video),
        "is_collab": bool(item.get("coauthor_producers")),
        "is_paid": bool(item.get("is_paid_partnership")),
        "followers_at_collection": user.get("follower_count"),
        "audio_type": "original" if (item.get("clips_metadata") or {}).get("original_sound_info") else (
            "licensed" if (item.get("clips_metadata") or {}).get("music_info") else None),
    }


def _richness(r: dict) -> int:
    return sum(v not in (None, "", False) for v in r.values())


def load_raw(raw_dir: Path) -> pd.DataFrame:
    rows = {}
    for f in sorted(Path(raw_dir).glob("*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        items = payload.get("items", payload if isinstance(payload, list) else [])
        user = payload.get("username") if isinstance(payload, dict) else None
        user = user or f.stem.split("_reels_")[0]
        collected = payload.get("collected_at") if isinstance(payload, dict) else None
        collected_at = (datetime.fromisoformat(collected.replace("Z", "+00:00")) if collected
                        else datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc))
        for item in items:
            r = parse_item(item, user, collected_at)
            # The page can also load suggested reels from other accounts; keep only the profile's own.
            if r is None or (user and r["username"] and r["username"] != user.lower()):
                continue
            prev = rows.get(r["reel_id"])
            if prev is None or _richness(r) > _richness(prev):
                rows[r["reel_id"]] = r
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    df = df[df["is_video"] & df["views"].notna() & df["posted_at"].notna()].copy()
    for c in ("views", "likes", "comments", "duration", "followers_at_collection"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["username", "posted_at"]).reset_index(drop=True)


def build_tables(raw_dir: Path, creators_csv: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reels = load_raw(raw_dir)
    sectors = pd.read_csv(creators_csv) if Path(creators_csv).exists() else pd.DataFrame(columns=["username", "sector"])
    sectors["username"] = sectors["username"].str.lower().str.lstrip("@")
    reels = reels.merge(sectors[["username", "sector"]], on="username", how="left")
    creators = (reels.groupby("username")
                .agg(sector=("sector", "first"), n_reels=("reel_id", "size"),
                     followers=("followers_at_collection", "max"),
                     first_post=("posted_at", "min"), last_post=("posted_at", "max"))
                .reset_index())
    out_dir.mkdir(parents=True, exist_ok=True)
    reels.to_parquet(out_dir / "reels.parquet", index=False)
    creators.to_parquet(out_dir / "creators.parquet", index=False)
    return reels, creators
