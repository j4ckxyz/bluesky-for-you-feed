from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from math import exp, log1p, sqrt
from typing import Dict, List, Optional, Set, Tuple

from peewee import fn

from server import config
from server.database import (
    Post,
    PostMetadata,
    Like,
    Repost,
    Follow,
    Block,
    UserAction,
    ServedPost,
)

uri = config.FEED_URI
CURSOR_EOF = 'eof'

_ACTION_WEIGHTS = {
    'like': 1.0,
    'repost': 2.0,
    'reply': 2.5,
    'post': 0.2,
}


@dataclass
class Candidate:
    uri: str
    cid: str
    author: str
    indexed_at: datetime
    created_at: datetime
    reply_parent: Optional[str]
    has_media: bool
    has_video: bool
    like_count: int
    repost_count: int
    reply_count: int
    in_network: bool
    social_proof: int
    author_affinity: float
    topics: List[str]
    language: Optional[str]
    safety: Optional[str]
    embedding: Optional[List[float]]
    topic_boost: float = 0.0
    embedding_boost: float = 0.0
    safety_penalty: float = 0.0
    score: float = 0.0


def handler(cursor: Optional[str], limit: int, viewer_did: str) -> dict:
    limit = max(1, min(limit, 100))
    now = datetime.utcnow()

    following = _load_following(viewer_did)
    blocked = _load_blocked(viewer_did)
    seen_posts = _load_recently_served(viewer_did, now)
    author_affinity = _load_author_affinity(viewer_did, now)
    social_proof = _load_social_proof(following, now)

    candidates = _load_candidates(following, now)
    candidates = _filter_candidates(
        candidates,
        viewer_did,
        blocked,
        seen_posts,
        cursor,
    )
    if not candidates:
        return {'cursor': CURSOR_EOF, 'feed': []}

    metadata_by_uri = _load_post_metadata(candidates) if config.LLM_ENABLED else {}
    topic_profile, user_embedding = (
        _load_user_llm_profile(viewer_did, now) if config.LLM_ENABLED else ({}, None)
    )

    scored = _score_candidates(
        candidates,
        following,
        author_affinity,
        social_proof,
        metadata_by_uri,
        topic_profile,
        user_embedding,
        now,
    )
    scored = _apply_author_diversity(scored)

    selected = scored[:limit]
    if not selected:
        return {'cursor': CURSOR_EOF, 'feed': []}

    _record_served(viewer_did, selected, now)

    cursor = _build_cursor(selected[-1])
    feed = [{'post': candidate.uri} for candidate in selected]
    return {
        'cursor': cursor,
        'feed': feed,
    }


def _load_following(viewer_did: str) -> Set[str]:
    return {
        row.subject
        for row in Follow.select(Follow.subject)
        .where(Follow.follower == viewer_did)
    }


def _load_blocked(viewer_did: str) -> Set[str]:
    return {
        row.subject
        for row in Block.select(Block.subject)
        .where(Block.blocker == viewer_did)
    }


def _load_recently_served(viewer_did: str, now: datetime) -> Set[str]:
    cutoff = now - timedelta(hours=config.SERVED_POST_TTL_HOURS)
    ServedPost.delete().where(ServedPost.served_at < cutoff).execute()
    return {
        row.post_uri
        for row in ServedPost.select(ServedPost.post_uri)
        .where(
            (ServedPost.viewer == viewer_did)
            & (ServedPost.served_at >= cutoff)
        )
    }


def _load_author_affinity(viewer_did: str, now: datetime) -> Dict[str, float]:
    cutoff = now - timedelta(hours=config.USER_ACTION_LOOKBACK_HOURS)
    affinity: Dict[str, float] = {}

    actions = (
        UserAction.select(UserAction.action, UserAction.subject_author)
        .where(
            (UserAction.user == viewer_did)
            & (UserAction.created_at >= cutoff)
            & UserAction.subject_author.is_null(False)
        )
        .order_by(UserAction.created_at.desc())
        .limit(config.USER_ACTION_HISTORY_LIMIT)
    )

    for action in actions:
        weight = _ACTION_WEIGHTS.get(action.action, 0.0)
        if not action.subject_author or weight <= 0:
            continue
        affinity[action.subject_author] = affinity.get(action.subject_author, 0.0) + weight

    return affinity


def _load_social_proof(following: Set[str], now: datetime) -> Dict[str, int]:
    if not following:
        return {}

    cutoff = now - timedelta(hours=config.SOCIAL_PROOF_MAX_AGE_HOURS)
    counts: Dict[str, int] = {}

    likes = (
        Like.select(Like.subject_uri, fn.COUNT(Like.uri).alias('count'))
        .where((Like.author.in_(following)) & (Like.created_at >= cutoff))
        .group_by(Like.subject_uri)
    )
    for row in likes:
        counts[row.subject_uri] = counts.get(row.subject_uri, 0) + int(row.count)

    reposts = (
        Repost.select(Repost.subject_uri, fn.COUNT(Repost.uri).alias('count'))
        .where((Repost.author.in_(following)) & (Repost.created_at >= cutoff))
        .group_by(Repost.subject_uri)
    )
    for row in reposts:
        counts[row.subject_uri] = counts.get(row.subject_uri, 0) + int(row.count) * 2

    return counts


def _load_post_metadata(candidates: List[Post]) -> Dict[str, PostMetadata]:
    if not candidates:
        return {}
    uris = [post.uri for post in candidates]
    rows = PostMetadata.select().where(PostMetadata.uri.in_(uris))
    return {row.uri: row for row in rows}


def _load_user_llm_profile(
    viewer_did: str,
    now: datetime,
) -> Tuple[Dict[str, float], Optional[List[float]]]:
    cutoff = now - timedelta(hours=config.USER_ACTION_LOOKBACK_HOURS)
    actions = (
        UserAction.select(UserAction.action, UserAction.subject_uri)
        .where(
            (UserAction.user == viewer_did)
            & (UserAction.created_at >= cutoff)
            & UserAction.subject_uri.is_null(False)
        )
        .order_by(UserAction.created_at.desc())
        .limit(config.USER_ACTION_HISTORY_LIMIT)
    )

    weighted_uris: List[Tuple[str, float]] = []
    for action in actions:
        weight = _ACTION_WEIGHTS.get(action.action, 0.0)
        if weight <= 0:
            continue
        if action.subject_uri:
            weighted_uris.append((action.subject_uri, weight))

    if not weighted_uris:
        return {}, None

    unique_uris = {uri for uri, _ in weighted_uris}
    metadata_rows = (
        PostMetadata.select()
        .where(PostMetadata.uri.in_(unique_uris))
    )
    metadata_map = {row.uri: row for row in metadata_rows}

    topic_profile: Dict[str, float] = {}
    embedding_sum: Optional[List[float]] = None
    total_weight = 0.0

    for uri, weight in weighted_uris:
        metadata = metadata_map.get(uri)
        if not metadata:
            continue

        topics = _parse_topics(metadata.topics)
        for topic in topics:
            topic_profile[topic] = topic_profile.get(topic, 0.0) + weight

        embedding = _parse_embedding(metadata.embedding)
        if embedding:
            if embedding_sum is None:
                embedding_sum = [0.0 for _ in embedding]
            if len(embedding_sum) != len(embedding):
                continue
            for idx, value in enumerate(embedding):
                embedding_sum[idx] += value * weight
            total_weight += weight

    user_embedding = None
    if embedding_sum and total_weight > 0:
        user_embedding = [value / total_weight for value in embedding_sum]

    return topic_profile, user_embedding


def _load_candidates(following: Set[str], now: datetime) -> List[Post]:
    candidates: List[Post] = []

    in_network_cutoff = now - timedelta(hours=config.IN_NETWORK_MAX_AGE_HOURS)
    if following:
        in_network = (
            Post.select()
            .where(
                (Post.author.in_(following))
                & (Post.indexed_at >= in_network_cutoff)
            )
            .order_by(Post.indexed_at.desc())
            .limit(config.IN_NETWORK_CANDIDATE_LIMIT)
        )
        candidates.extend(list(in_network))

    oon_cutoff = now - timedelta(hours=config.OON_MAX_AGE_HOURS)
    oon_query = Post.select().where(Post.indexed_at >= oon_cutoff)
    if following:
        oon_query = oon_query.where(Post.author.not_in(following))
    oon_posts = (
        oon_query
        .order_by(
            Post.like_count.desc(),
            Post.repost_count.desc(),
            Post.reply_count.desc(),
            Post.indexed_at.desc(),
        )
        .limit(config.OON_CANDIDATE_LIMIT)
    )
    candidates.extend(list(oon_posts))

    if following:
        proof_cutoff = now - timedelta(hours=config.SOCIAL_PROOF_MAX_AGE_HOURS)
        like_uris = (
            Like.select(Like.subject_uri)
            .where((Like.author.in_(following)) & (Like.created_at >= proof_cutoff))
            .distinct()
            .limit(config.SOCIAL_PROOF_CANDIDATE_LIMIT)
        )
        repost_uris = (
            Repost.select(Repost.subject_uri)
            .where((Repost.author.in_(following)) & (Repost.created_at >= proof_cutoff))
            .distinct()
            .limit(config.SOCIAL_PROOF_CANDIDATE_LIMIT)
        )
        proof_uris = {row.subject_uri for row in like_uris}
        proof_uris.update({row.subject_uri for row in repost_uris})
        if proof_uris:
            proof_posts = (
                Post.select()
                .where(Post.uri.in_(proof_uris))
                .order_by(Post.indexed_at.desc())
                .limit(config.SOCIAL_PROOF_CANDIDATE_LIMIT)
            )
            candidates.extend(list(proof_posts))

    if len(candidates) > config.MAX_CANDIDATES:
        candidates.sort(key=lambda post: post.indexed_at, reverse=True)
        candidates = candidates[: config.MAX_CANDIDATES]

    return candidates


def _filter_candidates(
    candidates: List[Post],
    viewer_did: str,
    blocked: Set[str],
    seen_posts: Set[str],
    cursor: Optional[str],
) -> List[Post]:
    cutoff = datetime.utcnow() - timedelta(hours=config.MAX_POST_AGE_HOURS)
    deduped: Dict[str, Post] = {}

    for post in candidates:
        if post.uri in deduped:
            continue
        if post.author == viewer_did:
            continue
        if post.author in blocked:
            continue
        if post.uri in seen_posts:
            continue
        if post.indexed_at < cutoff:
            continue
        if config.IGNORE_REPLY_POSTS and post.reply_parent:
            continue
        deduped[post.uri] = post

    filtered = list(deduped.values())
    if not cursor or cursor == CURSOR_EOF:
        return filtered

    parts = cursor.split('::')
    if len(parts) != 2:
        raise ValueError('Malformed cursor')

    cursor_time = datetime.utcfromtimestamp(int(parts[0]) / 1000)
    cursor_cid = parts[1]

    return [
        post
        for post in filtered
        if (post.indexed_at < cursor_time)
        or (post.indexed_at == cursor_time and post.cid < cursor_cid)
    ]


def _parse_topics(value: Optional[str]) -> List[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        return []
    return []


def _parse_embedding(value: Optional[str]) -> Optional[List[float]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return None


def _topic_boost(topics: List[str], profile: Dict[str, float]) -> float:
    if not topics or not profile:
        return 0.0
    score = sum(profile.get(topic, 0.0) for topic in topics)
    if score <= 0:
        return 0.0
    return log1p(score) * config.LLM_TOPIC_BOOST


def _embedding_boost(embedding: Optional[List[float]], user_embedding: Optional[List[float]]) -> float:
    if not embedding or not user_embedding:
        return 0.0
    if len(embedding) != len(user_embedding):
        return 0.0
    similarity = _cosine_similarity(embedding, user_embedding)
    if similarity <= 0:
        return 0.0
    return similarity * config.LLM_EMBEDDING_BOOST


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _safety_penalty(safety: Optional[str]) -> float:
    if not safety:
        return 0.0
    label = safety.strip().lower()
    if label in config.LLM_SAFETY_PENALTY_LABELS:
        return config.LLM_SAFETY_PENALTY
    return 0.0


def _score_candidates(
    candidates: List[Post],
    following: Set[str],
    author_affinity: Dict[str, float],
    social_proof: Dict[str, int],
    metadata_by_uri: Dict[str, PostMetadata],
    topic_profile: Dict[str, float],
    user_embedding: Optional[List[float]],
    now: datetime,
) -> List[Candidate]:
    scored: List[Candidate] = []

    for post in candidates:
        metadata = metadata_by_uri.get(post.uri)
        topics = _parse_topics(metadata.topics) if metadata else []
        language = metadata.language if metadata else None
        safety = metadata.safety if metadata else None
        embedding = _parse_embedding(metadata.embedding) if metadata else None

        if safety and safety.lower() in config.LLM_SAFETY_BLOCKLIST:
            continue

        topic_boost = _topic_boost(topics, topic_profile)
        embedding_boost = _embedding_boost(embedding, user_embedding)
        safety_penalty = _safety_penalty(safety)
        candidate = Candidate(
            uri=post.uri,
            cid=post.cid,
            author=post.author,
            indexed_at=post.indexed_at,
            created_at=post.created_at,
            reply_parent=post.reply_parent,
            has_media=post.has_media,
            has_video=post.has_video,
            like_count=post.like_count,
            repost_count=post.repost_count,
            reply_count=post.reply_count,
            in_network=post.author in following,
            social_proof=social_proof.get(post.uri, 0),
            author_affinity=author_affinity.get(post.author, 0.0),
            topics=topics,
            language=language,
            safety=safety,
            embedding=embedding,
            topic_boost=topic_boost,
            embedding_boost=embedding_boost,
            safety_penalty=safety_penalty,
        )
        candidate.score = _score_candidate(candidate, now)
        scored.append(candidate)

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def _score_candidate(candidate: Candidate, now: datetime) -> float:
    age_hours = max((now - candidate.indexed_at).total_seconds() / 3600, 0.0)
    recency = exp(-age_hours / config.RECENCY_HALF_LIFE_HOURS)

    engagement = (
        log1p(candidate.like_count) * config.WEIGHT_LIKE
        + log1p(candidate.repost_count) * config.WEIGHT_REPOST
        + log1p(candidate.reply_count) * config.WEIGHT_REPLY
    )
    affinity = log1p(candidate.author_affinity) * config.WEIGHT_AUTHOR_AFFINITY
    social = log1p(candidate.social_proof) * config.WEIGHT_SOCIAL_PROOF

    base = config.WEIGHT_IN_NETWORK if candidate.in_network else config.WEIGHT_OON
    media = config.WEIGHT_MEDIA if candidate.has_media else 0.0
    reply_penalty = config.WEIGHT_REPLY_PENALTY if candidate.reply_parent else 0.0

    return (
        engagement
        + affinity
        + social
        + base
        + media
        + (recency * config.WEIGHT_RECENCY)
        + reply_penalty
        + candidate.topic_boost
        + candidate.embedding_boost
        + candidate.safety_penalty
    )


def _apply_author_diversity(candidates: List[Candidate]) -> List[Candidate]:
    if not candidates:
        return candidates

    author_counts: Dict[str, int] = {}
    for candidate in candidates:
        count = author_counts.get(candidate.author, 0)
        if count > 0:
            candidate.score *= config.AUTHOR_DIVERSITY_DECAY ** count
        author_counts[candidate.author] = count + 1

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _record_served(viewer_did: str, selected: List[Candidate], now: datetime) -> None:
    if not selected:
        return

    rows = [
        {
            'viewer': viewer_did,
            'post_uri': candidate.uri,
            'served_at': now,
        }
        for candidate in selected
    ]

    ServedPost.insert_many(rows).on_conflict_replace().execute()


def _build_cursor(candidate: Candidate) -> str:
    timestamp_ms = int(candidate.indexed_at.replace(tzinfo=timezone.utc).timestamp() * 1000)
    return f'{timestamp_ms}::{candidate.cid}'
