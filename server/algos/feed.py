from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from math import exp, log1p, sqrt
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from peewee import JOIN, fn

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
from server.user_context import ViewerContext

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
    follow_like_count: int
    follow_repost_count: int
    author_affinity: float
    topics: List[str]
    language: Optional[str]
    safety: Optional[str]
    embedding: Optional[List[float]]
    embedding_similarity: float
    topic_boost: float = 0.0
    embedding_boost: float = 0.0
    safety_penalty: float = 0.0
    score: float = 0.0


def handler(
    cursor: Optional[str],
    limit: int,
    viewer_did: str,
    viewer_context: Optional[ViewerContext] = None,
) -> dict:
    limit = max(1, min(limit, 100))
    now = datetime.utcnow()

    following = viewer_context.following if viewer_context else _load_following(viewer_did)
    blocked = viewer_context.blocked if viewer_context else _load_blocked(viewer_did)
    topic_preferences = viewer_context.topic_preferences if viewer_context else {}
    seen_posts = _load_recently_served(viewer_did, now)
    author_affinity = _load_author_affinity(viewer_did, now)
    follow_like_counts, follow_repost_counts = _load_follow_social_counts(following, now)

    candidates = _load_candidates(following, now, follow_like_counts, follow_repost_counts)
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
    topic_profile, user_embeddings = (
        _load_user_llm_profile(viewer_did, now, topic_preferences)
        if config.LLM_ENABLED
        else ({}, [])
    )

    scored = _score_candidates(
        candidates,
        following,
        author_affinity,
        follow_like_counts,
        follow_repost_counts,
        metadata_by_uri,
        topic_profile,
        user_embeddings,
        now,
    )
    scored = _apply_author_diversity(scored)

    selected = scored[:limit]
    if not selected:
        return {'cursor': CURSOR_EOF, 'feed': []}

    _record_served(viewer_did, selected, now)

    cursor = _build_cursor(selected[-1])
    req_id = uuid4().hex
    feed = []
    for candidate in selected:
        item = {'post': candidate.uri}
        if candidate.topics:
            item['feedContext'] = json.dumps({'topics': candidate.topics})
        feed.append(item)
    return {
        'cursor': cursor,
        'feed': feed,
        'reqId': req_id,
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


def _load_follow_social_counts(
    following: Set[str],
    now: datetime,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    if not following:
        return {}, {}

    cutoff = now - timedelta(hours=config.FOLLOW_NETWORK_WINDOW_HOURS)
    like_counts: Dict[str, int] = {}
    repost_counts: Dict[str, int] = {}

    likes = (
        Like.select(Like.subject_uri, fn.COUNT(fn.DISTINCT(Like.author)).alias('count'))
        .where((Like.author.in_(following)) & (Like.created_at >= cutoff))
        .group_by(Like.subject_uri)
    )
    for row in likes:
        like_counts[row.subject_uri] = int(row.count)

    reposts = (
        Repost.select(Repost.subject_uri, fn.COUNT(fn.DISTINCT(Repost.author)).alias('count'))
        .where((Repost.author.in_(following)) & (Repost.created_at >= cutoff))
        .group_by(Repost.subject_uri)
    )
    for row in reposts:
        repost_counts[row.subject_uri] = int(row.count)

    return like_counts, repost_counts


def _load_post_metadata(candidates: List[Post]) -> Dict[str, PostMetadata]:
    if not candidates:
        return {}
    uris = [post.uri for post in candidates]
    rows = PostMetadata.select().where(PostMetadata.uri.in_(uris))
    return {row.uri: row for row in rows}


def _load_user_llm_profile(
    viewer_did: str,
    now: datetime,
    topic_preferences: Dict[str, float],
) -> Tuple[Dict[str, float], List[List[float]]]:
    cutoff = now - timedelta(hours=config.USER_ACTION_LOOKBACK_HOURS)
    actions = (
        UserAction.select(UserAction.action, UserAction.subject_uri)
        .where(
            (UserAction.user == viewer_did)
            & (UserAction.created_at >= cutoff)
            & UserAction.subject_uri.is_null(False)
            & UserAction.action.in_(['like', 'repost'])
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
        return {}, []

    unique_uris = {uri for uri, _ in weighted_uris}
    metadata_rows = (
        PostMetadata.select()
        .where(PostMetadata.uri.in_(unique_uris))
    )
    metadata_map = {row.uri: row for row in metadata_rows}

    topic_profile: Dict[str, float] = {}
    embeddings: List[List[float]] = []

    for uri, weight in weighted_uris:
        metadata = metadata_map.get(uri)
        if not metadata:
            continue

        topics = _parse_topics(metadata.topics)
        for topic in topics:
            topic_profile[topic] = topic_profile.get(topic, 0.0) + weight

        embedding = _parse_embedding(metadata.embedding)
        if embedding:
            embeddings.append(embedding)
        if len(embeddings) >= config.EMBEDDING_PROFILE_SIZE:
            break

    if topic_preferences:
        for topic, weight in topic_preferences.items():
            topic_profile[topic] = topic_profile.get(topic, 0.0) + (weight * config.TOPIC_PREF_BOOST)

    return topic_profile, embeddings


def _load_candidates(
    following: Set[str],
    now: datetime,
    follow_like_counts: Dict[str, int],
    follow_repost_counts: Dict[str, int],
) -> List[Post]:
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

    network_uris = {
        uri
        for uri, count in follow_like_counts.items()
        if count >= config.FOLLOW_NETWORK_MIN_LIKES
    }
    network_uris.update({
        uri
        for uri, count in follow_repost_counts.items()
        if count >= config.FOLLOW_NETWORK_MIN_REPOSTS
    })

    if network_uris:
        network_posts = (
            Post.select()
            .where(Post.uri.in_(network_uris))
            .order_by(Post.indexed_at.desc())
            .limit(config.FOLLOW_NETWORK_CANDIDATE_LIMIT)
        )
        candidates.extend(list(network_posts))

    similarity_posts = _load_similarity_candidates(now)
    candidates.extend(similarity_posts)

    if len(candidates) > config.MAX_CANDIDATES:
        candidates.sort(key=lambda post: post.indexed_at, reverse=True)
        candidates = candidates[: config.MAX_CANDIDATES]

    return candidates


def _load_similarity_candidates(now: datetime) -> List[Post]:
    cutoff = now - timedelta(hours=config.SIMILARITY_MAX_AGE_HOURS)
    query = (
        Post.select()
        .join(PostMetadata, JOIN.LEFT_OUTER, on=(PostMetadata.uri == Post.uri))
        .where(
            (Post.indexed_at >= cutoff)
            & PostMetadata.embedding.is_null(False)
        )
        .order_by(
            Post.like_count.desc(),
            Post.repost_count.desc(),
            Post.reply_count.desc(),
            Post.indexed_at.desc(),
        )
        .limit(config.SIMILARITY_CANDIDATE_LIMIT)
    )
    return list(query)


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
    if score == 0:
        return 0.0
    if score > 0:
        return log1p(score) * config.LLM_TOPIC_BOOST
    return -log1p(abs(score)) * config.LLM_TOPIC_BOOST


def _embedding_similarity(embedding: Optional[List[float]], user_embeddings: List[List[float]]) -> float:
    if not embedding or not user_embeddings:
        return 0.0
    best = 0.0
    for profile_embedding in user_embeddings:
        if len(profile_embedding) != len(embedding):
            continue
        similarity = _cosine_similarity(embedding, profile_embedding)
        if similarity > best:
            best = similarity
    return best


def _embedding_boost(similarity: float) -> float:
    if similarity <= 0:
        return 0.0
    return similarity * config.LLM_EMBEDDING_BOOST


def _passes_network_or_similarity(
    in_network: bool,
    follow_like_count: int,
    follow_repost_count: int,
    similarity: float,
) -> bool:
    if in_network:
        return True
    if follow_like_count >= config.FOLLOW_NETWORK_MIN_LIKES:
        return True
    if follow_repost_count >= config.FOLLOW_NETWORK_MIN_REPOSTS:
        return True
    return similarity >= config.EMBEDDING_SIMILARITY_MIN


def _contains_blocked_topic(topics: List[str]) -> bool:
    return any(topic in config.TOPIC_BLOCKLIST for topic in topics)


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
    follow_like_counts: Dict[str, int],
    follow_repost_counts: Dict[str, int],
    metadata_by_uri: Dict[str, PostMetadata],
    topic_profile: Dict[str, float],
    user_embeddings: List[List[float]],
    now: datetime,
) -> List[Candidate]:
    scored: List[Candidate] = []

    for post in candidates:
        metadata = metadata_by_uri.get(post.uri)
        topics = _parse_topics(metadata.topics) if metadata else []
        language = metadata.language if metadata else None
        safety = metadata.safety if metadata else None
        embedding = _parse_embedding(metadata.embedding) if metadata else None

        if topics and _contains_blocked_topic(topics):
            continue

        if safety and safety.lower() in config.LLM_SAFETY_BLOCKLIST:
            continue

        follow_like_count = follow_like_counts.get(post.uri, 0)
        follow_repost_count = follow_repost_counts.get(post.uri, 0)
        embedding_similarity = _embedding_similarity(embedding, user_embeddings)

        if not _passes_network_or_similarity(
            post.author in following,
            follow_like_count,
            follow_repost_count,
            embedding_similarity,
        ):
            continue

        topic_boost = _topic_boost(topics, topic_profile)
        embedding_boost = _embedding_boost(embedding_similarity)
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
            follow_like_count=follow_like_count,
            follow_repost_count=follow_repost_count,
            author_affinity=author_affinity.get(post.author, 0.0),
            topics=topics,
            language=language,
            safety=safety,
            embedding=embedding,
            embedding_similarity=embedding_similarity,
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
    follow_social = (
        log1p(candidate.follow_like_count) * config.FOLLOW_NETWORK_LIKE_WEIGHT
        + log1p(candidate.follow_repost_count) * config.FOLLOW_NETWORK_REPOST_WEIGHT
    )

    base = config.WEIGHT_IN_NETWORK if candidate.in_network else config.WEIGHT_OON
    media = config.WEIGHT_MEDIA if candidate.has_media else 0.0
    reply_penalty = config.WEIGHT_REPLY_PENALTY if candidate.reply_parent else 0.0

    return (
        engagement
        + affinity
        + follow_social
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
