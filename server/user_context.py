from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from typing import Dict, Iterable, List, Optional, Set, Tuple

import httpx
from atproto import DidInMemoryCache, IdResolver

from server import config
from server.database import (
    db,
    Block,
    Follow,
    Like,
    PostMetadata,
    UserAction,
    UserGraphState,
    UserTopicPreference,
)
from server.logger import logger


_CACHE = DidInMemoryCache()
_ID_RESOLVER = IdResolver(cache=_CACHE)


@dataclass
class ViewerContext:
    following: Set[str]
    blocked: Set[str]
    topic_preferences: Dict[str, float]


def get_viewer_context(viewer_did: str, auth_token: Optional[str]) -> ViewerContext:
    pds_url = _resolve_pds_endpoint(viewer_did)
    if pds_url and auth_token:
        _refresh_follows(viewer_did, auth_token, pds_url)
        _refresh_blocks(viewer_did, auth_token, pds_url)
        _refresh_likes(viewer_did, auth_token, pds_url)

    following = {
        row.subject
        for row in Follow.select(Follow.subject).where(Follow.follower == viewer_did)
    }
    blocked = {
        row.subject
        for row in Block.select(Block.subject).where(Block.blocker == viewer_did)
    }
    topic_preferences = {
        row.topic: row.weight
        for row in UserTopicPreference.select().where(UserTopicPreference.viewer == viewer_did)
    }

    return ViewerContext(
        following=following,
        blocked=blocked,
        topic_preferences=topic_preferences,
    )


def apply_interactions(viewer_did: str, interactions: Iterable[dict]) -> None:
    for interaction in interactions:
        event = interaction.get('event')
        item = interaction.get('item')
        feed_context = interaction.get('feedContext')
        if not event or not item:
            continue

        if event.endswith('#requestMore'):
            _apply_topic_preference(viewer_did, item, feed_context, config.TOPIC_PREF_MORE_DELTA)
        elif event.endswith('#requestLess'):
            _apply_topic_preference(viewer_did, item, feed_context, config.TOPIC_PREF_LESS_DELTA)


def _apply_topic_preference(
    viewer_did: str,
    post_uri: str,
    feed_context: Optional[str],
    delta: float,
) -> None:
    topics = _topics_for_interaction(post_uri, feed_context)
    if not topics:
        return

    for topic in topics:
        row = UserTopicPreference.get_or_none(
            (UserTopicPreference.viewer == viewer_did) & (UserTopicPreference.topic == topic)
        )
        weight = delta
        if row:
            weight = row.weight + delta
        weight = max(config.TOPIC_PREF_MIN, min(config.TOPIC_PREF_MAX, weight))

        if row:
            row.weight = weight
            row.updated_at = datetime.utcnow()
            row.save()
        else:
            UserTopicPreference.create(
                viewer=viewer_did,
                topic=topic,
                weight=weight,
                updated_at=datetime.utcnow(),
            )


def _topics_for_interaction(post_uri: str, feed_context: Optional[str]) -> List[str]:
    context_topics = _topics_from_feed_context(feed_context)
    if context_topics:
        return context_topics
    return _topics_for_uri(post_uri)


def _topics_for_uri(post_uri: str) -> List[str]:
    metadata = PostMetadata.get_or_none(PostMetadata.uri == post_uri)
    if not metadata or not metadata.topics:
        return []
    try:
        parsed = json.loads(metadata.topics)
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        return []
    return []


def _topics_from_feed_context(feed_context: Optional[str]) -> List[str]:
    if not feed_context:
        return []
    try:
        parsed = json.loads(feed_context)
        topics = parsed.get('topics') if isinstance(parsed, dict) else None
        if isinstance(topics, list):
            return [str(item).strip().lower() for item in topics if str(item).strip()]
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []
    return []


def _refresh_follows(viewer_did: str, auth_token: str, pds_url: str) -> None:
    state = _get_state(viewer_did)
    if not _needs_refresh(state.follows_refreshed_at, config.GRAPH_FOLLOWS_REFRESH_HOURS):
        return

    follows = _fetch_paged(
        pds_url,
        'app.bsky.graph.getFollows',
        auth_token,
        'follows',
        viewer_did,
        config.GRAPH_MAX_FOLLOWS,
    )
    if follows is None:
        return

    follow_rows = []
    for follow in follows:
        subject = follow.get('did')
        if not subject:
            continue
        follow_rows.append({
            'uri': f'{viewer_did}:{subject}',
            'follower': viewer_did,
            'subject': subject,
            'created_at': datetime.utcnow(),
        })

    with db.atomic():
        Follow.delete().where(Follow.follower == viewer_did).execute()
        if follow_rows:
            Follow.insert_many(follow_rows).on_conflict_ignore().execute()

    state.follows_refreshed_at = datetime.utcnow()
    state.save()


def _refresh_blocks(viewer_did: str, auth_token: str, pds_url: str) -> None:
    state = _get_state(viewer_did)
    if not _needs_refresh(state.blocks_refreshed_at, config.GRAPH_BLOCKS_REFRESH_HOURS):
        return

    blocks = _fetch_paged(
        pds_url,
        'app.bsky.graph.getBlocks',
        auth_token,
        'blocks',
        viewer_did,
        config.GRAPH_MAX_BLOCKS,
    )
    if blocks is None:
        return

    block_rows = []
    for block in blocks:
        subject = block.get('did')
        if not subject:
            continue
        block_rows.append({
            'uri': f'{viewer_did}:{subject}',
            'blocker': viewer_did,
            'subject': subject,
            'created_at': datetime.utcnow(),
        })

    with db.atomic():
        Block.delete().where(Block.blocker == viewer_did).execute()
        if block_rows:
            Block.insert_many(block_rows).on_conflict_ignore().execute()

    state.blocks_refreshed_at = datetime.utcnow()
    state.save()


def _refresh_likes(viewer_did: str, auth_token: str, pds_url: str) -> None:
    state = _get_state(viewer_did)
    if not _needs_refresh(state.likes_refreshed_at, config.GRAPH_LIKES_REFRESH_HOURS):
        return

    likes = _fetch_paged(
        pds_url,
        'app.bsky.feed.getActorLikes',
        auth_token,
        'feed',
        viewer_did,
        config.GRAPH_MAX_LIKES,
    )
    if likes is None:
        return

    like_rows = []
    action_rows = []
    now = datetime.utcnow()
    for item in likes:
        post = item.get('post') or {}
        viewer_state = post.get('viewer') or {}
        like_uri = viewer_state.get('like')
        subject_uri = post.get('uri')
        author = post.get('author') or {}
        subject_author = author.get('did')
        if not subject_uri or not subject_author or not like_uri:
            continue

        like_rows.append({
            'uri': like_uri,
            'subject_uri': subject_uri,
            'subject_author': subject_author,
            'author': viewer_did,
            'created_at': now,
        })
        action_rows.append({
            'record_uri': like_uri,
            'user': viewer_did,
            'action': 'like',
            'subject_uri': subject_uri,
            'subject_author': subject_author,
            'created_at': now,
        })

    with db.atomic():
        Like.delete().where(Like.author == viewer_did).execute()
        UserAction.delete().where(
            (UserAction.user == viewer_did) & (UserAction.action == 'like')
        ).execute()
        if like_rows:
            Like.insert_many(like_rows).on_conflict_ignore().execute()
        if action_rows:
            UserAction.insert_many(action_rows).on_conflict_ignore().execute()

    state.likes_refreshed_at = now
    state.save()


def _fetch_paged(
    pds_url: str,
    nsid: str,
    auth_token: str,
    list_key: str,
    actor: str,
    max_items: int,
) -> Optional[List[dict]]:
    headers = {'Authorization': f'Bearer {auth_token}'}
    cursor = None
    items: List[dict] = []

    with httpx.Client(timeout=20.0) as client:
        while True:
            params = {'actor': actor, 'limit': 100}
            if cursor:
                params['cursor'] = cursor
            url = f'{pds_url}/xrpc/{nsid}'
            try:
                response = client.get(url, params=params, headers=headers)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(f'PDS fetch failed for {nsid}: {exc}')
                return None

            data = response.json()
            batch = data.get(list_key) or []
            if not isinstance(batch, list):
                break
            items.extend(batch)
            if len(items) >= max_items:
                items = items[:max_items]
                break
            cursor = data.get('cursor')
            if not cursor:
                break

    return items


def _resolve_pds_endpoint(viewer_did: str) -> Optional[str]:
    state = _get_state(viewer_did)
    if state.pds_url and not _needs_refresh(state.pds_refreshed_at, config.PDS_CACHE_REFRESH_HOURS):
        return state.pds_url

    doc = _ID_RESOLVER.did.resolve_without_validation(viewer_did)
    if not doc:
        return None

    service_entries = doc.get('service') or []
    pds_url = None
    for entry in service_entries:
        if entry.get('type') in {'AtprotoPersonalDataServer', 'atproto_pds'}:
            pds_url = entry.get('serviceEndpoint')
            break

    if not pds_url:
        return None

    pds_url = pds_url.rstrip('/')
    state.pds_url = pds_url
    state.pds_refreshed_at = datetime.utcnow()
    state.save()
    return pds_url


def _get_state(viewer_did: str) -> UserGraphState:
    state = UserGraphState.get_or_none(UserGraphState.viewer == viewer_did)
    if state:
        return state
    return UserGraphState.create(viewer=viewer_did)


def _needs_refresh(timestamp: Optional[datetime], hours: int) -> bool:
    if not timestamp:
        return True
    return datetime.utcnow() - timestamp > timedelta(hours=hours)
