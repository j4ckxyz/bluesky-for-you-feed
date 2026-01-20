from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import List, Optional, Tuple

from atproto import AtUri, models

from server import config
from server.logger import logger
from server.database import (
    db,
    Post,
    Like,
    Repost,
    Follow,
    Block,
    UserAction,
)


def _parse_timestamp(value: str) -> datetime:
    if not value:
        return datetime.utcnow()
    if value.endswith('Z'):
        value = value.replace('Z', '+00:00')
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _get_author_from_uri(uri: str) -> Optional[str]:
    try:
        return AtUri.from_str(uri).host
    except Exception:
        return None


def _get_media_flags(record: 'models.AppBskyFeedPost.Record') -> Tuple[bool, bool]:
    has_media = False
    has_video = False

    if record.embed is None:
        return has_media, has_video

    if isinstance(record.embed, models.AppBskyEmbedImages.Main):
        has_media = True
    elif isinstance(record.embed, models.AppBskyEmbedVideo.Main):
        has_media = True
        has_video = True
    elif isinstance(record.embed, models.AppBskyEmbedExternal.Main):
        has_media = True
    elif isinstance(record.embed, models.AppBskyEmbedRecordWithMedia.Main):
        has_media = True
        media = record.embed.media
        if isinstance(media, models.AppBskyEmbedVideo.Main):
            has_video = True

    return has_media, has_video


def _record_user_action(
    record_uri: str,
    user: str,
    action: str,
    subject_uri: Optional[str],
    subject_author: Optional[str],
    created_at: datetime,
) -> None:
    UserAction.insert(
        record_uri=record_uri,
        user=user,
        action=action,
        subject_uri=subject_uri,
        subject_author=subject_author,
        created_at=created_at,
    ).on_conflict_ignore().execute()


def _update_post_counter(uri: str, field, delta: int) -> None:
    if not uri or delta == 0:
        return
    Post.update({field: field + delta}).where(Post.uri == uri).execute()


def is_archive_post(record: 'models.AppBskyFeedPost.Record') -> bool:
    # Sometimes users will import old posts from Twitter/X which con flood a feed with
    # old posts. Unfortunately, the only way to test for this is to look an old
    # created_at date. However, there are other reasons why a post might have an old
    # date, such as firehose or firehose consumer outages. It is up to you, the feed
    # creator to weigh the pros and cons, amd and optionally include this function in
    # your filter conditions, and adjust the threshold to your liking.
    #
    # See https://github.com/MarshalX/bluesky-feed-generator/pull/21

    archived_threshold = timedelta(days=1)
    created_at = _parse_timestamp(record.created_at)
    now = datetime.utcnow()

    return now - created_at > archived_threshold


def should_ignore_post(created_post: dict) -> bool:
    record = created_post['record']
    uri = created_post['uri']

    if config.IGNORE_ARCHIVED_POSTS and is_archive_post(record):
        logger.debug(f'Ignoring archived post: {uri}')
        return True

    if config.IGNORE_REPLY_POSTS and record.reply:
        logger.debug(f'Ignoring reply post: {uri}')
        return True

    return False


def _handle_post_creates(created_posts: List[dict]) -> None:
    posts_to_create = []

    for created_post in created_posts:
        author = created_post['author']
        record = created_post['record']

        if should_ignore_post(created_post):
            continue

        has_media, has_video = _get_media_flags(record)
        created_at = _parse_timestamp(record.created_at)

        reply_parent = None
        reply_root = None
        reply_parent_author = None
        if record.reply:
            reply_parent = record.reply.parent.uri
            reply_root = record.reply.root.uri
            reply_parent_author = _get_author_from_uri(reply_parent)

        posts_to_create.append({
            'uri': created_post['uri'],
            'cid': created_post['cid'],
            'author': author,
            'text': record.text or '',
            'reply_parent': reply_parent,
            'reply_root': reply_root,
            'reply_parent_author': reply_parent_author,
            'created_at': created_at,
            'indexed_at': datetime.utcnow(),
            'has_media': has_media,
            'has_video': has_video,
            'like_count': 0,
            'repost_count': 0,
            'reply_count': 0,
        })

        if reply_parent:
            _update_post_counter(reply_parent, Post.reply_count, 1)
            _record_user_action(
                created_post['uri'],
                author,
                'reply',
                reply_parent,
                reply_parent_author,
                created_at,
            )
        else:
            _record_user_action(
                created_post['uri'],
                author,
                'post',
                None,
                None,
                created_at,
            )

    if posts_to_create:
        with db.atomic():
            Post.insert_many(posts_to_create).on_conflict_ignore().execute()
        logger.debug(f'Added posts: {len(posts_to_create)}')


def _handle_post_deletes(deleted_posts: List[dict]) -> None:
    if not deleted_posts:
        return

    post_uris = [post['uri'] for post in deleted_posts]

    for post in Post.select().where(Post.uri.in_(post_uris)):
        if post.reply_parent:
            _update_post_counter(post.reply_parent, Post.reply_count, -1)

    Post.delete().where(Post.uri.in_(post_uris)).execute()
    UserAction.delete().where(UserAction.record_uri.in_(post_uris)).execute()
    logger.debug(f'Deleted posts: {len(post_uris)}')


def _handle_like_creates(created_likes: List[dict]) -> None:
    likes_to_create = []

    for created_like in created_likes:
        record = created_like['record']
        subject_uri = record.subject.uri
        subject_author = _get_author_from_uri(subject_uri)
        created_at = _parse_timestamp(record.created_at)

        likes_to_create.append({
            'uri': created_like['uri'],
            'subject_uri': subject_uri,
            'subject_author': subject_author,
            'author': created_like['author'],
            'created_at': created_at,
        })

        _update_post_counter(subject_uri, Post.like_count, 1)
        _record_user_action(
            created_like['uri'],
            created_like['author'],
            'like',
            subject_uri,
            subject_author,
            created_at,
        )

    if likes_to_create:
        with db.atomic():
            Like.insert_many(likes_to_create).on_conflict_ignore().execute()
        logger.debug(f'Added likes: {len(likes_to_create)}')


def _handle_like_deletes(deleted_likes: List[dict]) -> None:
    if not deleted_likes:
        return

    like_uris = [like['uri'] for like in deleted_likes]
    for like in Like.select().where(Like.uri.in_(like_uris)):
        _update_post_counter(like.subject_uri, Post.like_count, -1)

    Like.delete().where(Like.uri.in_(like_uris)).execute()
    UserAction.delete().where(UserAction.record_uri.in_(like_uris)).execute()


def _handle_repost_creates(created_reposts: List[dict]) -> None:
    reposts_to_create = []

    for created_repost in created_reposts:
        record = created_repost['record']
        subject_uri = record.subject.uri
        subject_author = _get_author_from_uri(subject_uri)
        created_at = _parse_timestamp(record.created_at)

        reposts_to_create.append({
            'uri': created_repost['uri'],
            'subject_uri': subject_uri,
            'subject_author': subject_author,
            'author': created_repost['author'],
            'created_at': created_at,
        })

        _update_post_counter(subject_uri, Post.repost_count, 1)
        _record_user_action(
            created_repost['uri'],
            created_repost['author'],
            'repost',
            subject_uri,
            subject_author,
            created_at,
        )

    if reposts_to_create:
        with db.atomic():
            Repost.insert_many(reposts_to_create).on_conflict_ignore().execute()
        logger.debug(f'Added reposts: {len(reposts_to_create)}')


def _handle_repost_deletes(deleted_reposts: List[dict]) -> None:
    if not deleted_reposts:
        return

    repost_uris = [repost['uri'] for repost in deleted_reposts]
    for repost in Repost.select().where(Repost.uri.in_(repost_uris)):
        _update_post_counter(repost.subject_uri, Post.repost_count, -1)

    Repost.delete().where(Repost.uri.in_(repost_uris)).execute()
    UserAction.delete().where(UserAction.record_uri.in_(repost_uris)).execute()


def _handle_follow_creates(created_follows: List[dict]) -> None:
    follows_to_create = []

    for created_follow in created_follows:
        record = created_follow['record']
        created_at = _parse_timestamp(record.created_at)
        follows_to_create.append({
            'uri': created_follow['uri'],
            'follower': created_follow['author'],
            'subject': record.subject,
            'created_at': created_at,
        })

    if follows_to_create:
        with db.atomic():
            Follow.insert_many(follows_to_create).on_conflict_ignore().execute()
        logger.debug(f'Added follows: {len(follows_to_create)}')


def _handle_follow_deletes(deleted_follows: List[dict]) -> None:
    if not deleted_follows:
        return

    follow_uris = [follow['uri'] for follow in deleted_follows]
    Follow.delete().where(Follow.uri.in_(follow_uris)).execute()


def _handle_block_creates(created_blocks: List[dict]) -> None:
    blocks_to_create = []

    for created_block in created_blocks:
        record = created_block['record']
        created_at = _parse_timestamp(record.created_at)
        blocks_to_create.append({
            'uri': created_block['uri'],
            'blocker': created_block['author'],
            'subject': record.subject,
            'created_at': created_at,
        })

    if blocks_to_create:
        with db.atomic():
            Block.insert_many(blocks_to_create).on_conflict_ignore().execute()


def _handle_block_deletes(deleted_blocks: List[dict]) -> None:
    if not deleted_blocks:
        return

    block_uris = [block['uri'] for block in deleted_blocks]
    Block.delete().where(Block.uri.in_(block_uris)).execute()


def operations_callback(ops: defaultdict) -> None:
    _handle_post_creates(ops[models.ids.AppBskyFeedPost]['created'])
    _handle_post_deletes(ops[models.ids.AppBskyFeedPost]['deleted'])

    _handle_like_creates(ops[models.ids.AppBskyFeedLike]['created'])
    _handle_like_deletes(ops[models.ids.AppBskyFeedLike]['deleted'])

    _handle_repost_creates(ops[models.ids.AppBskyFeedRepost]['created'])
    _handle_repost_deletes(ops[models.ids.AppBskyFeedRepost]['deleted'])

    _handle_follow_creates(ops[models.ids.AppBskyGraphFollow]['created'])
    _handle_follow_deletes(ops[models.ids.AppBskyGraphFollow]['deleted'])

    _handle_block_creates(ops[models.ids.AppBskyGraphBlock]['created'])
    _handle_block_deletes(ops[models.ids.AppBskyGraphBlock]['deleted'])
