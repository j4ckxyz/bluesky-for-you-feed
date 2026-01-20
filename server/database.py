from datetime import datetime

import peewee

db = peewee.SqliteDatabase('feed_database.db')


class BaseModel(peewee.Model):
    class Meta:
        database = db


class Post(BaseModel):
    uri = peewee.CharField(primary_key=True)
    cid = peewee.CharField()
    author = peewee.CharField(index=True)
    text = peewee.TextField(default='')
    reply_parent = peewee.CharField(null=True, default=None)
    reply_root = peewee.CharField(null=True, default=None)
    reply_parent_author = peewee.CharField(null=True, default=None, index=True)
    created_at = peewee.DateTimeField(index=True)
    indexed_at = peewee.DateTimeField(default=datetime.utcnow, index=True)
    has_media = peewee.BooleanField(default=False)
    has_video = peewee.BooleanField(default=False)
    like_count = peewee.IntegerField(default=0)
    repost_count = peewee.IntegerField(default=0)
    reply_count = peewee.IntegerField(default=0)


class Like(BaseModel):
    uri = peewee.CharField(primary_key=True)
    subject_uri = peewee.CharField(index=True)
    subject_author = peewee.CharField(null=True, default=None, index=True)
    author = peewee.CharField(index=True)
    created_at = peewee.DateTimeField(index=True)


class Repost(BaseModel):
    uri = peewee.CharField(primary_key=True)
    subject_uri = peewee.CharField(index=True)
    subject_author = peewee.CharField(null=True, default=None, index=True)
    author = peewee.CharField(index=True)
    created_at = peewee.DateTimeField(index=True)


class Follow(BaseModel):
    uri = peewee.CharField(primary_key=True)
    follower = peewee.CharField(index=True)
    subject = peewee.CharField(index=True)
    created_at = peewee.DateTimeField(index=True)


class Block(BaseModel):
    uri = peewee.CharField(primary_key=True)
    blocker = peewee.CharField(index=True)
    subject = peewee.CharField(index=True)
    created_at = peewee.DateTimeField(index=True)


class UserAction(BaseModel):
    record_uri = peewee.CharField(primary_key=True)
    user = peewee.CharField(index=True)
    action = peewee.CharField(index=True)
    subject_uri = peewee.CharField(null=True, default=None)
    subject_author = peewee.CharField(null=True, default=None, index=True)
    created_at = peewee.DateTimeField(index=True)


class ServedPost(BaseModel):
    viewer = peewee.CharField(index=True)
    post_uri = peewee.CharField(index=True)
    served_at = peewee.DateTimeField(default=datetime.utcnow, index=True)

    class Meta:
        indexes = ((('viewer', 'post_uri'), True),)


class SubscriptionState(BaseModel):
    service = peewee.CharField(unique=True)
    cursor = peewee.BigIntegerField()


if db.is_closed():
    db.connect()
    db.create_tables([
        Post,
        Like,
        Repost,
        Follow,
        Block,
        UserAction,
        ServedPost,
        SubscriptionState,
    ], safe=True)
