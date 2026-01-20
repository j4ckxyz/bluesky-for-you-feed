from dataclasses import dataclass
import json
import queue
import random
import threading
import time
from typing import Dict, List, Optional

import httpx

from server import config
from server.logger import logger
from server.database import Post, PostMetadata
from datetime import datetime


_CLASSIFICATION_PROMPT = (
    'You are a classifier for short social posts. Return ONLY JSON with keys: '
    'language (BCP-47 like "en" or "es" or "und"), '
    'topics (array of 2-6 short lowercase tags, hyphens ok, no spaces), '
    'safety (one of "safe", "suggestive", "nsfw", "graphic").\n\n'
    'Post:\n'
)


@dataclass
class LLMJob:
    uri: str
    text: str


class BaseLLMClient:
    def classify(self, text: str) -> Optional[Dict[str, object]]:
        raise NotImplementedError

    def embed(self, text: str) -> Optional[List[float]]:
        raise NotImplementedError


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, embedding_model: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self._client = httpx.Client(timeout=30.0)

    def classify(self, text: str) -> Optional[Dict[str, object]]:
        if not self.model:
            return None
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': 'Return only JSON.'},
                {'role': 'user', 'content': f'{_CLASSIFICATION_PROMPT}{text}'},
            ],
            'temperature': 0.2,
        }
        headers = {'Authorization': f'Bearer {self.api_key}'}
        try:
            response = self._client.post(f'{self.base_url}/chat/completions', json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f'LLM classify failed: {exc}')
            return None

        data = response.json()
        content = _extract_message_content(data)
        return _parse_json_object(content)

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.embedding_model:
            return None
        payload = {
            'model': self.embedding_model,
            'input': text,
        }
        headers = {'Authorization': f'Bearer {self.api_key}'}
        try:
            response = self._client.post(f'{self.base_url}/embeddings', json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f'LLM embed failed: {exc}')
            return None

        data = response.json()
        embeddings = data.get('data') or []
        if embeddings:
            return embeddings[0].get('embedding')
        return None


class GeminiClient(BaseLLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, embedding_model: str) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self._client = httpx.Client(timeout=30.0)

    def classify(self, text: str) -> Optional[Dict[str, object]]:
        if not self.model:
            return None
        payload = {
            'contents': [
                {
                    'role': 'user',
                    'parts': [{'text': f'{_CLASSIFICATION_PROMPT}{text}'}],
                }
            ],
            'generationConfig': {
                'temperature': 0.2,
            },
        }
        url = f'{self.base_url}/models/{self.model}:generateContent?key={self.api_key}'
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f'Gemini classify failed: {exc}')
            return None

        data = response.json()
        candidates = data.get('candidates') or []
        if not candidates:
            return None
        parts = candidates[0].get('content', {}).get('parts') or []
        if not parts:
            return None
        content = parts[0].get('text', '')
        return _parse_json_object(content)

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.embedding_model:
            return None
        payload = {
            'content': {
                'parts': [{'text': text}],
            }
        }
        url = f'{self.base_url}/models/{self.embedding_model}:embedContent?key={self.api_key}'
        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f'Gemini embed failed: {exc}')
            return None

        data = response.json()
        embedding = data.get('embedding')
        if embedding and isinstance(embedding, dict):
            return embedding.get('values')
        return None


class MinimaxClient(BaseLLMClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        group_id: Optional[str],
        chat_path: str,
        embed_path: str,
        model: str,
        embedding_model: str,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.group_id = group_id
        self.chat_path = chat_path
        self.embed_path = embed_path
        self.model = model
        self.embedding_model = embedding_model
        self._client = httpx.Client(timeout=30.0)

    def _headers(self) -> Dict[str, str]:
        headers = {'Authorization': f'Bearer {self.api_key}'}
        if self.group_id:
            headers['X-Group-ID'] = self.group_id
        return headers

    def classify(self, text: str) -> Optional[Dict[str, object]]:
        if not self.model:
            return None
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': f'{_CLASSIFICATION_PROMPT}{text}'},
            ],
            'temperature': 0.2,
        }
        url = f'{self.base_url}{self.chat_path}'
        try:
            response = self._client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f'Minimax classify failed: {exc}')
            return None

        data = response.json()
        content = _extract_message_content(data)
        return _parse_json_object(content)

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.embedding_model:
            return None
        payload = {
            'model': self.embedding_model,
            'input': text,
        }
        url = f'{self.base_url}{self.embed_path}'
        try:
            response = self._client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(f'Minimax embed failed: {exc}')
            return None

        data = response.json()
        return _extract_embedding(data)


class LLMEnricher:
    def __init__(self) -> None:
        self._queue: queue.Queue[LLMJob] = queue.Queue(maxsize=config.LLM_MAX_QUEUE)
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client = self._build_client()
        self._embedding_model = ''
        if self._client:
            self._embedding_model = getattr(self._client, 'embedding_model', '')
        self._min_interval = 0.0
        if config.LLM_MAX_CALLS_PER_MINUTE > 0:
            self._min_interval = 60.0 / config.LLM_MAX_CALLS_PER_MINUTE
        self._last_call = 0.0

    def start(self) -> None:
        if not self._client or self._thread:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def maybe_enqueue_post(
        self,
        uri: str,
        text: str,
        like_count: int = 0,
        repost_count: int = 0,
        reply_count: int = 0,
        reason: str = 'sample',
    ) -> None:
        if not self._client or not config.LLM_ENABLED:
            return

        if not uri or not text:
            return

        if PostMetadata.get_or_none(PostMetadata.uri == uri):
            return

        if not self._should_enqueue(like_count, repost_count, reply_count, reason):
            return

        text = self._trim_text(text)
        if not text:
            return

        with self._lock:
            if uri in self._pending:
                return
            self._pending.add(uri)

        try:
            self._queue.put_nowait(LLMJob(uri=uri, text=text))
        except queue.Full:
            with self._lock:
                self._pending.discard(uri)
            logger.warning('LLM queue full; dropping job')

    def maybe_enqueue_by_uri(self, uri: str, reason: str = 'engagement') -> None:
        if not self._client or not config.LLM_ENABLED:
            return
        post = Post.get_or_none(Post.uri == uri)
        if not post:
            return
        self.maybe_enqueue_post(
            uri=post.uri,
            text=post.text,
            like_count=post.like_count,
            repost_count=post.repost_count,
            reply_count=post.reply_count,
            reason=reason,
        )

    def _should_enqueue(
        self,
        like_count: int,
        repost_count: int,
        reply_count: int,
        reason: str,
    ) -> bool:
        if reason != 'sample':
            return (
                like_count >= config.LLM_MIN_LIKES
                or repost_count >= config.LLM_MIN_REPOSTS
                or reply_count >= config.LLM_MIN_REPLIES
            )
        return random.random() < config.LLM_SAMPLE_RATE

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_job(job)
            finally:
                with self._lock:
                    self._pending.discard(job.uri)
                self._queue.task_done()

    def _process_job(self, job: LLMJob) -> None:
        if not self._client:
            return

        classification = self._classify(job.text)
        embedding = self._embed(job.text)

        if not classification and not embedding:
            return

        topics = None
        language = None
        safety = None
        if classification:
            language = _safe_str(classification.get('language'))
            topics = _safe_json_list(classification.get('topics'))
            safety = _safe_str(classification.get('safety'))

        embedding_json = None
        embedding_model = None
        if embedding:
            embedding_json = json.dumps(embedding)
            embedding_model = self._embedding_model

        PostMetadata.insert(
            uri=job.uri,
            language=language,
            topics=json.dumps(topics) if topics else None,
            safety=safety,
            embedding=embedding_json,
            embedding_model=embedding_model,
        ).on_conflict(
            conflict_target=[PostMetadata.uri],
            update={
                PostMetadata.language: language,
                PostMetadata.topics: json.dumps(topics) if topics else None,
                PostMetadata.safety: safety,
                PostMetadata.embedding: embedding_json,
                PostMetadata.embedding_model: embedding_model,
                PostMetadata.classified_at: datetime.utcnow(),
            },
        ).execute()

    def _classify(self, text: str) -> Optional[Dict[str, object]]:
        self._rate_limit()
        return self._client.classify(text)

    def _embed(self, text: str) -> Optional[List[float]]:
        if not self._embedding_model:
            return None
        self._rate_limit()
        return self._client.embed(text)

    def _rate_limit(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def _build_client(self) -> Optional[BaseLLMClient]:
        if not config.LLM_ENABLED:
            return None

        provider = config.LLM_PROVIDER.lower()
        if provider == 'openai':
            key = config.OPENAI_API_KEY or config.LLM_API_KEY
            if not key:
                logger.warning('OpenAI API key missing; LLM disabled')
                return None
            return OpenAICompatibleClient(
                config.LLM_BASE_URL,
                key,
                _normalize_model_name(config.OPENAI_MODEL or config.LLM_MODEL),
                _normalize_model_name(
                    config.OPENAI_EMBEDDING_MODEL or config.LLM_EMBEDDING_MODEL
                ),
            )

        if provider == 'openrouter':
            key = config.OPENROUTER_API_KEY or config.LLM_API_KEY
            if not key:
                logger.warning('OpenRouter API key missing; LLM disabled')
                return None
            return OpenAICompatibleClient(
                config.OPENROUTER_BASE_URL,
                key,
                _normalize_model_name(config.OPENROUTER_MODEL or config.LLM_MODEL),
                _normalize_model_name(
                    config.OPENROUTER_EMBEDDING_MODEL or config.LLM_EMBEDDING_MODEL
                ),
            )

        if provider == 'gemini':
            key = config.GEMINI_API_KEY or config.LLM_API_KEY
            if not key:
                logger.warning('Gemini API key missing; LLM disabled')
                return None
            return GeminiClient(
                config.GEMINI_BASE_URL,
                key,
                _normalize_model_name(config.GEMINI_MODEL or config.LLM_MODEL),
                _normalize_model_name(
                    config.GEMINI_EMBEDDING_MODEL or config.LLM_EMBEDDING_MODEL
                ),
            )

        if provider == 'minimax':
            key = config.MINIMAX_API_KEY or config.LLM_API_KEY
            if not key:
                logger.warning('Minimax API key missing; LLM disabled')
                return None
            return MinimaxClient(
                config.MINIMAX_BASE_URL,
                key,
                config.MINIMAX_GROUP_ID,
                config.MINIMAX_CHAT_PATH,
                config.MINIMAX_EMBED_PATH,
                _normalize_model_name(config.MINIMAX_MODEL or config.LLM_MODEL),
                _normalize_model_name(
                    config.MINIMAX_EMBEDDING_MODEL or config.LLM_EMBEDDING_MODEL
                ),
            )

        logger.warning(f'Unknown LLM provider: {provider}')
        return None

    def _trim_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ''
        limit = max(config.LLM_MAX_TEXT_CHARS, 1)
        if len(text) > limit:
            return text[:limit]
        return text


def _extract_message_content(data: Dict[str, object]) -> str:
    choices = data.get('choices') if isinstance(data, dict) else None
    if choices and isinstance(choices, list):
        choice = choices[0]
        message = choice.get('message') if isinstance(choice, dict) else None
        if message and isinstance(message, dict) and message.get('content'):
            return str(message.get('content'))
        if isinstance(choice, dict) and choice.get('text'):
            return str(choice.get('text'))
    if isinstance(data, dict) and data.get('result'):
        return str(data.get('result'))
    if isinstance(data, dict) and data.get('output'):
        output = data.get('output')
        if isinstance(output, dict) and output.get('text'):
            return str(output.get('text'))
    return ''


def _extract_embedding(data: Dict[str, object]) -> Optional[List[float]]:
    if not isinstance(data, dict):
        return None
    if data.get('embedding') and isinstance(data.get('embedding'), list):
        return data.get('embedding')
    if data.get('data') and isinstance(data.get('data'), list):
        item = data['data'][0]
        if isinstance(item, dict) and item.get('embedding'):
            return item.get('embedding')
    if data.get('vectors') and isinstance(data.get('vectors'), list):
        return data.get('vectors')[0]
    return None


def _parse_json_object(text: str) -> Optional[Dict[str, object]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _safe_json_list(value: object) -> Optional[List[str]]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return None


def _safe_str(value: object) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value if value else None


def _normalize_model_name(value: Optional[str]) -> str:
    if not value:
        return ''
    cleaned = str(value).strip()
    if cleaned.lower() in {'none', 'null', 'false', 'off', 'disabled'}:
        return ''
    return cleaned


llm_enricher = LLMEnricher()
