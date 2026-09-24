"""Strategy knowledge base (RAG #2): вечнозелёные знания «как играть в FPL» с цитатами.

News RAG (`fplcopilot.rag`) отвечает на «что происходит с игроком X»; этот подпакет — на
«как играть правильно»: правила, тайминг чипов, математика хитов, накопление трансферов,
шаблон vs дифференциалы, защита ранга, цены, структура состава, капитанство. Знания здесь
СОВЕТЫ и ПРАВИЛА с источником, а не числа: все числа в рекомендациях по-прежнему считает
оптимизатор (`fplcopilot.core`).

Модули: registry (реестр источников sources_kb.yaml), fetch (html / reddit_json / internal,
дисковый кэш), chunking (чанки с заголовком раздела), ingest (kb_docs/kb_chunks + эмбеддинги),
retrieve (KBRetriever: dense + BM25 -> RRF -> reranker, без time-decay), answer
(цитируемый ответ gpt-4o-mini + детерминированная проверка цитат). CLI — `python -m fplcopilot.rag.kb`.
Схема — миграция 008_strategy_kb.sql; документация — docs/strategy_kb.md.
"""
