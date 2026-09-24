"""Чанкинг статей: склейка абзацев, один чанк для заголовков, порядок индексов (без сети/БД)."""

from fplcopilot.rag.chunking import (
    MIN_TAIL_TOKENS,
    chunk_article,
    single_chunk_text,
    split_long_paragraph,
    split_paragraphs,
)


def words(text: str) -> int:
    """Дешёвый счётчик «токенов» для тестов: слова."""
    return len(text.split())


def para(n: int, tag: str) -> str:
    """Абзац из n «слов» (буквы, без цифр — иначе фильтр боилерплейта сочтёт его мусором)."""
    return " ".join(f"{tag}word" for _ in range(n)) + "."


def test_headline_only_article_is_single_chunk_without_duplication():
    chunks = chunk_article(
        title="Saka injury update",
        summary="Saka injury update Sky Sports",
        content="Saka injury update Sky Sports",
        fulltext=False,
        count=words,
    )
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == "Saka injury update Sky Sports"  # тело покрывает заголовок — берём его
    assert chunks[0].n_tokens == 5


def test_fpl_api_article_keeps_title_and_body():
    text = single_chunk_text(
        "Saka: Knee injury", "Bukayo Saka (Arsenal, MID) — FPL status: injured"
    )
    assert text == "Saka: Knee injury\nBukayo Saka (Arsenal, MID) — FPL status: injured"
    assert single_chunk_text("Title", None) == "Title"
    assert single_chunk_text("Title", "  title ") == "Title"


def test_fulltext_paragraphs_are_merged_up_to_budget_and_keep_order():
    title = "Arsenal team news"
    content = "\n".join(para(20, f"p{i}w") for i in range(12))  # 12 абзацев по 20 слов
    chunks = chunk_article(
        title=title, summary=None, content=content, fulltext=True, max_tokens=100, count=words
    )
    # бюджет тела = 100 - 3 (заголовок) - 1 = 96 слов -> по 4 абзаца (80) в чанк
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert len(chunks) == 3
    assert all(c.text.startswith(title + "\n") for c in chunks)
    assert all(c.n_tokens == 83 for c in chunks)
    # абзацы идут подряд и ничего не потеряно
    joined = "\n".join(c.text.removeprefix(title + "\n") for c in chunks)
    assert joined == content


def test_short_tail_is_glued_to_previous_chunk():
    title = "T"
    body = [para(50, "a"), para(50, "b"), para(5, "tail")]  # хвост 6 слов < MIN_TAIL_TOKENS
    chunks = chunk_article(
        title=title,
        summary=None,
        content="\n".join(body),
        fulltext=True,
        max_tokens=60,
        count=words,
    )
    assert len(chunks) == 2
    assert chunks[-1].text.endswith(para(5, "tail"))
    assert 6 < MIN_TAIL_TOKENS


def test_long_paragraph_is_split_by_sentences():
    sentences = [f"Sentence number {i} is here." for i in range(12)]  # по 5 слов
    pieces = split_long_paragraph(" ".join(sentences), max_tokens=12, count=words)
    # по 2 предложения (10 слов) в кусок = 6 кусков, но последний (10 слов < MIN_TAIL_TOKENS)
    # приклеивается к предыдущему -> 5
    assert len(pieces) == 5
    assert all(len(p.split("\n")) == 2 for p in pieces[:-1])
    assert len(pieces[-1].split("\n")) == 4
    assert " ".join(" ".join(pieces).split()) == " ".join(sentences)


def test_boilerplate_and_title_duplicates_are_dropped():
    paras = split_paragraphs(
        "- Published\n  - Published2 days ago\nImage source, Getty Images\n"
        "Real paragraph with enough words.\n\n\nok\n"
    )
    assert paras == ["Real paragraph with enough words."]
    chunks = chunk_article(
        title="Headline here",
        summary="s",
        content="Headline here\nBody paragraph with several words in it.",
        fulltext=True,
        count=words,
    )
    assert len(chunks) == 1
    assert chunks[0].text == "Headline here\nBody paragraph with several words in it."


def test_fulltext_flag_without_content_falls_back_to_summary():
    chunks = chunk_article(
        title="Only title", summary="A summary line", content=None, fulltext=True, count=words
    )
    assert len(chunks) == 1
    assert chunks[0].text == "Only title\nA summary line"
