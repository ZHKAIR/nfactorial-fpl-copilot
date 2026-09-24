"""Чанкинг KB с заголовками разделов: путь заголовков в каждом чанке, списки/таблицы целиком,
бюджет токенов, порядок, склейка крошечных разделов (без сети/БД)."""

from fplcopilot.rag.chunking import MIN_BODY_BUDGET, MIN_TAIL_TOKENS
from fplcopilot.rag.kb.chunking import (
    chunk_document,
    clean_title,
    parse_heading,
    split_kb_paragraphs,
    split_sections,
)


def words(text: str) -> int:
    return len(text.split())


def para(n: int, tag: str) -> str:
    return " ".join(f"{tag}word" for _ in range(n)) + "."


DOC = "\n".join(
    [
        "Intro paragraph about the guide with enough words here.",
        "",
        "# FPL Chips Guide",
        "",
        "## Wildcard",
        "",
        para(30, "wc"),
        "",
        "### When to play",
        "",
        para(30, "when"),
        para(30, "when2"),
        "",
        "## Free Hit",
        "",
        "**Rules:**",
        "- 2 Goalkeepers",
        "- 5 Defenders",
        "- Cannot be cancelled once confirmed",
        "",
        "| Chip | Effect |",
        "|---|---|",
        "| Free Hit | One-week squad |",
        "| Wildcard | Permanent changes |",
        "",
        para(30, "fh"),
        "",
        "**Squad Size**",
        "",
        "Fifteen players in total per squad, always.",
        "",
        "## Bench Boost",
        "",
        para(30, "bb"),
    ]
)


def test_parse_heading_markdown_bold_and_negatives():
    assert parse_heading("## Free Hit") == (2, "Free Hit")
    assert parse_heading("#   Title with trailing ##") == (1, "Title with trailing")
    assert parse_heading("**Squad Size**") == (3, "Squad Size")
    assert parse_heading("**Selecting Your Initial Squad** ") == (3, "Selecting Your Initial Squad")
    assert parse_heading("**Rules:**") == (3, "Rules")
    assert parse_heading("**This is a full bold sentence that ends with a period.**") is None
    assert parse_heading("**" + " ".join(["word"] * 13) + "**") is None  # слишком длинно
    assert parse_heading("plain text") is None
    assert parse_heading("#hashtag") is None


def test_split_sections_builds_two_level_paths_and_skips_document_h1():
    sections = split_sections(DOC)
    headings = [s.heading for s in sections]
    assert headings[0] == ""  # пролог до первого заголовка
    assert "Wildcard" in headings
    assert "Wildcard > When to play" in headings  # H1 документа в путь не входит
    assert "Free Hit > Rules" in headings
    assert "Free Hit > Squad Size" in headings
    assert headings[-1] == "Bench Boost"
    assert all(s.paragraphs for s in sections)  # пустых разделов нет


def test_split_sections_understands_underline_headings_and_drops_empty_ones():
    text = "Title\n=====\n\n## Empty\n## Full\n\nBody text with enough words in it.\n"
    sections = split_sections(text)
    assert [s.heading for s in sections] == ["Full"]  # «Empty» без абзацев не создаётся
    assert sections[0].paragraphs == ["Body text with enough words in it."]


def test_kb_paragraphs_keep_lists_and_tables_but_filter_boilerplate():
    block = (
        "- Published2 days ago\n"
        "Real paragraph with several words.\n"
        "- 2 Goalkeepers\n"
        "- 5 Defenders\n"
        "1. First step\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "ok\n"
    )
    paras = split_kb_paragraphs(block)
    assert paras == [
        "Real paragraph with several words.",
        "- 2 Goalkeepers\n- 5 Defenders\n1. First step",  # короткие пункты не выброшены
        "| a | b |\n| 1 | 2 |",  # разделитель шапки таблицы удалён
    ]


def test_chunks_start_with_title_and_heading_path_and_preserve_order():
    chunks = chunk_document(title="Chips Guide | Site | Site", text=DOC, max_tokens=60, count=words)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.text.startswith("Chips Guide\n") for c in chunks)  # хвост сайта срезан
    assert any(c.text.startswith("Chips Guide\nWildcard > When to play\n") for c in chunks)
    fh = [c for c in chunks if "Free Hit" in c.text.split("\n")[1]]
    assert fh and "- 2 Goalkeepers\n- 5 Defenders" in "\n".join(c.text for c in fh)
    assert "| Free Hit | One-week squad |" in "\n".join(c.text for c in fh)
    # порядок разделов: Wildcard раньше Free Hit раньше Bench Boost
    joined = "\n".join(c.text for c in chunks)
    assert joined.index("wcword") < joined.index("Goalkeepers") < joined.index("bbword")
    # бюджет тела = max(MIN_BODY_BUDGET, лимит − префикс) плюс короткий хвост, который _merge
    # приклеивает к предыдущему чанку (то же поведение, что у новостного чанкера)
    for c in chunks:
        body = c.text.split("\n", 2)[-1]
        assert words(body) <= max(MIN_BODY_BUDGET, 60) + MIN_TAIL_TOKENS or "\n" not in body


def test_tiny_section_is_folded_into_previous_with_inline_heading():
    text = "## Big\n\n" + para(50, "big") + "\n\n## Tiny\n\nJust three words here.\n"
    chunks = chunk_document(title="T", text=text, max_tokens=200, count=words)
    assert len(chunks) == 1
    assert "Tiny: Just three words here." in chunks[0].text
    assert words("Just three words here.") < MIN_TAIL_TOKENS


def test_document_without_headings_and_title_dedup():
    chunks = chunk_document(
        title="Only Title",
        text="Only Title\nBody with a handful of words.",
        max_tokens=100,
        count=words,
    )
    assert len(chunks) == 1
    assert chunks[0].text == "Only Title\nBody with a handful of words."
    assert chunk_document(title="Empty", text="", count=words)[0].text == "Empty"


def test_clean_title_strips_site_suffixes_only():
    assert clean_title("FPL Beginner's Guide | FPLWatch | FPLWatch") == "FPL Beginner's Guide"
    long_suffix = (
        "Best FPL Tips, Advice, Team News, Picks, and Statistics from Fantasy Football Scout"
    )
    assert clean_title(f"What is EO? - {long_suffix}") == "What is EO?"
    assert (
        clean_title("Wildcard - When to use") == "Wildcard - When to use"
    )  # короткий хвост — часть названия
    assert clean_title("  spaced   title ") == "spaced title"
