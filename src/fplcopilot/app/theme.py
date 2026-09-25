"""Визуальная система UI: эмблема, токены цвета, типографика, палитра сложности матча, общий CSS.

Язык дизайна — по референсу Figma «FPL Copilot UI/UX Exploration», светлый вариант: светлые
шапка, сайдбар и фон, белые карточки с тонкой рамкой, синий эмблемы #06A8FD — главное и
рекомендация. Семантика цвета: синий — рекомендация, зелёный — плюс, жёлтый — риск, красный —
проблема. Типографика — чёткая иерархия: мелкая подпись секции капсом → крупный заголовок
карточки → текст → крупные числа. Шрифты с кириллицей: Manrope — весь текст (одна строка —
один шрифт), JetBrains Mono — только числа и короткие метки.

Тема по умолчанию — светлая: Streamlit по умолчанию следует настройке ОС, поэтому при первом
открытии `DEFAULT_LIGHT_JS` записывает выбор «Light» туда же, куда его пишет меню ⋮ → Light
(localStorage), и перезагружает страницу; выбор пользователя в меню дальше не трогается.

Цвета темы Streamlit (светлая / тёмная) — в `.streamlit/config.toml` рядом с Home.py. Здесь —
то, чего конфиг не умеет: наши HTML-компоненты (`ui_kit`), таблицы, подсказки и полировка
элементов Streamlit. Streamlit ставит `color-scheme` на контейнер приложения, поэтому
`light-dark()` в CSS выбирает вариант под активную тему без Python-логики.
"""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"
LOGO = STATIC / "logo.svg"  # круглая эмблема на прозрачном фоне (st.logo)
FAVICON = STATIC / "favicon.png"

BRAND = "#06A8FD"  # синий с эмблемы
PRIMARY = "#0277B8"  # кнопки / виджеты: тот же тон темнее, белый текст читается

SANS = "Manrope, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace"

# Сложность матча 1–5 (0 — нет матча): (фон, текст) для светлой и тёмной темы. Крайние
# значения насыщенные, средние — приглушённые: глаз сразу ловит очень лёгкие и очень тяжёлые.
FSI_PALETTE: dict[int, tuple[tuple[str, str], tuple[str, str]]] = {
    0: (("#EEF2F5", "#8A979F"), ("#141C22", "#6F7F88")),
    1: (("#147A44", "#FFFFFF"), ("#1E9A58", "#FFFFFF")),
    2: (("#B5E2C4", "#0C2A17"), ("#1F4A33", "#CFEBDB")),
    3: (("#E4E9EC", "#2A3036"), ("#253039", "#C9D2D7")),
    4: (("#F7C3B7", "#3D120B"), ("#5C2A27", "#F6D2CC")),
    5: (("#C8373E", "#FFFFFF"), ("#D4424A", "#FFFFFF")),
}


def fsi_class(level: int | None) -> str:
    """CSS-класс фона по сложности 1–5; «» — сложности нет; 0 — тур без матча."""
    if level is None or int(level) not in FSI_PALETTE:
        return ""
    return f"fpl-fsi-{int(level)}"


FSI_CSS_RULES = "\n".join(
    f".fpl-fsi-{k} {{ background: light-dark({lb}, {db}); color: light-dark({lf}, {df}); }}"
    for k, ((lb, lf), (db, df)) in FSI_PALETTE.items()
)

# Токены — на :root; `light-dark()` внутри var() вычисляется в месте использования, где
# Streamlit уже выставил color-scheme темы.
TOKENS_CSS = f"""
:root {{
  --fpl-brand: {BRAND};
  --fpl-primary: {PRIMARY};
  --fpl-sans: {SANS};
  --fpl-mono: {MONO};
  --fpl-accent: light-dark(#0277B8, #4CC3FF);
  --fpl-accent-bg: light-dark(#E6F6FE, #0A2533);
  --fpl-accent-line: light-dark(#8FD5FA, #145A7C);
  --fpl-text: light-dark(#12202A, #E6EDF1);
  --fpl-text-2: light-dark(#3B4B55, #B9C6CD);
  --fpl-muted: light-dark(#667883, #93A3AC);
  --fpl-faint: light-dark(#94A3AD, #62737C);
  --fpl-bg: light-dark(#F4F7F9, #0B1116);
  --fpl-chrome: light-dark(#FFFFFF, #0E151B);
  --fpl-line: light-dark(#E3EAEE, #1E2B33);
  --fpl-line-strong: light-dark(#D6E0E6, #2A3A44);
  --fpl-surface: light-dark(#FFFFFF, #111A20);
  --fpl-surface-2: light-dark(#F7FAFB, #16212A);
  --fpl-hover: light-dark(rgba(2, 119, 184, 0.05), rgba(255, 255, 255, 0.04));
  --fpl-good: light-dark(#15803D, #3FB97A);
  --fpl-good-bg: light-dark(#E8F6EE, #12291D);
  --fpl-warn: light-dark(#B45309, #E0B040);
  --fpl-warn-bg: light-dark(#FEF6E4, #2A2112);
  --fpl-warn-line: light-dark(#F2C94C, #8C6A1E);
  --fpl-bad: light-dark(#C8373E, #F06A6F);
  --fpl-bad-bg: light-dark(#FDECEC, #2E1517);
  --fpl-tip-bg: light-dark(#12202A, #F0F4F6);
  --fpl-tip-fg: light-dark(#F0F4F6, #12202A);
  --fpl-pitch-a: light-dark(#2E7A52, #1C4A33);
  --fpl-pitch-b: light-dark(#338457, #20543A);
  --fpl-shadow: light-dark(0 1px 2px rgba(18, 32, 42, 0.05), 0 1px 2px rgba(0, 0, 0, 0.3));
}}
"""

# Полировка элементов Streamlit (data-testid стабильны между минорными версиями).
STREAMLIT_CSS = """
html, body, [data-testid="stApp"] { -webkit-font-smoothing: antialiased; }
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding: 5rem 2.75rem 5rem; }

/* шапка — светлая, с тонкой линией; активная вкладка — подчёркивание цветом эмблемы */
[data-testid="stHeader"] {
  background: var(--fpl-chrome); border-bottom: 1px solid var(--fpl-line); height: 3.75rem;
}
[data-testid="stTopNavLink"] { border-radius: 6px; }
[data-testid="stTopNavLink"] p { color: var(--fpl-muted); font-weight: 600; font-size: 14.5px; }
[data-testid="stTopNavLink"]:hover { background: var(--fpl-hover); }
[data-testid="stTopNavLink"]:hover p { color: var(--fpl-text); }
[data-testid="stTopNavLink"][aria-current="page"] {
  background: transparent; box-shadow: inset 0 -2px 0 var(--fpl-brand); border-radius: 0;
}
[data-testid="stTopNavLink"][aria-current="page"] p { color: var(--fpl-text); font-weight: 700; }
[data-testid="stAppDeployButton"] { display: none; }

/* шапка: Streamlit крутит коляску / велосипед / бег — прячем и ставим мяч */
@keyframes fpl-ball {
  0%, 100% { transform: translateY(1px) rotate(-10deg); }
  50% { transform: translateY(-3px) rotate(16deg); }
}
[data-testid="stStatusWidgetRunningIcon"] { position: relative !important; }
[data-testid="stStatusWidgetRunningIcon"] > * { opacity: 0 !important; }
[data-testid="stStatusWidgetRunningIcon"]::after {
  content: ""; position: absolute; inset: 1px;
  background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Ccircle cx='32' cy='32' r='30' fill='%23fff' stroke='%2306A8FD' stroke-width='3'/%3E%3Cpolygon points='32,20 40,26 37,36 27,36 24,26' fill='%2306A8FD'/%3E%3Cpath d='M32 20L32 8M40 26L52 20M37 36L46 50M27 36L18 50M24 26L12 20' fill='none' stroke='%2306A8FD' stroke-width='2.4' stroke-linecap='round'/%3E%3Cpath d='M52 20C56 26 58 32 58 36M46 50C40 56 34 58 32 58C30 58 24 56 18 50M12 20C8 26 6 32 6 36' fill='none' stroke='%2306A8FD' stroke-width='2.4' stroke-linecap='round'/%3E%3C/svg%3E") center / contain no-repeat;
  animation: fpl-ball 0.7s ease-in-out infinite;
}
@media (prefers-reduced-motion: reduce) {
  [data-testid="stStatusWidgetRunningIcon"]::after { animation: none; }
}

/* эмблема: крупно и по центру сайдбара (надписи на кольце герба читаются, поэтому отдельного
   вордмарка нет), кнопка сворачивания — в правом верхнем углу поверх свободного места;
   в шапке (сайдбар свёрнут) — заметно */
[data-testid="stSidebarHeader"] {
  position: relative; height: auto; justify-content: center;
  padding: 1.1rem 1rem 0.35rem; margin-bottom: 0.25rem;
}
[data-testid="stSidebarHeader"] > div:first-child { margin: 0 auto; }
[data-testid="stSidebarLogo"] { height: 150px; width: 150px; max-width: none; }
[data-testid="stSidebarCollapseButton"] { position: absolute; top: 0.6rem; right: 0.6rem; }
[data-testid="stHeaderLogo"] { height: 46px; width: 46px; max-width: none; }

/* заголовки */
[data-testid="stHeading"] h1 { font-weight: 750; letter-spacing: -0.03em; padding: 0 0 0.25rem; }
[data-testid="stHeading"] h2, [data-testid="stHeading"] h3 { letter-spacing: -0.02em; }
[data-testid="stHeading"] h3 { padding-top: 1.5rem; }
[data-testid="stCaptionContainer"] { color: var(--fpl-muted); }

/* карточки: контейнеры с рамкой, раскрывающиеся блоки, метрики */
[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlock"][class*="st-key-card"] {
  background: var(--fpl-surface); border-color: var(--fpl-line-strong); border-radius: 12px;
  box-shadow: var(--fpl-shadow);
}
div[data-testid="stVerticalBlock"][class*="st-key-card"] { padding: 22px 24px; }
[data-testid="stExpander"] details {
  background: var(--fpl-surface); border-color: var(--fpl-line-strong); border-radius: 10px;
}
[data-testid="stExpander"] summary p { font-weight: 600; }
[data-testid="stExpander"] summary:hover { background: var(--fpl-hover); }
[data-testid="stMetric"] {
  background: var(--fpl-surface); border: 1px solid var(--fpl-line-strong); border-radius: 12px;
  padding: 14px 18px 13px; height: 100%; box-shadow: var(--fpl-shadow);
}
[data-testid="stMetricLabel"] p {
  font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--fpl-muted);
}
[data-testid="stMetricValue"] {
  font-family: var(--fpl-mono); font-weight: 600; letter-spacing: -0.02em; line-height: 1.3;
  color: var(--fpl-text);
}
[data-testid="stAlertContainer"] { border-radius: 10px; }

/* кнопки */
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-secondary"] p,
[data-testid="stBaseButton-tertiary"] p { font-weight: 650; }
[data-testid="stBaseButton-secondary"] { background: var(--fpl-surface); }
[data-testid="stBaseButton-tertiary"] p { color: var(--fpl-accent); }

/* быстрые вопросы в чате — чипы */
.st-key-quick_prompts button { border-radius: 999px; padding: 4px 14px; min-height: 34px; }
.st-key-quick_prompts button p { font-weight: 500; font-size: 13.5px; }

/* вкладки */
[data-testid="stTabs"] [role="tab"] p { font-weight: 650; font-size: 14.5px; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: var(--fpl-text); }

/* сайдбар — светлая панель настроек */
[data-testid="stSidebarUserContent"] { padding-top: 0.25rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { font-size: 12.5px; }

/* числа в таблицах Streamlit */
[data-testid="stTable"] td, [data-testid="stTable"] th { font-variant-numeric: tabular-nums; }
"""

# Светлая тема по умолчанию: тот же ключ localStorage, что пишет меню ⋮ (Streamlit 1.64:
# `stActiveTheme-<pathname>-v2`, значение "Light" / "Dark" / "System"). Streamlit сам пишет туда
# "System" при первой загрузке, поэтому отличаем «ещё не выбирали» своей меткой
# `fplc-theme-default-<pathname>`: один раз на адрес ставим "Light" (перезагрузка — только
# если ОС в тёмной теме), дальше выбор пользователя в меню не трогаем.
DEFAULT_LIGHT_JS = """
<script>
(function () {
  if (window.__fplThemeChecked) return;
  window.__fplThemeChecked = true;
  try {
    var ls = window.localStorage, path = window.location.pathname;
    var mark = "fplc-theme-default-" + path, key = "stActiveTheme-" + path + "-v2";
    if (ls.getItem(mark)) return;
    ls.setItem(mark, "1");
    var cur = ls.getItem(key);
    if (cur === null || cur === JSON.stringify("System")) {
      ls.setItem(key, JSON.stringify("Light"));
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        window.location.reload();
      }
    }
  } catch (e) {}
})();
</script>
"""


def global_css() -> str:
    # модули компонентов импортируют theme — импорт здесь, без цикла на импорте
    from fplcopilot.app import format as fmt
    from fplcopilot.app import player_card, ui_kit

    return (
        "<style>\n"
        + TOKENS_CSS
        + STREAMLIT_CSS
        + FSI_CSS_RULES
        + fmt.TIP_CSS_RULES
        + ui_kit.KIT_CSS
        + player_card.PLAYER_CSS
        + "\n</style>"
    )


def inject() -> None:
    """Общий CSS (и выбор светлой темы по умолчанию) для всех страниц — вызывается один раз в
    Home.py до st.navigation().run()."""
    import streamlit as st  # модуль без streamlit на импорте: его читают чистые помощники format

    st.html(global_css())
    st.html(DEFAULT_LIGHT_JS, unsafe_allow_javascript=True)
