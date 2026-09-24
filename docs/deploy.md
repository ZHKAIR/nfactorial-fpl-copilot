# Выкладка на VPS (Ubuntu 24.04 / 26.04)

Пошаговая инструкция для первого деплоя: один сервер, Docker Compose, HTTPS через Caddy,
вход в приложение по паролю. Время — около часа. Всё, что нужно, уже лежит в репозитории:

| файл | что делает |
|---|---|
| `docker-compose.prod.yml` | стек: `db` (Postgres + pgvector, наружу закрыт), `app` (Streamlit, только для Caddy), `ingest` (цикл новостей каждые 30 мин), `caddy` (80/443, авто-HTTPS) |
| `deploy/Caddyfile` | обратный прокси на Streamlit (вебсокеты работают), сертификат Let's Encrypt для `DOMAIN` |
| `.env.prod.example` | все переменные с комментариями — копируется в `.env` на сервере |
| `scripts/db_dump.sh` / `scripts/db_restore.sh` | перенос базы с ноутбука на сервер |
| `scripts/deploy.sh` | обновление: `git pull` + сборка + перезапуск |
| `scripts/db_backup.sh` | ежедневный бэкап базы на сервере (cron) |

## 1. Что арендовать

- Любой облачный VPS: **Hetzner Cloud** (Create Server → Falkenstein / Nuremberg / Helsinki),
  Gcore, DigitalOcean и т.п. Живое демо работает на Gcore Cloud (x86_64, Ubuntu 26.04 LTS).
- Образ: **Ubuntu 24.04 или 26.04 LTS**.
- Тариф: **2 vCPU / 4 GB RAM** — x86 (линейка CX, например CX22) или ARM (линейка CAX,
  например CAX11). Подходят оба: образ собирается на сервере под его архитектуру
  (python:3.12-slim, pgvector, caddy и все Python-колёса есть для amd64 и arm64).
  Диск 40 GB хватает с запасом (образ ~1.4 GB, база ~0.3 GB).
- **SSH key**: при создании добавьте свой публичный ключ (см. шаг 2). Пароль root не нужен.
- **Firewall** (Hetzner → Firewalls, Gcore → Security groups) и/или `ufw` на сервере
  (`ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443 && ufw enable`): входящие TCP **22, 80, 443** (и UDP 443 для HTTP/3), всё
  остальное закрыть. Порт Postgres наружу не публикуется самим compose-файлом.

Примерная стоимость (сверьте актуальные цены на hetzner.com и openai.com/pricing — они
меняются):

| статья | в месяц |
|---|---|
| сервер 2 vCPU / 4 GB | ≈ €4–7 (+ IPv4 ≈ €0.5–1) |
| OpenAI (gpt-4o-mini: чат ≈ $0.001–0.005 за вопрос, эмбеддинги новостей и разбор новостей) | ≈ $1–10 при нескольких пользователях |
| The Odds API, LangSmith | бесплатные тарифы |

Поставьте лимит трат в кабинете OpenAI (Settings → Limits) — это жёсткий потолок; в
приложении есть ещё мягкий суточный лимит `APP_DAILY_LLM_LIMIT`.

## 2. SSH-ключ (на своём компьютере)

```bash
ssh-keygen -t ed25519 -C "fpl-copilot"      # Enter на все вопросы
cat ~/.ssh/id_ed25519.pub                   # вставьте в Hetzner при создании сервера
ssh root@<IP-сервера>                       # первый вход
```

## 3. Подготовка сервера

```bash
# пользователь для приложения (не root)
adduser --disabled-password --gecos "" deploy
usermod -aG sudo deploy
mkdir -p /home/deploy/.ssh && cp ~/.ssh/authorized_keys /home/deploy/.ssh/
chown -R deploy:deploy /home/deploy/.ssh && chmod 700 /home/deploy/.ssh
echo "deploy ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/deploy

# обновления и Docker: официальный скрипт (Docker Engine + compose plugin); если он не знает
# свежий релиз Ubuntu (так бывает первые месяцы после выхода, например 26.04), — пакеты Ubuntu
apt update && apt -y upgrade
curl -fsSL https://get.docker.com | sh \
  || apt -y install docker.io docker-compose-v2 docker-buildx
systemctl enable --now docker
usermod -aG docker deploy

# вход по паролю по SSH выключить (только ключи)
sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart ssh
exit
```

Дальше всё — от пользователя `deploy`: `ssh deploy@<IP-сервера>`. Проверка: `docker compose version`.

## 4. Код с GitHub

Репозиторий публичный — клонируется без ключей:

```bash
git clone https://github.com/ZHKAIR/nfactorial-fpl-copilot.git ~/fpl-copilot
cd ~/fpl-copilot
```

Для приватного форка серверу нужен **deploy key** (ключ только на чтение этого репозитория):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/github_deploy -N "" -C "fpl-copilot-server"
cat ~/.ssh/github_deploy.pub
# GitHub → репозиторий → Settings → Deploy keys → Add deploy key (Allow write — не нужно)
cat >> ~/.ssh/config <<'EOF'
Host github.com
  IdentityFile ~/.ssh/github_deploy
  IdentitiesOnly yes
EOF
git clone git@github.com:<ваш-логин>/<репозиторий>.git ~/fpl-copilot
```

## 5. Переменные окружения

```bash
cp .env.prod.example .env
nano .env          # DOMAIN, APP_PASSWORD, POSTGRES_PASSWORD, OPENAI_API_KEY, ODDS_API_KEY
chmod 600 .env
```

- **DOMAIN**: свой домен (A-запись на IP сервера) или без домена — `<IP>.sslip.io`, например
  `203.0.113.10.sslip.io`. Caddy сам получит сертификат Let's Encrypt (порты 80 и 443 должны быть
  открыты).
- **APP_PASSWORD**: пароль для входа (`openssl rand -base64 18`) — отправьте его проверяющим.
- **POSTGRES_PASSWORD**: `openssl rand -hex 24`.
- `APP_DEFAULT_MANAGER_ID` — пустой: каждый вводит свой ID FPL (запоминается в браузере).
- `XPTS_FIXTURE_PROVIDER=odds` + `ODDS_API_KEY` — сила соперников по коэффициентам.

## 6. Перенос базы с ноутбука

На ноутбуке (в папке проекта, локальный Postgres `fpl-copilot-db` запущен):

```bash
scripts/db_dump.sh                                   # -> backups/fpl-YYYYmmdd-HHMM.dump
scp backups/fpl-*.dump deploy@<IP-сервера>:~/fpl-copilot/backups/
```

На сервере (сначала создайте каталог: `mkdir -p ~/fpl-copilot/backups`):

```bash
cd ~/fpl-copilot
scripts/db_restore.sh backups/fpl-YYYYmmdd-HHMM.dump  # поднимет db и восстановит дамп
```

Скрипт печатает число статей, сигналов и строк истории — они должны совпасть с локальными.

## 7. Запуск и проверка

```bash
docker compose -f docker-compose.prod.yml up -d --build   # первая сборка 3–8 минут
docker compose -f docker-compose.prod.yml ps              # db и app — healthy, caddy и ingest — running
curl -I https://<DOMAIN>/_stcore/health                   # HTTP/2 200
```

Откройте `https://<DOMAIN>` → экран входа с гербом → пароль → «Брифинг». Введите ID менеджера
FPL в сайдбаре. Миграции БД применяются автоматически при старте контейнеров.

## 8. Обновление

```bash
cd ~/fpl-copilot && scripts/deploy.sh      # git pull --ff-only, сборка, перезапуск, чистка образов
```

## 9. Бэкапы

```bash
crontab -e
# каждый день в 03:15: дамп базы в ~/fpl-copilot/backups, хранится 14 дней
15 3 * * * cd /home/deploy/fpl-copilot && scripts/db_backup.sh >> backups/backup.log 2>&1
```

Забрать бэкап себе: `scp deploy@<IP>:~/fpl-copilot/backups/prod-*.dump .`. Восстановить —
`scripts/db_restore.sh <файл>`. В Hetzner можно дополнительно включить Backups сервера (+20 %).

## 10. Логи и частые проблемы

```bash
docker compose -f docker-compose.prod.yml logs -f app       # Streamlit
docker compose -f docker-compose.prod.yml logs -f ingest    # цикл новостей
docker compose -f docker-compose.prod.yml logs caddy        # сертификат, запросы
docker stats                                                # память по контейнерам
```

Логи Docker ограничены (json-file, 3 × 10 MB на контейнер), память — `mem_limit` в compose (db 1 GB,
app 1.6 GB, ingest 768 MB, caddy 128 MB).

| симптом | что проверить |
|---|---|
| сайт не открывается, `caddy` пишет про ACME / challenge | DNS домена указывает на IP сервера; в Firewall открыты 80 и 443; для `sslip.io` IP в имени совпадает с сервером |
| «База данных недоступна» в сайдбаре | `docker compose ... ps` — `db` healthy? `POSTGRES_PASSWORD` в `.env` не меняли после первого запуска (пароль задаётся при создании тома) |
| страница «висит» / переподключается | вебсокеты: Caddy проксирует их сам; не ставьте перед сервером другой прокси без поддержки WebSocket |
| нет xPts / пустые прогнозы | база не восстановлена (шаг 6) или ingest ещё не прошёл цикл |
| «Суточный лимит запросов к ИИ исчерпан» | увеличьте `APP_DAILY_LLM_LIMIT` в `.env` и `docker compose -f docker-compose.prod.yml up -d` |
| кончилась память при сборке на 2 GB | используйте тариф 4 GB или добавьте swap: `sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` |

## Проверено локально (25.09.2026)

Стек поднят в изолированном проекте `docker compose -f docker-compose.prod.yml -p fplc-prodtest`
на Apple Silicon (arm64) с `DOMAIN=localhost` (внутренний CA Caddy) и портами 18080/18443: дамп
локальной базы `scripts/db_dump.sh` — 28 MB, восстановление `scripts/db_restore.sh` (1770
статей, 308 сигналов, 3216 строк истории), `app` healthy, `https://localhost:18443/_stcore/health`
= 200, экран входа → неверный пароль → «Неверный пароль», верный → «Брифинг» с живыми данными;
один цикл `ingest --once --limit 2` — 9 новых статей. Образ `fpl-copilot:prod` — 1.36 GB.
После проверки тестовый стек и его тома удалены (`down -v`).
