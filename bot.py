import os
import re
import time
import sqlite3
import asyncio
import json, base64
from contextlib import closing
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Optional

import aiohttp
from telegram import (
    Update,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
    InlineKeyboardButton, InlineKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# .env (опционально)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ===== Config =====
TZ = ZoneInfo("Europe/Moscow")
DB_PATH = os.getenv("DB_PATH", "subs.sqlite3")
CACHE_TTL = 60.0
CACHE: Dict[Tuple[str, Tuple[str, ...]], Tuple[float, Dict[str, float]]] = {}
SUPPORT_CHAT_ID = int(os.getenv("SUPPORT_CHAT_ID", "0"))
SUPPORT_COOLDOWN = 10

TARGETS = {
    "USD": ["UZS", "RUB", "KZT", "KGS"],
    "EUR": ["UZS", "RUB", "KZT", "KGS"],
    "RUB": ["UZS", "KZT", "KGS", "TJS"],
}
ALIAS = {"KGZ": "KGS", "RUR": "RUB"}

FLAGS = {
    "USD": "🇺🇸", "EUR": "🇪🇺", "RUB": "🇷🇺",
    "UZS": "🇺🇿", "KZT": "🇰🇿", "KGS": "🇰🇬",
    "TRY": "🇹🇷", "TJS": "🇹🇯", "JPY": "🇯🇵",
}

def flag(code: str) -> str:
    return FLAGS.get(code.upper(), "")

def targets_str_multiline() -> str:
    order = ["USD", "EUR", "RUB"]
    blocks = []
    for base in order:
        if base in TARGETS:
            lines = [f"{flag(base)} <b>{base}</b> → {flag(sym)} {sym}" for sym in TARGETS[base]]
            blocks.append("\n".join(lines))
    return "\n-------------------\n".join(blocks)

# ===== Buttons =====
BTN_RATE_USD   = "🇺🇸 Курс USD"
BTN_RATE_EUR   = "🇪🇺 Курс EUR"
BTN_RATE_RUB   = "🇷🇺 Курс RUB"
BTN_REFBOOK    = "📖 Справочник"
BTN_ENTER_PAIR = "✏️ Ввести свою пару"
BTN_SUB        = "🔔 Подписаться"
BTN_SUB_PAIR   = "📌 Подписка на пару"
BTN_STATUS     = "📊 Мои подписки"
BTN_TIME       = "⏰ Изменить время"
BTN_UNSUB      = "❌ Отписаться"
BTN_RESET      = "🧹 Очистить диалог"
BTN_SUPPORT    = "🆘 Поддержка"

# Кнопки общего назначения
BTN_ENTER_OTHER = "✏️Ввести другой"
BTN_BACK = "⬅ Назад"

# Submenus
SUB_TARGETS = {
    "USD": ["RUB", "KZT", "KGS", "UZS"],
    "EUR": ["RUB", "KZT", "KGS", "UZS"],
    "RUB": ["UZS", "KZT", "KGS", "TJS"],
}

def make_submenu_kb(base: str) -> ReplyKeyboardMarkup:
    ps = SUB_TARGETS[base]
    row1 = [
        KeyboardButton(f"{flag(base)} {base} - {flag(ps[0])} {ps[0]}"),
        KeyboardButton(f"{flag(base)} {base} - {flag(ps[1])} {ps[1]}"),
    ]
    row2 = [
        KeyboardButton(f"{flag(base)} {base} - {flag(ps[2])} {ps[2]}"),
        KeyboardButton(f"{flag(base)} {base} - {flag(ps[3])} {ps[3]}"),
    ]
    row3 = [KeyboardButton(BTN_ENTER_OTHER), KeyboardButton(BTN_BACK)]
    return ReplyKeyboardMarkup([row1, row2, row3], resize_keyboard=True)

USD_SUB_KB = make_submenu_kb("USD")
EUR_SUB_KB = make_submenu_kb("EUR")
RUB_SUB_KB = make_submenu_kb("RUB")

MAIN_KB = ReplyKeyboardMarkup(
    [
        [KeyboardButton(BTN_RATE_USD), KeyboardButton(BTN_RATE_EUR)],
        [KeyboardButton(BTN_RATE_RUB), KeyboardButton(BTN_REFBOOK)],
        [KeyboardButton(BTN_ENTER_PAIR), KeyboardButton(BTN_SUB_PAIR)],
        [KeyboardButton(BTN_SUB), KeyboardButton(BTN_UNSUB)],
        [KeyboardButton(BTN_STATUS), KeyboardButton(BTN_RESET)],
        [KeyboardButton(BTN_TIME), KeyboardButton(BTN_SUPPORT)],
    ],
    resize_keyboard=True,
    input_field_placeholder="Выберите действие…",
)

def inline_sub_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔔 Подписаться (09:00)", callback_data="sub:09:00")],
         [InlineKeyboardButton("❎ Отписаться", callback_data="unsub")]]
    )

def time_kb() -> InlineKeyboardMarkup:
    presets = ["08:00", "09:00", "10:00", "12:00", "15:00", "18:00", "21:00"]
    row1 = [InlineKeyboardButton(t, callback_data=f"time:{t}") for t in presets[:4]]
    row2 = [InlineKeyboardButton(t, callback_data=f"time:{t}") for t in presets[4:]]
    other = [InlineKeyboardButton("Другое время…", callback_data="time:custom")]
    return InlineKeyboardMarkup([row1, row2, other])

# ===== DB: миграции =====
def _drop_wrong_unique_indexes(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA index_list('pair_subscriptions')")
    idx_rows = cur.fetchall()
    for _, name, unique, *_ in idx_rows:
        if not unique:
            continue
        cols = [r[2] for r in conn.execute(f"PRAGMA index_info('{name}')").fetchall()]
        if cols == ["chat_id"] or (len(cols) == 1 and cols[0] == "chat_id"):
            try:
                conn.execute(f'DROP INDEX IF EXISTS "{name}"')
            except Exception:
                pass

def _pair_table_has_unique_chat_id(conn: sqlite3.Connection) -> bool:
    cols = conn.execute("PRAGMA table_info('pair_subscriptions')").fetchall()
    pk_cols = [c for c in cols if c[5] == 1]
    if any(c[1] == "chat_id" for c in pk_cols):
        return True
    return False

def _recreate_pair_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pair_subscriptions_new (
            chat_id INTEGER NOT NULL,
            base TEXT NOT NULL,
            quote TEXT NOT NULL,
            hour INTEGER NOT NULL,
            minute INTEGER NOT NULL
        )
    """)
    conn.execute("""
        INSERT INTO pair_subscriptions_new (chat_id, base, quote, hour, minute)
        SELECT chat_id, base, quote,
               MAX(hour) AS hour, MAX(minute) AS minute
        FROM pair_subscriptions
        GROUP BY chat_id, base, quote
    """)
    conn.execute("DROP TABLE pair_subscriptions")
    conn.execute("ALTER TABLE pair_subscriptions_new RENAME TO pair_subscriptions")

def db_init() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                chat_id INTEGER PRIMARY KEY,
                hour INTEGER NOT NULL,
                minute INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pair_subscriptions (
                chat_id INTEGER NOT NULL,
                base TEXT NOT NULL,
                quote TEXT NOT NULL,
                hour INTEGER NOT NULL,
                minute INTEGER NOT NULL
            )
        """)
        _drop_wrong_unique_indexes(conn)
        if _pair_table_has_unique_chat_id(conn):
            _recreate_pair_table(conn)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_pair_subscriptions_unique
            ON pair_subscriptions(chat_id, base, quote)
        """)
        conn.execute("""
            DELETE FROM pair_subscriptions
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM pair_subscriptions
                GROUP BY chat_id, base, quote
            )
        """)
        conn.commit()

# ===== DB ops =====
def db_get_all() -> List[Tuple[int, int, int]]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return list(conn.execute("SELECT chat_id, hour, minute FROM subscriptions").fetchall())

def db_upsert(chat_id: int, hour: int, minute: int) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
            INSERT INTO subscriptions(chat_id, hour, minute)
            VALUES (?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET hour=excluded.hour, minute=excluded.minute
        """, (chat_id, hour, minute))
        conn.commit()

def db_delete(chat_id: int) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("DELETE FROM subscriptions WHERE chat_id=?", (chat_id,))
        conn.commit()

def db_upsert_pair(chat_id: int, base: str, quote: str, hour: int, minute: int) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("DELETE FROM pair_subscriptions WHERE chat_id=? AND base=? AND quote=?",
                     (chat_id, base, quote))
        conn.execute("INSERT INTO pair_subscriptions(chat_id, base, quote, hour, minute) VALUES(?,?,?,?,?)",
                     (chat_id, base, quote, hour, minute))
        conn.commit()

def db_get_pairs(chat_id: int) -> List[Tuple[str, str, int, int]]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute("""
            SELECT base, quote, hour, minute
            FROM pair_subscriptions
            WHERE chat_id=?
            ORDER BY base, quote
        """, (chat_id,))
        return list(cur.fetchall())

def db_get_all_pairs() -> List[Tuple[int, str, str, int, int]]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute("SELECT chat_id, base, quote, hour, minute FROM pair_subscriptions")
        return list(cur.fetchall())

def db_delete_pair(chat_id: int, base: Optional[str]=None, quote: Optional[str]=None) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if base and quote:
            conn.execute("DELETE FROM pair_subscriptions WHERE chat_id=? AND base=? AND quote=?", (chat_id, base, quote))
        else:
            conn.execute("DELETE FROM pair_subscriptions WHERE chat_id=?", (chat_id,))
        conn.commit()

def db_has_default(chat_id: int) -> bool:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT 1 FROM subscriptions WHERE chat_id=?",
            (chat_id,)
        ).fetchone()
        return bool(row)

def db_update_all_pairs_time(chat_id: int, hour: int, minute: int) -> list[tuple[str, str]]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute(
            "SELECT base, quote FROM pair_subscriptions WHERE chat_id=?",
            (chat_id,)
        )
        pairs = [(b, q) for (b, q) in cur.fetchall()]
        if pairs:
            conn.executemany(
                "UPDATE pair_subscriptions SET hour=?, minute=? "
                "WHERE chat_id=? AND base=? AND quote=?",
                [(hour, minute, chat_id, b, q) for (b, q) in pairs]
            )
            conn.commit()
    return pairs

# --- banned users ---
def db_init_banned():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS banned(
            chat_id INTEGER PRIMARY KEY
        )
        """)
        conn.commit()

def db_ban(chat_id: int):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("INSERT OR IGNORE INTO banned(chat_id) VALUES(?)", (chat_id,))
        conn.commit()

def db_unban(chat_id: int):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute("DELETE FROM banned WHERE chat_id=?", (chat_id,))
        conn.commit()

def db_is_banned(chat_id: int) -> bool:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute("SELECT 1 FROM banned WHERE chat_id=?", (chat_id,)).fetchone()
        return bool(row)

# ===== Utils =====
def normalize_code(code: str) -> Optional[str]:
    code = (code or "").strip().upper()
    if code in ALIAS: code = ALIAS[code]
    return code if re.fullmatch(r"[A-Z]{3}", code) else None

def parse_pair(s: str) -> Optional[Tuple[str, str]]:
    if not s: return None
    s = s.strip().upper()
    s = re.sub(r"\s*(?:-|/|→|->|—|\s)\s*", "-", s)
    m = re.fullmatch(r"([A-Z]{3})-([A-Z]{3})", s)
    if not m: return None
    base, quote = normalize_code(m.group(1)), normalize_code(m.group(2))
    if not base or not quote or base == quote: return None
    return base, quote

def remove_flags(s: str) -> str:
    return re.sub(r'[\U0001F1E6-\U0001F1FF]', '', s)

def extract_pair_from_button(text: str):
    clean = re.sub(r'\s+', ' ', remove_flags(text)).strip()
    return parse_pair(clean)

async def fetch_rates(base: str, symbols: List[str]) -> Dict[str, float]:
    base = base.upper()
    symbols = [ALIAS.get(s.upper(), s.upper()) for s in symbols]
    key = (base, tuple(sorted(symbols)))
    now_ts = time.time()

    cached = CACHE.get(key)
    if cached and (now_ts - cached[0] < CACHE_TTL):
        return cached[1]

    # provider 1
    try:
        url = f"https://open.er-api.com/v6/latest/{base}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=12) as resp:
                resp.raise_for_status()
                data = await resp.json()
        if data.get("result") == "success" and isinstance(data.get("rates"), dict):
            payload = data["rates"]
            out: Dict[str, float] = {s: float(payload[s]) for s in symbols if s in payload}
            if out:
                CACHE[key] = (now_ts, out)
                return out
    except Exception:
        pass

    # provider 2
    try:
        to_param = ",".join(symbols)
        url2 = f"https://api.frankfurter.app/latest?from={base}&to={to_param}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url2, timeout=12) as resp:
                resp.raise_for_status()
                d2 = await resp.json()
        rates2 = d2.get("rates") or {}
        out2 = {k.upper(): float(v) for k, v in rates2.items() if k.upper() in set(symbols)}
        if out2:
            CACHE[key] = (now_ts, out2)
            return out2
    except Exception:
        pass

    raise ValueError("Не удалось получить курс. Попробуйте позже.")

async def fetch_currency_by_country(query: str) -> List[str]:
    q = (query or "").strip()
    if not q: return []
    for url in [
        f"https://restcountries.com/v3.1/translation/{q}?fields=currencies,name,translations",
        f"https://restcountries.com/v3.1/name/{q}?fields=currencies,name,translations",
    ]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=12) as resp:
                    if resp.status != 200: continue
                    data = await resp.json()
            if isinstance(data, list) and data:
                codes = set()
                for item in data:
                    curr = item.get("currencies") or {}
                    for code in curr.keys():
                        if re.fullmatch(r"[A-Z]{3}", code.upper()):
                            codes.add(code.upper())
                if codes: return sorted(codes)
        except Exception:
            return []
    return []

def parse_hhmm(arg: Optional[str]) -> Tuple[int, int]:
    if not arg: return 9, 0
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", arg.strip())
    if not m: raise ValueError("Время должно быть в формате HH:MM, например 08:30.")
    h, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mm <= 59): raise ValueError("Недопустимое время. Часы 0–23, минуты 0–59.")
    return h, mm

def fmt_value(base: str, val: float) -> str:
    return f"{val:.2f}"

def format_block(base: str, rates: Dict[str, float]) -> str:
    lines = [f"{flag(base)} <b>{base}</b>:"]
    for sym in TARGETS[base]:
        v = rates.get(sym)
        if v is None:
            lines.append(f"{flag(base)} {base} → {flag(sym)} <b>{sym}</b>: <i>нет данных</i>")
        else:
            lines.append(f"{flag(base)} {base} → {flag(sym)} <b>{sym}</b>: <code>{fmt_value(base, v)}</code>")
    return "\n".join(lines)

# ===== Commands =====
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет! Выбирай кнопки ниже. Для USD/EUR/RUB есть подменю.", parse_mode=ParseMode.HTML)
    msg = await update.message.reply_text("Главное меню 👇", reply_markup=MAIN_KB)
    context.chat_data["kb_msg_id"] = msg.message_id

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)

async def cmd_usd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["submenu"] = "USD"
    await update.message.reply_text("Выберите пару USD → … или «Ввести другой».", reply_markup=USD_SUB_KB)

async def cmd_eur(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["submenu"] = "EUR"
    await update.message.reply_text("Выберите пару EUR → … или «Ввести другой».", reply_markup=EUR_SUB_KB)

async def cmd_rub(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["submenu"] = "RUB"
    await update.message.reply_text("Выберите пару RUB → … или «Ввести другой».", reply_markup=RUB_SUB_KB)

async def cmd_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    for k in ("submenu", "wait_base", "wait_country", "wait_pair", "wait_pair_sub"):
        context.user_data.pop(k, None)
    msg = await update.message.reply_text("Главное меню 👇", reply_markup=MAIN_KB)
    context.chat_data["kb_msg_id"] = msg.message_id

async def cmd_hide(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Клавиатура скрыта. Вернуть: /menu", reply_markup=ReplyKeyboardRemove())

async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        arg = context.args[0] if context.args else None
        hour, minute = parse_hhmm(arg)
        chat_id = update.effective_chat.id
        db_upsert(chat_id, hour, minute)
        schedule_job_for(context.application, chat_id, hour, minute)
        await update.message.reply_text(
            "✅ Вы подписаны на автообновление по умолчанию:\n"
            + targets_str_multiline()
            + f"\nБудни в {hour:02d}:{minute:02d} ({TZ.key})",
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        await update.message.reply_text(f"⚠️ {e}")

async def cmd_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Выберите время рассылки по будням:", reply_markup=time_kb())

async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    db_delete(chat_id)
    if scheduler and scheduler.get_job(f"digest_{chat_id}"):
        scheduler.remove_job(f"digest_{chat_id}")
    pairs = db_get_pairs(chat_id)
    for b, q, h, m in pairs:
        job_id = f"pair_{chat_id}_{b}_{q}"
        if scheduler and scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
    db_delete_pair(chat_id)
    await update.message.reply_text("❎ Ежедневные рассылки отключены.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT hour, minute FROM subscriptions WHERE chat_id=?",
            (chat_id,)
        ).fetchone()

    lines: List[str] = []
    if row:
        hour, minute = row
        lines.append(f"🕰️ <b>Сводка</b>:\nБудни в {hour:02d}:{minute:02d} ({TZ.key}).")
        lines.append("<b>Ваши пары</b>:")
        lines.append(targets_str_multiline())

    pairs = db_get_pairs(chat_id)
    if pairs:
        lines.append("📌 Пары подписки:")
        for b, q, h, m in pairs:
            lines.append(f" • {flag(b)}{b}-{flag(q)}{q} в {h:02d}:{m:02d} ({TZ.key})")

    if not lines:
        lines.append("Подписок нет. Нажмите «Подписаться» или «Подписаться на пару».")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

def _export_payload(chat_id: int) -> dict:
    data = {"version": 1}
    with closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute("SELECT hour, minute FROM subscriptions WHERE chat_id=?", (chat_id,)).fetchone()
        data["default"] = {"hour": row[0], "minute": row[1]} if row else None
        pairs = conn.execute("SELECT base, quote, hour, minute FROM pair_subscriptions WHERE chat_id=?",
                             (chat_id,)).fetchall()
        data["pairs"] = [{"base": b, "quote": q, "hour": h, "minute": m} for b, q, h, m in pairs]
    return data

async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    payload = _export_payload(chat_id)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    token = base64.urlsafe_b64encode(raw).decode("ascii")
    text = ("📤 Резервная копия подписок готова.\n"
            "Скопируйте команду и выполните её у другого бота/на другом устройстве:\n\n"
            f"/import {token}")
    await update.message.reply_text(text)

async def cmd_import(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Передайте данные после команды, например: /import <строка>")
        return
    chat_id = update.effective_chat.id
    try:
        token = context.args[0]
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
        if "version" not in payload:
            raise ValueError("Неверный формат")

        if payload.get("default"):
            h = int(payload["default"]["hour"]); m = int(payload["default"]["minute"])
            db_upsert(chat_id, h, m); schedule_job_for(context.application, chat_id, h, m)

        db_delete_pair(chat_id)
        for item in payload.get("pairs", []):
            b = normalize_code(item["base"]); q = normalize_code(item["quote"])
            h = int(item["hour"]); m = int(item["minute"])
            if not b or not q or b == q:
                continue
            db_upsert_pair(chat_id, b, q, h, m)
            schedule_pair_job(context.application, chat_id, b, q, h, m)

        await update.message.reply_text("📥 Импорт выполнен. Подписки восстановлены.", reply_markup=MAIN_KB)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Не удалось импортировать: {e}")

async def cmd_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Формат: /now USD-RUB")
        return

    query = " ".join(context.args)
    try:
        pair = parse_pair(remove_flags(query))
        if not pair:
            await update.message.reply_text("Неверная пара. Пример: /now EUR-USD")
            return

        base, quote = pair
        rates = await fetch_rates(base, [quote])
        val = rates.get(quote)
        if val is None:
            await update.message.reply_text(f"Нет данных для пары {base}-{quote}")
        else:
            await update.message.reply_text(
                f"{flag(base)} {base} → {flag(quote)} {quote} {fmt_value(base, val)}"
            )
    except Exception as e:
        await update.message.reply_text(f"⚠️ {e}")

async def cmd_support(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not SUPPORT_CHAT_ID:
        await update.message.reply_text("Служба поддержки не подключена.")
        return
    context.user_data["wait_support"] = True
    await update.message.reply_text(
        "Напишите одним сообщением, что у вас не работает. "
        "Я перешлю в поддержку. Для отмены — напишите: Отмена.",
        reply_markup=MAIN_KB
    )

async def on_support_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or update.effective_chat.id != SUPPORT_CHAT_ID:
        return
    if not update.message.reply_to_message:
        return

    support_map = context.application.bot_data.get("support_map", {})
    replied_id = update.message.reply_to_message.message_id
    target_chat_id = support_map.get(replied_id)

    if not target_chat_id:
        src = update.message.reply_to_message.text_html or update.message.reply_to_message.text or ""
        m = re.search(r"chat_id:\s*<code>(-?\d+)</code>", src)
        if m:
            target_chat_id = int(m.group(1))
    if not target_chat_id:
        await update.message.reply_text("Не удалось определить получателя. Ответьте именно на карточку обращения.")
        return

    if update.message.text and update.message.text.strip().lower().startswith("/close"):
        try:
            await context.bot.send_message(target_chat_id, "🔒 Диалог с поддержкой закрыт. Если появятся вопросы — напишите ещё раз.")
            await update.message.reply_text("🔒 Закрыто.")
        except Exception as e:
            await update.message.reply_text(f"⚠️ Не удалось закрыть: {e}")
        return

    if update.message.text:
        txt = update.message.text.strip().lower()
        if txt.startswith("/ban"):
            db_ban(target_chat_id)
            await update.message.reply_text(f"⛔ Пользователь {target_chat_id} забанен.")
            return
        if txt.startswith("/unban"):
            db_unban(target_chat_id)
            await update.message.reply_text(f"✅ Пользователь {target_chat_id} разбанен.")
            return

    try:
        if update.message.photo or update.message.document or update.message.video or \
           update.message.audio or update.message.voice or update.message.video_note or \
           update.message.sticker:
            await context.bot.copy_message(
                chat_id=target_chat_id,
                from_chat_id=SUPPORT_CHAT_ID,
                message_id=update.message.message_id
            )
        else:
            reply_text = update.message.text or update.message.caption or ""
            if not reply_text:
                await update.message.reply_text("Пустой ответ. Напишите текстом или пришлите медиа.")
                return
            await context.bot.send_message(
                target_chat_id,
                f"👨‍💻 <b>Поддержка</b>:\n{reply_text}",
                parse_mode=ParseMode.HTML
            )
        await update.message.reply_text("✅ Доставлено пользователю.")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Не удалось отправить: {e}")

async def cmd_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    msg = f"chat_id: <code>{chat.id}</code>\n"
    if user:
        msg += f"user_id: <code>{user.id}</code>\n"
        if user.username:
            msg += f"username: @{user.username}\n"
        if user.full_name:
            msg += f"name: {user.full_name}"
    await update.message.reply_text(msg, parse_mode="HTML")

async def cmd_ban(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != SUPPORT_CHAT_ID: return
    if not update.message.reply_to_message:
        await update.message.reply_text("Ответьте этой командой на карточку обращения.")
        return
    src = update.message.reply_to_message.text_html or update.message.reply_to_message.text or ""
    m = re.search(r"chat_id:\s*<code>(-?\d+)</code>", src)
    if not m:
        await update.message.reply_text("Не нашёл chat_id в карточке.")
        return
    db_ban(int(m.group(1)))
    await update.message.reply_text("⛔ Забанен.")

async def cmd_unban(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id != SUPPORT_CHAT_ID: return
    if not context.args:
        await update.message.reply_text("Формат: /unban <chat_id>")
        return
    try:
        db_unban(int(context.args[0]))
        await update.message.reply_text("✅ Разбанен.")
    except Exception as e:
        await update.message.reply_text(f"⚠️ {e}")

# ===== Reset / очистка =====
async def reset_dialog(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    for k in ("submenu", "wait_base", "wait_country", "wait_pair", "wait_pair_sub"):
        context.user_data.pop(k, None)

    menu_msg = await context.bot.send_message(
        chat_id,
        "Главное меню 👇",
        reply_markup=MAIN_KB
    )
    context.chat_data["kb_msg_id"] = menu_msg.message_id

    async def _cleanup():
        kb_keep_id = context.chat_data.get("kb_msg_id")
        try:
            last_id = menu_msg.message_id - 1
            N = 150
            for mid in range(last_id, max(last_id - N, 1), -1):
                if kb_keep_id and mid == kb_keep_id:
                    continue
                try:
                    await context.bot.delete_message(chat_id, mid)
                    await asyncio.sleep(0.01)
                except Exception:
                    pass
        except Exception as e:
            print(f"[reset] cleanup error: {e!r}")

    asyncio.create_task(_cleanup())

    msg = await context.bot.send_message(
        chat_id,
        "👇Выберите новое действия👇",
        reply_markup=MAIN_KB
    )
    context.chat_data["kb_msg_id"] = msg.message_id

# /subpair USD-RUB [HH:MM], /unsubpair USD-RUB
async def cmd_subpair(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if not context.args:
            await update.message.reply_text("Формат: /subpair USD-RUB [HH:MM]")
            return
        base_quote = context.args[0]
        hm = context.args[1] if len(context.args) > 1 else None
        pair = parse_pair(base_quote)
        if not pair:
            await update.message.reply_text("Неверная пара. Пример: /subpair USD-RUB 09:30")
            return
        base, quote = pair
        if hm:
            hour, minute = parse_hhmm(hm)
        else:
            with closing(sqlite3.connect(DB_PATH)) as conn:
                row = conn.execute("SELECT hour, minute FROM subscriptions WHERE chat_id=?", (update.effective_chat.id,)).fetchone()
            hour, minute = (row[0], row[1]) if row else (9, 0)

        chat_id = update.effective_chat.id
        db_upsert_pair(chat_id, base, quote, hour, minute)
        schedule_pair_job(context.application, chat_id, base, quote, hour, minute)
        await update.message.reply_text(f"✅ Пара {base}-{quote} подписана. Будни в {hour:02d}:{minute:02d} ({TZ.key}).")
    except Exception as e:
        await update.message.reply_text(f"⚠️ {e}")

async def cmd_unsubpair(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if not context.args:
            await update.message.reply_text("Формат: /unsubpair USD-RUB")
            return
        pair = parse_pair(context.args[0])
        if not pair:
            await update.message.reply_text("Неверная пара. Пример: /unsubpair USD-RUB")
            return
        base, quote = pair
        chat_id = update.effective_chat.id
        job_id = f"pair_{chat_id}_{base}_{quote}"
        if scheduler and scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
        db_delete_pair(chat_id, base, quote)
        await update.message.reply_text(f"❎ Отписка от {base}-{quote} выполнена.")
    except Exception as e:
        await update.message.reply_text(f"⚠️ {e}")

# ===== Callbacks & text =====
async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    chat_id = query.message.chat.id

    if data.startswith("sub:"):
        try:
            _, hm = data.split(":", 1)
            h, m = hm.split(":")
            hour, minute = int(h), int(m)
            db_upsert(chat_id, hour, minute)
            schedule_job_for(context.application, chat_id, hour, minute)
            await context.bot.send_message(chat_id, f"✅ Подписка оформлена. Будни в {hour:02d}:{minute:02d} ({TZ.key}).")
        except Exception as e:
            await context.bot.send_message(chat_id, f"⚠️ Не удалось оформить подписку: {e}")

    elif data.startswith("time:"):
        _, payload = data.split(":", 1)

        if payload == "custom":
            context.user_data["wait_time"] = True
            await context.bot.send_message(chat_id, "Введите время в формате HH:MM (например, 09:30).")
            return

        hour, minute = map(int, payload.split(":"))

        if db_has_default(chat_id):
            db_upsert(chat_id, hour, minute)
            schedule_job_for(context.application, chat_id, hour, minute)
            await context.bot.send_message(chat_id, f"⏰ Время подписки обновлено: {hour:02d}:{minute:02d} ({TZ.key}).")
            return

        pairs = db_get_pairs(chat_id)
        if pairs:
            updated = db_update_all_pairs_time(chat_id, hour, minute)
            for b, q in updated:
                schedule_pair_job(context.application, chat_id, b, q, hour, minute)
            await context.bot.send_message(chat_id, f"⏰ Время подписок на пары обновлено: {hour:02d}:{minute:02d} ({TZ.key}).")
            return

        await context.bot.send_message(chat_id, "ℹ️ У вас пока нет подписок. Оформите подписку и затем задайте время.")
        return

    elif data == "unsub":
        await cmd_unsubscribe(update, context)

async def on_text_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()

    # режим поддержки (текст)
    if context.user_data.pop("wait_support", None):
        text = (update.message.text or "").strip()
        if text.lower() in ("отмена", "cancel", "назад"):
            await update.message.reply_text("Отменено.", reply_markup=MAIN_KB)
            return

        now = time.time()
        last = context.user_data.get("last_support_at", 0)
        if now - last < SUPPORT_COOLDOWN:
            await update.message.reply_text("Немного подождите и попробуйте снова.", reply_markup=MAIN_KB)
            return

        if not SUPPORT_CHAT_ID:
            await update.message.reply_text("Служба поддержки не подключена.", reply_markup=MAIN_KB)
            return

        user = update.effective_user
        chat = update.effective_chat
        try:
            msg = (
                "📩 <b>Сообщение в поддержку</b>\n"
                f"from: {user.full_name} (id <code>{user.id}</code>)\n"
                f"username: @{user.username if user.username else '—'}\n"
                f"chat_id: <code>{chat.id}</code>\n\n"
                f"{text}"
            )
            support_msg = await context.bot.send_message(
                SUPPORT_CHAT_ID, msg, parse_mode=ParseMode.HTML, disable_web_page_preview=True
            )
            context.application.bot_data.setdefault("support_map", {})[support_msg.message_id] = chat.id

            await update.message.reply_text("Спасибо! Сообщение отправлено в поддержку. Мы ответим здесь.", reply_markup=MAIN_KB)
            context.user_data["last_support_at"] = now
        except Exception as e:
            await update.message.reply_text(f"⚠️ Не удалось отправить: {e}", reply_markup=MAIN_KB)
        return

    # ручной ввод времени
    if context.user_data.pop("wait_time", None):
        try:
            hour, minute = parse_hhmm(text)
            chat_id = update.effective_chat.id

            if db_has_default(chat_id):
                db_upsert(chat_id, hour, minute)
                schedule_job_for(context.application, chat_id, hour, minute)
                msg = f"⏰ Время подписки обновлено: {hour:02d}:{minute:02d} ({TZ.key})."
            else:
                pairs = db_get_pairs(chat_id)
                if pairs:
                    updated = db_update_all_pairs_time(chat_id, hour, minute)
                    for b, q in updated:
                        schedule_pair_job(context.application, chat_id, b, q, hour, minute)
                    msg = f"⏰ Время подписок на пары обновлено: {hour:02d}:{minute:02d} ({TZ.key})."
                else:
                    msg = "ℹ️ У вас пока нет подписок. Оформите подписку и затем задайте время."

            await update.message.reply_text(msg, reply_markup=MAIN_KB)
        except Exception as e:
            await update.message.reply_text(f"⚠️ {e}\nПример: 09:30", reply_markup=MAIN_KB)
        return

    # ожидание ISO-кода после "Ввести другой"
    wait_base = context.user_data.get("wait_base")
    if wait_base:
        code_in = normalize_code(text)
        kb = USD_SUB_KB if wait_base == "USD" else (EUR_SUB_KB if wait_base == "EUR" else RUB_SUB_KB)
        if not code_in or code_in == wait_base:
            await update.message.reply_text("Введите корректный ISO-код (например: JPY / UZS).", reply_markup=kb)
            return
        try:
            rates = await fetch_rates(wait_base, [code_in])
            value = rates.get(code_in)
            if value is None:
                await update.message.reply_text(f"Нет данных для пары {wait_base} - {code_in}", reply_markup=kb)
            else:
                await update.message.reply_text(f"{wait_base} - {code_in} {fmt_value(wait_base, value)}", reply_markup=kb)
        except Exception as e:
            await update.message.reply_text(f"⚠️ {e}")
        finally:
            context.user_data.pop("wait_base", None)
        return

    # справочник по стране
    if context.user_data.get("wait_country") is True:
        context.user_data.pop("wait_country", None)
        codes = await fetch_currency_by_country(text)
        if not codes:
            await update.message.reply_text("Не нашёл валюту для этой страны. Попробуйте: Japan / Япония.", reply_markup=MAIN_KB)
        else:
            await update.message.reply_text("Коды валют: " + ", ".join(codes), reply_markup=MAIN_KB)
        return

    # разовая пара
    if context.user_data.get("wait_pair") is True:
        context.user_data.pop("wait_pair", None)
        p = parse_pair(text)
        if not p:
            await update.message.reply_text("Формат: USD - EUR", reply_markup=MAIN_KB)
            return
        base, quote = p
        try:
            rates = await fetch_rates(base, [quote])
            val = rates.get(quote)
            if val is None:
                await update.message.reply_text(f"Нет данных для пары {base} - {quote}", reply_markup=MAIN_KB)
            else:
                await update.message.reply_text(f"{flag(base)}{base} → {flag(quote)}{quote} {fmt_value(base, val)}", reply_markup=MAIN_KB)
        except Exception as e:
            await update.message.reply_text(f"⚠️ {e}", reply_markup=MAIN_KB)
        return

    # подписка на пару (ввод)
    if context.user_data.get("wait_pair_sub") is True:
        context.user_data.pop("wait_pair_sub", None)
        p = parse_pair(text)
        if not p:
            await update.message.reply_text("Формат: USD - RUB", reply_markup=MAIN_KB)
            return
        base, quote = p
        with closing(sqlite3.connect(DB_PATH)) as conn:
            row = conn.execute("SELECT hour, minute FROM subscriptions WHERE chat_id=?", (update.effective_chat.id,)).fetchone()
        hour, minute = (row[0], row[1]) if row else (9, 0)
        chat_id = update.effective_chat.id
        db_upsert_pair(chat_id, base, quote, hour, minute)
        schedule_pair_job(context.application, chat_id, base, quote, hour, minute)
        await update.message.reply_text(f"✅ Вы подписаны на обновления курса {base}-{quote}. Будни в {hour:02d}:{minute:02d} ({TZ.key}).", reply_markup=MAIN_KB)
        return

    # подменю
    submenu = context.user_data.get("submenu")

    if submenu == "USD":
        clean = re.sub(r'\s+', ' ', remove_flags(text)).strip()
        if clean == BTN_BACK:
            context.user_data.pop("submenu", None)
            await update.message.reply_text("Главное меню 👇", reply_markup=MAIN_KB); return
        if clean == BTN_ENTER_OTHER:
            context.user_data["wait_base"] = "USD"
            await update.message.reply_text("Введите ISO-код валюты, например: JPY", reply_markup=USD_SUB_KB); return

        p = extract_pair_from_button(text)
        if p and p[0] == "USD":
            _, sym = p
            try:
                val = (await fetch_rates("USD", [sym])).get(sym)
                if val is None:
                    await update.message.reply_text(f"Нет данных для пары USD - {sym}", reply_markup=USD_SUB_KB)
                else:
                    await update.message.reply_text(f"{flag('USD')}USD → {flag(sym)}{sym} {fmt_value('USD', val)}", reply_markup=USD_SUB_KB)
            except Exception as e:
                await update.message.reply_text(f"⚠️ {e}", reply_markup=USD_SUB_KB)
            return

        await update.message.reply_text("Выберите пару из списка или нажмите «Ввести другой».", reply_markup=USD_SUB_KB); return

    if submenu == "EUR":
        clean = re.sub(r'\s+', ' ', remove_flags(text)).strip()
        if clean == BTN_BACK:
            context.user_data.pop("submenu", None)
            await update.message.reply_text("Главное меню 👇", reply_markup=MAIN_KB); return
        if clean == BTN_ENTER_OTHER:
            context.user_data["wait_base"] = "EUR"
            await update.message.reply_text("Введите ISO-код валюты, например: JPY", reply_markup=EUR_SUB_KB); return

        p = extract_pair_from_button(text)
        if p and p[0] == "EUR":
            _, sym = p
            try:
                val = (await fetch_rates("EUR", [sym])).get(sym)
                if val is None:
                    await update.message.reply_text(f"Нет данных для пары EUR - {sym}", reply_markup=EUR_SUB_KB)
                else:
                    await update.message.reply_text(f"{flag('EUR')}EUR → {flag(sym)}{sym} {fmt_value('EUR', val)}", reply_markup=EUR_SUB_KB)
            except Exception as e:
                await update.message.reply_text(f"⚠️ {e}", reply_markup=EUR_SUB_KB)
            return

        await update.message.reply_text("Выберите пару из списка или нажмите «Ввести другой».", reply_markup=EUR_SUB_KB); return

    if submenu == "RUB":
        clean = re.sub(r'\s+', ' ', remove_flags(text)).strip()
        if clean == BTN_BACK:
            context.user_data.pop("submenu", None)
            await update.message.reply_text("Главное меню 👇", reply_markup=MAIN_KB); return
        if clean == BTN_ENTER_OTHER:
            context.user_data["wait_base"] = "RUB"
            await update.message.reply_text("Введите ISO-код валюты, например: UZS", reply_markup=RUB_SUB_KB); return

        p = extract_pair_from_button(text)
        if p and p[0] == "RUB":
            _, sym = p
            try:
                val = (await fetch_rates("RUB", [sym])).get(sym)
                if val is None:
                    await update.message.reply_text(f"Нет данных для пары RUB - {sym}", reply_markup=RUB_SUB_KB)
                else:
                    await update.message.reply_text(f"{flag('RUB')}RUB → {flag(sym)}{sym} {fmt_value('RUB', val)}", reply_markup=RUB_SUB_KB)
            except Exception as e:
                await update.message.reply_text(f"⚠️ {e}", reply_markup=RUB_SUB_KB)
            return

        await update.message.reply_text("Выберите пару из списка или нажмите «Ввести другой».", reply_markup=RUB_SUB_KB); return

    # Главные кнопки
    if text == BTN_RATE_USD: return await cmd_usd(update, context)
    if text == BTN_RATE_EUR: return await cmd_eur(update, context)
    if text == BTN_RATE_RUB: return await cmd_rub(update, context)
    if text == BTN_REFBOOK:
        context.user_data["wait_country"] = True
        return await update.message.reply_text("Введите страну, чтобы узнать код валюты (напр.: Япония / Japan)", reply_markup=MAIN_KB)
    if text == BTN_ENTER_PAIR:
        context.user_data["wait_pair"] = True
        return await update.message.reply_text("Введите пару в формате: USD - EUR", reply_markup=MAIN_KB)
    if text == BTN_SUB_PAIR:
        context.user_data["wait_pair_sub"] = True
        return await update.message.reply_text("Какую пару хотите получать каждый день? Напишите в формате: USD - RUB", reply_markup=MAIN_KB)
    if text == BTN_SUB:
        hour, minute = (9, 0)
        chat_id = update.effective_chat.id
        db_upsert(chat_id, hour, minute)
        schedule_job_for(context.application, chat_id, hour, minute)
        return await update.message.reply_text(
            "✅ Вы подписаны на автообновление по умолчанию:\n"
            + targets_str_multiline()
            + f"\nБудни в {hour:02d}:{minute:02d} (Europe/Moscow)",
            parse_mode=ParseMode.HTML
        )
    if text == BTN_TIME:   return await cmd_time(update, context)
    if text == BTN_STATUS: return await cmd_status(update, context)
    if text == BTN_UNSUB:  return await cmd_unsubscribe(update, context)
    if text == BTN_RESET:  return await reset_dialog(update, context)
    if text == BTN_SUPPORT: return await cmd_support(update, context)

    return await update.message.reply_text("Не понял. Нажмите кнопку на клавиатуре или /menu.", reply_markup=MAIN_KB)

async def on_support_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Приём медиа в режиме поддержки (фото/видео/док)."""
    if update.effective_chat.id == SUPPORT_CHAT_ID:
        return
    if not context.user_data.get("wait_support"):
        return

    now = time.time()
    last = context.user_data.get("last_support_at", 0)
    if now - last < SUPPORT_COOLDOWN:
        await update.message.reply_text("Немного подождите и попробуйте снова.", reply_markup=MAIN_KB)
        return

    user = update.effective_user
    chat = update.effective_chat

    if not (update.message.photo or update.message.video or update.message.document):
        await update.message.reply_text("Можно отправить только фото, видео или документ.", reply_markup=MAIN_KB)
        context.user_data.pop("wait_support", None)
        return

    caption = update.message.caption or ""
    card_text = (
        "📩 <b>Сообщение в поддержку</b>\n"
        f"from: {user.full_name} (id <code>{user.id}</code>)\n"
        f"username: @{user.username if user.username else '—'}\n"
        f"chat_id: <code>{chat.id}</code>\n\n"
        f"{caption or '🖼 Медиа без текста'}"
    )

    try:
        support_msg = await context.bot.send_message(
            SUPPORT_CHAT_ID, card_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True
        )
        context.application.bot_data.setdefault("support_map", {})[support_msg.message_id] = chat.id
        await context.bot.copy_message(
            chat_id=SUPPORT_CHAT_ID,
            from_chat_id=chat.id,
            message_id=update.message.message_id,
            reply_to_message_id=support_msg.message_id,
        )
        await update.message.reply_text("Спасибо! Сообщение отправлено в поддержку. Мы ответим здесь.", reply_markup=MAIN_KB)
        context.user_data["last_support_at"] = now
    except Exception as e:
        await update.message.reply_text(f"⚠️ Не удалось отправить: {e}", reply_markup=MAIN_KB)
    finally:
        context.user_data.pop("wait_support", None)

# ===== Scheduler & startup =====
scheduler: Optional[AsyncIOScheduler] = None

def ensure_scheduler() -> None:
    global scheduler
    if scheduler is None:
        scheduler = AsyncIOScheduler(timezone=TZ)
        scheduler.start()
    elif not scheduler.running:
        scheduler.start()

async def send_digest_to_chat(app: Application, chat_id: int) -> None:
    try:
        now_local = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
        blocks = []
        for base in ["USD", "EUR", "RUB"]:
            if base in TARGETS:
                rates = await fetch_rates(base, TARGETS[base])
                blocks.append(format_block(base, rates))
        text = f"📈 Сводка • {now_local} {TZ.key}\n\n" + "\n\n".join(blocks)
        await app.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        print(f"[digest] {chat_id=} error={e}")

async def send_pair_to_chat(app: Application, chat_id: int, base: str, quote: str) -> None:
    try:
        rates = await fetch_rates(base, [quote])
        val = rates.get(quote)
        text = (f"Нет данных для пары {flag(base)}{base} - {flag(quote)}{quote}"
                if val is None else f"{flag(base)}{base} → {flag(quote)}{quote} {fmt_value(base, val)}")
        await app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        print(f"[pair] {chat_id=} {base=}-{quote=} error={e}")

def schedule_job_for(app: Application, chat_id: int, hour: int, minute: int) -> None:
    ensure_scheduler()
    job_id = f"digest_{chat_id}"
    if scheduler.get_job(job_id): scheduler.remove_job(job_id)
    trigger = CronTrigger(hour=hour, minute=minute, timezone=TZ, day_of_week="mon-fri")
    scheduler.add_job(send_digest_to_chat, trigger=trigger, id=job_id, args=[app, chat_id], replace_existing=True, misfire_grace_time=1800)

def schedule_pair_job(app: Application, chat_id: int, base: str, quote: str, hour: int, minute: int) -> None:
    ensure_scheduler()
    base = base.upper(); quote = quote.upper()
    job_id = f"pair_{chat_id}_{base}_{quote}"
    if scheduler.get_job(job_id): scheduler.remove_job(job_id)
    trigger = CronTrigger(hour=hour, minute=minute, timezone=TZ, day_of_week="mon-fri")
    scheduler.add_job(send_pair_to_chat, trigger=trigger, id=job_id, args=[app, chat_id, base, quote], replace_existing=True, misfire_grace_time=1800)

async def on_startup(app: Application) -> None:
    db_init()
    ensure_scheduler()
    for chat_id, hour, minute in db_get_all():
        schedule_job_for(app, chat_id, hour, minute)
    for chat_id, base, quote, hour, minute in db_get_all_pairs():
        schedule_pair_job(app, chat_id, base, quote, hour, minute)
    try:
        from telegram import (BotCommandScopeDefault, BotCommandScopeAllPrivateChats,
                              BotCommandScopeAllGroupChats, BotCommandScopeAllChatAdministrators)
        await app.bot.delete_my_commands(scope=BotCommandScopeDefault())
        await app.bot.delete_my_commands(scope=BotCommandScopeAllPrivateChats())
        await app.bot.delete_my_commands(scope=BotCommandScopeAllGroupChats())
        await app.bot.delete_my_commands(scope=BotCommandScopeAllChatAdministrators())
    except Exception:
        pass

async def on_shutdown(app: Application) -> None:
    if scheduler:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass

# ===== Error handler =====
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        chat_id = None
        if isinstance(update, Update) and update.effective_chat:
            chat_id = update.effective_chat.id
        err = context.error
        print(f"[ERROR] {err!r}")
        if chat_id:
            await context.bot.send_message(chat_id, "⚠️ Произошла ошибка, уже чиним. Попробуйте ещё раз.")
    except Exception as e:
        print(f"[ERROR-HANDLER] {e!r}")

# ===== Main (WEBHOOK) =====
async def main() -> None:
    token = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN or BOT_TOKEN is missing. Set it in env")

    # Render подставляет публичный URL в переменную RENDER_EXTERNAL_URL
    public_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PUBLIC_URL")
    if not public_url:
        raise RuntimeError("PUBLIC URL is missing. Set RENDER_EXTERNAL_URL (Render) или PUBLIC_URL")

    port = int(os.getenv("PORT", "10000"))
    webhook_path = token  # секретный путь
    webhook_url = f"{public_url.rstrip('/')}/{webhook_path}"

    app = Application.builder().token(token).build()

    # handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("menu", cmd_menu))
    app.add_handler(CommandHandler("hide", cmd_hide))
    app.add_handler(CommandHandler("usd", cmd_usd))
    app.add_handler(CommandHandler("eur", cmd_eur))
    app.add_handler(CommandHandler("rub", cmd_rub))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("subpair", cmd_subpair))
    app.add_handler(CommandHandler("unsubpair", cmd_unsubpair))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Chat(SUPPORT_CHAT_ID), on_text_buttons))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(CommandHandler("time", cmd_time))
    app.add_handler(CommandHandler("now", cmd_now))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("import", cmd_import))
    app.add_handler(CommandHandler("support", cmd_support))
    app.add_handler(CommandHandler("whoami", cmd_whoami))
    app.add_handler(CommandHandler("ban", cmd_ban))
    app.add_handler(CommandHandler("unban", cmd_unban))
    if SUPPORT_CHAT_ID:
        app.add_handler(MessageHandler(filters.Chat(SUPPORT_CHAT_ID) & filters.REPLY, on_support_reply), group=-1)
    app.add_handler(MessageHandler((~filters.Chat(SUPPORT_CHAT_ID)) & filters.ATTACHMENT, on_support_media))
    app.add_error_handler(on_error)

    # старт/вебхук/сервер
    await on_startup(app)
    await app.initialize()
    await app.start()
    await app.bot.set_webhook(url=webhook_url, drop_pending_updates=True)
    db_init_banned()
    print(f"Bot started (webhook): {webhook_url}")

    try:
        await app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=webhook_path,
            webhook_url=webhook_url,
        )
    finally:
        await on_shutdown(app)
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
