from datetime import datetime, timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from app.config import DATABASE_URL

_ENGINE_KW: dict = {"echo": False}
if "sqlite" in (DATABASE_URL or ""):
    _ENGINE_KW["connect_args"] = {"timeout": 30}
engine = create_async_engine(DATABASE_URL, **_ENGINE_KW)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# Mevcut DB'ye sonradan eklenen kolonlar (basit ileri-migrasyon).
_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
    "signals": [
        ("entry_filled_time", "ALTER TABLE signals ADD COLUMN entry_filled_time DATETIME"),
        ("smt_pair", "ALTER TABLE signals ADD COLUMN smt_pair VARCHAR"),
        ("tg_message_id", "ALTER TABLE signals ADD COLUMN tg_message_id INTEGER"),
        ("pd_array", "ALTER TABLE signals ADD COLUMN pd_array VARCHAR"),
        ("c2_closed", "ALTER TABLE signals ADD COLUMN c2_closed BOOLEAN DEFAULT 0"),
        ("weekly_bias", "ALTER TABLE signals ADD COLUMN weekly_bias VARCHAR"),
        ("initial_stop_loss", "ALTER TABLE signals ADD COLUMN initial_stop_loss FLOAT"),
        ("partial_hit", "ALTER TABLE signals ADD COLUMN partial_hit BOOLEAN DEFAULT 0"),
        ("trail_active", "ALTER TABLE signals ADD COLUMN trail_active BOOLEAN DEFAULT 0"),
        ("mfe_price", "ALTER TABLE signals ADD COLUMN mfe_price FLOAT"),
        ("planned_rr", "ALTER TABLE signals ADD COLUMN planned_rr FLOAT"),
        ("mss_ref_time", "ALTER TABLE signals ADD COLUMN mss_ref_time DATETIME"),
        ("entry_model", "ALTER TABLE signals ADD COLUMN entry_model VARCHAR"),
        ("bpr_low", "ALTER TABLE signals ADD COLUMN bpr_low FLOAT"),
        ("bpr_high", "ALTER TABLE signals ADD COLUMN bpr_high FLOAT"),
        ("ifvg_low", "ALTER TABLE signals ADD COLUMN ifvg_low FLOAT"),
        ("ifvg_high", "ALTER TABLE signals ADD COLUMN ifvg_high FLOAT"),
        ("protection_armed_time", "ALTER TABLE signals ADD COLUMN protection_armed_time DATETIME"),
        ("exit_reason", "ALTER TABLE signals ADD COLUMN exit_reason VARCHAR"),
        ("partial_size", "ALTER TABLE signals ADD COLUMN partial_size FLOAT"),
        ("partial_price", "ALTER TABLE signals ADD COLUMN partial_price FLOAT"),
        ("partial_rr", "ALTER TABLE signals ADD COLUMN partial_rr FLOAT"),
        ("partial_time", "ALTER TABLE signals ADD COLUMN partial_time DATETIME"),
        ("tg_potential_id", "ALTER TABLE signals ADD COLUMN tg_potential_id INTEGER"),
        ("tg_potential_state", "ALTER TABLE signals ADD COLUMN tg_potential_state VARCHAR"),
        ("closed_at", "ALTER TABLE signals ADD COLUMN closed_at DATETIME"),
    ],
    "setup_journal": [
        ("shadow", "ALTER TABLE setup_journal ADD COLUMN shadow VARCHAR"),
        ("score_parts", "ALTER TABLE setup_journal ADD COLUMN score_parts VARCHAR"),
        ("features", "ALTER TABLE setup_journal ADD COLUMN features VARCHAR"),
        ("parts_at_levels", "ALTER TABLE setup_journal ADD COLUMN parts_at_levels VARCHAR"),
        ("features_at_levels", "ALTER TABLE setup_journal ADD COLUMN features_at_levels VARCHAR"),
        ("entries", "ALTER TABLE setup_journal ADD COLUMN entries VARCHAR"),
        ("retrace", "ALTER TABLE setup_journal ADD COLUMN retrace VARCHAR"),
        ("prefill", "ALTER TABLE setup_journal ADD COLUMN prefill VARCHAR"),
    ],
}


async def _backfill_closed_at(conn) -> None:
    """closed_at kolonundan once kapanan kayitlar: dolum (yoksa CISD) + duration_hours.

    Iki kapanis yolu da sureyi bu referanstan kapanis anina kadar yazdigi icin kapanis
    ani ±3 dk (0.1 saat yuvarlama) hassasiyetle geri elde edilir. Bir kez doldurulan
    kayda tekrar dokunulmaz.
    """
    rows = (await conn.execute(text(
        "SELECT id, entry_filled_time, cisd_time, duration_hours FROM signals "
        "WHERE closed_at IS NULL AND result IS NOT NULL AND duration_hours IS NOT NULL"
    ))).fetchall()
    is_sqlite = "sqlite" in (DATABASE_URL or "")
    for sid, filled, cisd, dur in rows:
        ref = filled or cisd
        if ref is None:
            continue
        if isinstance(ref, str):
            ref = datetime.fromisoformat(ref)
        closed = ref.replace(tzinfo=None) + timedelta(hours=float(dur))
        value = closed.strftime("%Y-%m-%d %H:%M:%S.%f") if is_sqlite else closed
        await conn.execute(text("UPDATE signals SET closed_at = :c WHERE id = :i"), {"c": value, "i": sid})


async def _relax_paper_entry_price(conn) -> None:
    """paper_trades.entry_price'taki NOT NULL'u kaldir (25.09: giris fiyati zorunlu degil).

    SQLite bir kolonun kisitini ALTER ile degistiremez; tablo kendi CREATE cumlesinden
    yeniden kurulur, veri ve indeksler aynen tasinir. Kisit zaten yoksa hicbir sey yapmaz.
    """
    if "sqlite" not in (DATABASE_URL or ""):
        return
    info = (await conn.exec_driver_sql("PRAGMA table_info(paper_trades)")).fetchall()
    if not any(row[1] == "entry_price" and row[3] for row in info):
        return
    create_sql = (await conn.exec_driver_sql(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='paper_trades'")).scalar()
    new_sql = create_sql.replace("entry_price FLOAT NOT NULL", "entry_price FLOAT", 1)
    if new_sql == create_sql:
        return                                         # beklenmeyen sema: dokunma
    new_sql = new_sql.replace("CREATE TABLE paper_trades", "CREATE TABLE paper_trades__new", 1)
    indexes = [r[0] for r in (await conn.exec_driver_sql(
        "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='paper_trades' "
        "AND sql IS NOT NULL")).fetchall()]
    await conn.exec_driver_sql(new_sql)
    await conn.exec_driver_sql("INSERT INTO paper_trades__new SELECT * FROM paper_trades")
    await conn.exec_driver_sql("DROP TABLE paper_trades")
    await conn.exec_driver_sql("ALTER TABLE paper_trades__new RENAME TO paper_trades")
    for ddl in indexes:
        await conn.exec_driver_sql(ddl)


async def _apply_column_migrations(conn) -> None:
    for table, columns in _MIGRATIONS.items():
        res = await conn.exec_driver_sql(f"PRAGMA table_info({table})")
        existing = {row[1] for row in res.fetchall()}
        for col_name, ddl in columns:
            if col_name not in existing:
                await conn.exec_driver_sql(ddl)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _apply_column_migrations(conn)
        if "sqlite" in (DATABASE_URL or ""):
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            await conn.exec_driver_sql("PRAGMA busy_timeout=5000")
            await conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
    # Veri guncellemesi AYRI transaction'da: UPDATE acik bir transaction baslatir ve ayni
    # blokta ardindan gelen `PRAGMA synchronous` "Safety level may not be changed inside a
    # transaction" ile uygulama acilisini dusuruyordu (14.09 16:11).
    async with engine.begin() as conn:
        await _backfill_closed_at(conn)
        await _relax_paper_entry_price(conn)


async def get_db():
    async with async_session() as session:
        yield session
