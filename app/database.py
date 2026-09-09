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
        ("ifvg_low", "ALTER TABLE signals ADD COLUMN ifvg_low FLOAT"),
        ("ifvg_high", "ALTER TABLE signals ADD COLUMN ifvg_high FLOAT"),
        ("protection_armed_time", "ALTER TABLE signals ADD COLUMN protection_armed_time DATETIME"),
    ],
}


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


async def get_db():
    async with async_session() as session:
        yield session
