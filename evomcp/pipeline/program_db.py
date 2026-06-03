"""SQLite program database for local evolution runs.

This is a small, dependency-free subset of the useful OpenEvolve/GigaEvo/
ThetaEvolve ideas: keep a queryable program archive with lineage, evaluations,
secondary metrics, archive membership, events, and short insight notes. The
JSONL traces remain the replayable ground truth; SQLite is the analysis index.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evomcp.pipeline.candidate import Budget, Candidate, EvalResult


SCHEMA_VERSION = 1


class ProgramDB:
    """Append-friendly SQLite index for one or more evolution runs."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        mode: str,
        config_path: Path,
        output_dir: Path,
        config: dict[str, Any],
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.mode = mode
        self.config_path = config_path
        self.output_dir = output_dir
        self.config = config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self.record_run(status="running")

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        config_path: str | Path,
        output_dir: Path,
        mode: str,
    ) -> "ProgramDB | None":
        db_cfg = cfg.get("program_db", cfg.get("database", {}))
        if db_cfg is False or db_cfg.get("enabled", True) is False:
            return None
        env_path = os.environ.get("EVOMCP_DB")
        raw_path = db_cfg.get("path") or env_path
        if raw_path is None:
            path = output_dir / "programs.sqlite"
        else:
            path = Path(raw_path)
        if raw_path is not None and not path.is_absolute():
            path = output_dir / path
        run_id = str(db_cfg.get("run_id") or cfg.get("run_id") or output_dir.name)
        return cls(
            path,
            run_id=run_id,
            mode=mode,
            config_path=Path(config_path).resolve(),
            output_dir=output_dir.resolve(),
            config=cfg,
        )

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                config_path TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                island TEXT NOT NULL DEFAULT 'main',
                candidate_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                prog_hash TEXT NOT NULL,
                text_genome_json TEXT NOT NULL,
                prog_genome_json TEXT NOT NULL,
                parent_ids_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(run_id, candidate_id)
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                candidate_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                stage INTEGER NOT NULL,
                is_aggregate INTEGER NOT NULL DEFAULT 0,
                success INTEGER NOT NULL,
                primary_score REAL NOT NULL,
                error_type TEXT,
                error_message TEXT,
                evaluator_version TEXT,
                dataset_version TEXT,
                trace_bundle_dir TEXT,
                cost_usd REAL NOT NULL DEFAULT 0,
                wall_s REAL NOT NULL DEFAULT 0,
                calls INTEGER NOT NULL DEFAULT 0,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(
                    run_id, candidate_id, seed, stage, is_aggregate,
                    evaluator_version, dataset_version
                )
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
                key TEXT NOT NULL,
                value_real REAL,
                value_text TEXT,
                UNIQUE(evaluation_id, key)
            );

            CREATE TABLE IF NOT EXISTS archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                candidate_id TEXT NOT NULL,
                archive_name TEXT NOT NULL DEFAULT 'pareto',
                rank INTEGER NOT NULL DEFAULT 0,
                primary_score REAL NOT NULL,
                objectives_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(run_id, archive_name, candidate_id)
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                generation INTEGER,
                type TEXT NOT NULL,
                candidate_id TEXT,
                event_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                generation INTEGER,
                candidate_id TEXT,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                score REAL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_programs_run_generation
                ON programs(run_id, generation, island);
            CREATE INDEX IF NOT EXISTS idx_evaluations_run_score
                ON evaluations(run_id, is_aggregate, success, primary_score DESC);
            CREATE INDEX IF NOT EXISTS idx_metrics_key_value
                ON metrics(key, value_real);
            CREATE INDEX IF NOT EXISTS idx_archive_run_score
                ON archive(run_id, archive_name, primary_score DESC);
            CREATE INDEX IF NOT EXISTS idx_events_run_type
                ON events(run_id, type);
            """
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.commit()

    def record_run(self, *, status: str) -> None:
        now = _utc_now()
        existing = self.conn.execute(
            "SELECT started_at FROM runs WHERE run_id = ?", (self.run_id,)
        ).fetchone()
        started_at = existing["started_at"] if existing else now
        self.conn.execute(
            """
            INSERT INTO runs(
                run_id, mode, config_path, output_dir, status,
                started_at, updated_at, config_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                mode=excluded.mode,
                config_path=excluded.config_path,
                output_dir=excluded.output_dir,
                status=excluded.status,
                updated_at=excluded.updated_at,
                config_json=excluded.config_json
            """,
            (
                self.run_id,
                self.mode,
                str(self.config_path),
                str(self.output_dir),
                status,
                started_at,
                now,
                _json(self.config),
            ),
        )
        self.conn.commit()

    def record_program(self, candidate: Candidate, *, generation: int, island: str = "main") -> None:
        self.conn.execute(
            """
            INSERT INTO programs(
                run_id, generation, island, candidate_id, text_hash, prog_hash,
                text_genome_json, prog_genome_json, parent_ids_json,
                metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, candidate_id) DO UPDATE SET
                generation=excluded.generation,
                island=excluded.island,
                text_hash=excluded.text_hash,
                prog_hash=excluded.prog_hash,
                text_genome_json=excluded.text_genome_json,
                prog_genome_json=excluded.prog_genome_json,
                parent_ids_json=excluded.parent_ids_json,
                metadata_json=excluded.metadata_json
            """,
            (
                self.run_id,
                generation,
                island,
                candidate.candidate_id,
                candidate.text_hash(),
                candidate.prog_hash(),
                _json(dict(candidate.text_genome)),
                _json(dict(candidate.prog_genome)),
                _json(list(candidate.parent_ids)),
                _json(dict(candidate.metadata)),
                _utc_now(),
            ),
        )
        self.conn.commit()

    def record_evaluation(
        self,
        candidate: Candidate,
        result: EvalResult,
        *,
        generation: int,
        seed: int,
        budget: Budget,
        is_aggregate: bool = False,
    ) -> int:
        now = _utc_now()
        key = (
            self.run_id,
            candidate.candidate_id,
            seed,
            budget.stage,
            int(is_aggregate),
            result.evaluator_version,
            result.dataset_version,
        )
        row = self.conn.execute(
            """
            SELECT id FROM evaluations
            WHERE run_id=? AND candidate_id=? AND seed=? AND stage=?
              AND is_aggregate=? AND evaluator_version=? AND dataset_version=?
            """,
            key,
        ).fetchone()
        if row:
            evaluation_id = int(row["id"])
            self.conn.execute(
                """
                UPDATE evaluations SET
                    generation=?, success=?, primary_score=?, error_type=?,
                    error_message=?, trace_bundle_dir=?, cost_usd=?, wall_s=?,
                    calls=?, input_tokens=?, output_tokens=?, result_json=?,
                    updated_at=?
                WHERE id=?
                """,
                (
                    generation,
                    int(result.success),
                    float(result.primary_score),
                    result.error_type.value if result.error_type else None,
                    result.error_message,
                    str(result.trace_bundle_dir) if result.trace_bundle_dir else None,
                    float(result.cost.usd),
                    float(result.cost.wall_s),
                    int(result.cost.calls),
                    int(result.cost.input_tokens),
                    int(result.cost.output_tokens),
                    _json(result.to_dict()),
                    now,
                    evaluation_id,
                ),
            )
            self.conn.execute("DELETE FROM metrics WHERE evaluation_id=?", (evaluation_id,))
        else:
            cur = self.conn.execute(
                """
                INSERT INTO evaluations(
                    run_id, generation, candidate_id, seed, stage, is_aggregate,
                    success, primary_score, error_type, error_message,
                    evaluator_version, dataset_version, trace_bundle_dir,
                    cost_usd, wall_s, calls, input_tokens, output_tokens,
                    result_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    generation,
                    candidate.candidate_id,
                    seed,
                    budget.stage,
                    int(is_aggregate),
                    int(result.success),
                    float(result.primary_score),
                    result.error_type.value if result.error_type else None,
                    result.error_message,
                    result.evaluator_version,
                    result.dataset_version,
                    str(result.trace_bundle_dir) if result.trace_bundle_dir else None,
                    float(result.cost.usd),
                    float(result.cost.wall_s),
                    int(result.cost.calls),
                    int(result.cost.input_tokens),
                    int(result.cost.output_tokens),
                    _json(result.to_dict()),
                    now,
                    now,
                ),
            )
            evaluation_id = int(cur.lastrowid)
        self._record_metrics(evaluation_id, result.secondary_scores)
        self.conn.commit()
        return evaluation_id

    def record_archive(
        self,
        entries: list[tuple[Candidate, EvalResult]],
        *,
        generation: int,
        archive_name: str = "pareto",
    ) -> None:
        now = _utc_now()
        for rank, (candidate, result) in enumerate(
            sorted(entries, key=lambda item: item[1].primary_score, reverse=True)
        ):
            self.conn.execute(
                """
                INSERT INTO archive(
                    run_id, generation, candidate_id, archive_name, rank,
                    primary_score, objectives_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, archive_name, candidate_id) DO UPDATE SET
                    generation=excluded.generation,
                    rank=excluded.rank,
                    primary_score=excluded.primary_score,
                    objectives_json=excluded.objectives_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self.run_id,
                    generation,
                    candidate.candidate_id,
                    archive_name,
                    rank,
                    float(result.primary_score),
                    _json(result.secondary_scores),
                    now,
                ),
            )
        self.conn.commit()

    def record_event(self, event: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO events(run_id, generation, type, candidate_id, event_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                _maybe_int(event.get("generation")),
                str(event.get("type", "event")),
                event.get("candidate_id"),
                _json(event),
                _utc_now(),
            ),
        )
        self.conn.commit()

    def record_insight(
        self,
        *,
        text: str,
        kind: str = "note",
        generation: int | None = None,
        candidate_id: str | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO insights(
                run_id, generation, candidate_id, kind, text, score, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                generation,
                candidate_id,
                kind,
                text,
                score,
                _json(metadata or {}),
                _utc_now(),
            ),
        )
        self.conn.commit()

    def best_programs(self, *, limit: int = 16, archive_name: str = "pareto") -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT p.candidate_id, a.primary_score, p.prog_genome_json, p.text_genome_json
            FROM archive a
            JOIN programs p ON p.run_id = a.run_id AND p.candidate_id = a.candidate_id
            WHERE a.run_id = ? AND a.archive_name = ?
            ORDER BY a.primary_score DESC
            LIMIT ?
            """,
            (self.run_id, archive_name, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def _record_metrics(self, evaluation_id: int, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, bool):
                value_real = float(int(value))
                value_text = None
            elif isinstance(value, (int, float)):
                value_real = float(value)
                value_text = None
            else:
                value_real = None
                value_text = _json(value)
            self.conn.execute(
                """
                INSERT OR REPLACE INTO metrics(evaluation_id, key, value_real, value_text)
                VALUES (?, ?, ?, ?)
                """,
                (evaluation_id, str(key), value_real, value_text),
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, default=str)


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
