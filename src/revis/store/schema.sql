-- Revis SQLite Schema

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE,  -- User-provided session name
    branch TEXT NOT NULL,
    base_sha TEXT NOT NULL,
    baseline_run_id TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    termination_reason TEXT,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    budget_type TEXT NOT NULL,
    budget_value INTEGER NOT NULL,
    budget_used INTEGER NOT NULL DEFAULT 0,
    pr_url TEXT,
    llm_cost_usd REAL NOT NULL DEFAULT 0,
    retry_budget INTEGER NOT NULL DEFAULT 3,
    iteration_count INTEGER NOT NULL DEFAULT 0,
    pid INTEGER,  -- Local process ID for orphan detection
    exported_at TIMESTAMP  -- When exported to remote
);

-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    iteration_number INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    git_sha TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    exit_code INTEGER,
    UNIQUE(session_id, iteration_number)
);

-- Params (flattened config for querying)
CREATE TABLE IF NOT EXISTS params (
    run_id TEXT NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY(run_id, key)
);

-- Metrics (time-series)
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Artifacts
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Decisions
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    action_type TEXT NOT NULL,
    rationale TEXT NOT NULL,
    commit_sha TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Traces (agent tool calls and results)
CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    timestamp TEXT DEFAULT (datetime('now')),
    event_type TEXT NOT NULL,  -- tool_call, tool_result
    data_json TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_sessions_name ON sessions(name);
CREATE INDEX IF NOT EXISTS idx_traces_run ON traces(run_id);
