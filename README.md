# World Cup Predictor

**English** | [中文](#世界杯预测系统)

A dynamic, extensible FIFA World Cup prediction system. Team strength is treated
as a latent variable: historical results feed a World Football Elo rating, the
Elo difference maps to expected goals, a Poisson model produces the full score
distribution for each match, and a Monte Carlo simulator turns match-level
distributions into group-qualification, knockout-advancement and championship
probabilities for the 48-team 2026 tournament. Once real results come in, the
system re-syncs, refits and re-predicts the remainder of the tournament.

Current features:

- Historical match CSV loading, field validation and team-name normalization
- World Football Elo with per-competition K factors, updated match by match
- Poisson goal model fitted by MLE on the Elo difference (with time decay)
- Per-match score matrix, win/draw/loss probabilities and most likely score
- Time-split backtesting with RPS, Log Loss, Brier Score and calibration tables
- JSON persistence of model parameters and Elo snapshots
- Training, ranking, prediction and simulation CLI
- 2026 World Cup simulator: 48 teams, 12 groups, FIFA Article 13 tie-breakers,
  the official Annexe C third-place bracket mapping (all 495 combinations),
  and Monte Carlo aggregation of per-round advancement probabilities
- Config-driven tournament formats: the 48-team 2026 edition and the 32-team
  2018/2022 edition (with the pre-2026 tie-breaker order) share one simulator
- Full-pipeline historical backtests of the 2018 and 2022 World Cups with
  per-stage skill scores against a structural baseline — see
  [docs/tournament_backtest.md](docs/tournament_backtest.md)
- Pre-simulation catch-up: download the latest results, fill fixture scores,
  refit the model, and pin already-played knockout matches to their real
  winners (penalty shootouts resolved from the companion shootouts data)
- Parquet export for processed data
- PostgreSQL schema / SQLAlchemy storage layer
- FastAPI service and Streamlit dashboard
- Celery + Redis dynamic-update entry points
- Full Dixon-Coles attack/defense MLE with the ρ low-score correction
- Empirical-Bayes hierarchical Poisson model with prediction intervals
- Player-level squad adjustments and a team+player hybrid predictor
- API-Football sync for live squads, injuries, lineups and player ratings
- Docker / docker-compose deployment skeleton

## Environment

The project uses a dedicated Conda environment named `WC`:

```bash
conda activate WC
python -m pip install -e ".[dev,bayes]"
```

Or rebuild it from the environment file:

```bash
conda env create -f environment.yml
```

Optional extras: `dev` (pytest), `bayes` (PyMC + ArviZ for the hierarchical
model), `gpu` (PyTorch; see the CPU/GPU section below for the CUDA wheel).
Everything except the Bayesian model and GPU acceleration works with just the
base dependencies.

## CPU / GPU compute

The default is `--device auto`: the GPU is used when a CUDA-enabled PyTorch
build and an NVIDIA driver are available, otherwise the CPU is used. You can
also pass `--device cpu` or `--device cuda` explicitly; an explicit CUDA
request fails loudly instead of silently falling back.

GPU support is an optional dependency. First install the
[PyTorch CUDA wheel](https://pytorch.org/get-started/locally/) matching your
local CUDA/driver setup inside the `WC` environment, then install the
project's GPU extra:

```bash
python -m pip install -e ".[gpu]"
```

Elo updates, Pandas processing and SciPy MLE training stay on the CPU; the GPU
accelerates the Elo-Poisson score-matrix computation and can be selected for
single-match prediction and the tournament simulator. Both backends use
float64, so results differ only at floating-point rounding level.

## CSV schema

Input files must contain the following fields:

| Field | Type | Description |
|---|---|---|
| `date` | `YYYY-MM-DD` | Match date |
| `home_team` | string | Home team |
| `away_team` | string | Away team |
| `home_goals` | integer | Home goals |
| `away_goals` | integer | Away goals |
| `competition_type` | string | Competition name |
| `neutral_venue` | boolean | Neutral venue flag |

`home_score` / `away_score` / `tournament` / `neutral` are accepted as
compatible column aliases.

## Usage

The example data only exercises the code paths and does not represent real
matches:

```bash
worldcup-predictor train \
  --matches data/examples/synthetic_matches.csv \
  --output models/elo_poisson_v1.json

worldcup-predictor rankings \
  --model models/elo_poisson_v1.json

worldcup-predictor predict \
  --model models/elo_poisson_v1.json \
  --home Atlas --away Comet --neutral \
  --device auto

worldcup-predictor backtest \
  --matches data/raw/international_results.csv \
  --cutoff 2024-01-01

# simulate runs a catch-up first by default: download the latest results,
# fill group fixture scores, refit the model, and pin knockout matches that
# were already played to their real winners; only remaining matches are
# sampled. Use --offline to skip the download-and-refit, and
# --no-condition-knockouts to disable pinning real knockout results.
worldcup-predictor simulate \
  --simulations 10000 \
  --device cuda \
  --output data/processed/simulation_2026.csv

# Or refresh data and the model without simulating:
worldcup-predictor catch-up

worldcup-predictor benchmark-prediction \
  --model models/elo_poisson_current.json \
  --home Argentina --away France --iterations 10000

worldcup-predictor export-parquet \
  --matches data/raw/international_results.csv \
  --output data/processed/matches.parquet

worldcup-predictor train-dixon-coles \
  --matches data/raw/international_results.csv \
  --since 2018-01-01 \
  --max-iterations 200 \
  --output models/dixon_coles_current.json

# Fair dynamic comparison: per-match Elo updates vs rolling Dixon-Coles refits
worldcup-predictor compare-dixon-coles \
  --matches data/raw/international_results.csv \
  --cutoff 2024-01-01

# Full-pipeline backtest: train before a past World Cup, simulate the whole
# tournament and score stage probabilities against what really happened
worldcup-predictor backtest-tournament \
  --matches data/raw/international_results.csv \
  --groups data/worldcup/groups_2018.csv \
  --fixtures data/worldcup/fixtures_2018.csv \
  --actual data/worldcup/actual_2018.csv \
  --format wc32 --train-before 2018-06-14 --label wc2018

worldcup-predictor train-bayesian \
  --matches data/raw/international_results.csv \
  --since 2018-01-01 \
  --max-iterations 200 \
  --posterior-draws 100 \
  --output models/bayesian_current.json

worldcup-predictor predict-player-adjusted \
  --model models/elo_poisson_v1.json \
  --players data/examples/synthetic_players.csv \
  --home Atlas --away Comet --neutral
```

### Live player data

Step 12 uses API-Football v3 as the real data provider. With an API key
configured, `sync-api-football` syncs the World Cup (`league=1`,
`season=2026`) teams, current squads and injuries; passing a started
`--fixture-id` also syncs confirmed lineups and per-match player ratings.

```bash
export API_FOOTBALL_KEY='your-api-football-key'

worldcup-predictor sync-api-football \
  --database-url postgresql+psycopg://worldcup:worldcup@localhost:5432/worldcup

worldcup-predictor predict-player-adjusted \
  --model models/elo_poisson_current.json \
  --database-url postgresql+psycopg://worldcup:worldcup@localhost:5432/worldcup \
  --home Argentina --away France --neutral
```

In the Docker environment, setting `API_FOOTBALL_KEY` and
`API_FOOTBALL_SYNC_ENABLED=true` makes Celery Beat run the sync every 6
hours. Without a key the task skips safely and team-level predictions are
unaffected.

Download the CC0 historical national-team results used by the MVP:

```bash
worldcup-predictor fetch-data \
  --output data/raw/international_results.csv
```

The download also writes a `.metadata.json` recording the source URL,
SHA-256, byte count and download time.

Run the tests:

```bash
pytest
```

Start the API and the dashboard:

```bash
uvicorn worldcup_predictor.api.main:app --reload
streamlit run app/dashboard.py
```

Docker:

```bash
docker compose up --build
```

## Roadmap status

| Step | Scope | Status |
|---|---|---|
| 1 | Elo on CSV data | Done |
| 2 | Elo + Poisson `predict_match()` | Done |
| 3 | Backtest + RPS + calibration | Done |
| 4 | World Cup simulator | Done |
| 5 | Parquet processed data | Done |
| 6 | PostgreSQL schema / storage | Storage layer, schema and tests done; the serving path (API / dynamic updates) persists to JSON/CSV files and is not wired to PostgreSQL yet — only the API-Football player sync writes to the database |
| 7 | FastAPI | Done |
| 8 | Streamlit dashboard | Done |
| 9 | Celery + Redis dynamic updates | Workflow, fixture updates, remaining-fixture re-prediction and Celery task done |
| 10 | Dixon-Coles upgrade | Identifiability constraints, analytic gradients, ρ correction, save/load, CLI and convergence on real 2018+ data done; a fair rolling-refit comparison beats Elo-Poisson on RPS/log-loss/Brier and fixes the draw under-estimation — see [docs/model_comparison.md](docs/model_comparison.md). `simulate --model models/dixon_coles_current.json` runs the tournament on it |
| 11 | PyMC Bayesian upgrade | Empirical-Bayes and PyMC MCMC hierarchical models with posterior-sample prediction intervals done |
| 12 | Player-level extension | CSV and API-Football sync, PostgreSQL persistence, injury/lineup/rating updates and the team+player hybrid predictor done |
| 13 | Docker / docker-compose / README | Done |

## Experimental modules

The following modules are research experiments kept alongside the main
pipeline. They are **not** wired into the CLI simulation, API or dashboard
serving paths, and none of them has (yet) demonstrated a backtest advantage
over the main Elo-Poisson line:

- `ratings/v2.py` + `models/rating_v2_poisson.py` — a Glicko-style rating
  with per-team uncertainty and recent-form terms; pending a fair backtest
  comparison against the production Elo.
- `models/neural_outcome.py` — a neural win/draw/loss classifier used purely
  as a comparison baseline (the main line always stays a score-distribution
  model).
- `models/tournament_value.py` + `workflows/distilled_value.py` — a value
  network distilled from a single simulator snapshot; it reproduces that one
  snapshot almost exactly, so it is functionally a cached lookup and serves
  only as a GPU-training exercise.

## Data notes

The MVP uses
[`martj42/international_results`](https://github.com/martj42/international_results)
`results.csv` as the default historical data. It is CC0-1.0 licensed, covers
men's senior national-team matches, and provides date, teams, score,
competition and neutral-venue fields.

Known limitation: full-time scores in the data may include extra time and
exclude penalty shootouts. A strict 90-minute goal model would need to
identify and handle knockout matches that went to extra time.

The 495 official third-place mappings for the 2026 Round of 32 are frozen
from FIFA Regulations Annexe C into
`data/worldcup/third_place_mapping_2026.csv`.

---

# 世界杯预测系统

[English](#world-cup-predictor) | **中文**

世界杯动态预测系统的 MVP。当前版本实现：

- 历史比赛 CSV 加载、字段校验和队名别名统一
- World Football Elo 逐场更新
- Elo 差到预期进球的泊松模型拟合
- 单场比分矩阵、胜平负概率和最可能比分
- 时间切分回测、RPS、Log Loss、Brier Score 和校准表
- 模型参数与 Elo 快照的 JSON 持久化
- 训练、排名和预测 CLI
- 2026 世界杯 48 队模拟器（FIFA 第 13 条排名规则、Annexe C 官方 495 种第三名对阵映射）
- 赛制配置化：48 队 2026 赛制与 32 队 2018/2022 赛制（含旧版排名规则）共用同一个模拟器
- 2018/2022 整届历史回测：各轮晋级概率对照真实结果的技能分——见
  [docs/tournament_backtest.md](docs/tournament_backtest.md)
- 模拟前 catch-up：下载最新数据、回填比分、重训模型，并把已踢完的淘汰赛钉为真实胜者
- Parquet processed data
- PostgreSQL schema / SQLAlchemy 存储层
- FastAPI API
- Streamlit dashboard
- Celery + Redis 动态更新入口
- 完整 Dixon-Coles attack/defense MLE + ρ 低比分修正
- Empirical-Bayes 层级 Poisson 模型 + 预测区间
- 球员级 squad adjustment 和 team+player hybrid predictor
- API-Football 实时 squad、伤病、首发和球员比赛评分同步
- Docker / docker-compose 部署骨架

## 环境

项目使用独立 Conda 环境 `WC`：

```bash
conda activate WC
python -m pip install -e ".[dev,bayes]"
```

也可以从环境文件重建：

```bash
conda env create -f environment.yml
```

可选依赖组：`dev`（pytest）、`bayes`（PyMC + ArviZ，层级贝叶斯模型用）、
`gpu`（PyTorch，CUDA wheel 安装见下方 CPU/GPU 一节）。除贝叶斯模型和 GPU
加速外，其余功能只需基础依赖即可运行。

## CPU / GPU 计算

默认 `--device auto`：有可用的 CUDA PyTorch 和 NVIDIA 驱动时使用 GPU，否则自动使用 CPU。
也可显式指定 `--device cpu` 或 `--device cuda`。显式 CUDA 不可用会报错，不会静默降级。

GPU 是可选依赖，先在 `WC` 环境按本机 CUDA/驱动版本安装匹配的
[PyTorch CUDA wheel](https://pytorch.org/get-started/locally/)，再安装项目 GPU
额外依赖：

```bash
python -m pip install -e ".[gpu]"
```

Elo 逐场更新、Pandas 处理和 SciPy MLE 训练仍在 CPU 上执行；GPU 用于
Elo-Poisson 的比分矩阵计算，并可供单场预测和世界杯模拟器选择。CPU/GPU
均使用 float64，结果应只存在浮点舍入级别的微小差异。

## CSV Schema

输入文件必须包含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `date` | `YYYY-MM-DD` | 比赛日期 |
| `home_team` | string | 主队 |
| `away_team` | string | 客队 |
| `home_goals` | integer | 主队进球 |
| `away_goals` | integer | 客队进球 |
| `competition_type` | string | 赛事类型 |
| `neutral_venue` | boolean | 是否中立场 |

`home_score` / `away_score` / `tournament` / `neutral` 也可作为兼容列名。

## 使用

示例数据只用于验证程序，不代表真实比赛：

```bash
worldcup-predictor train \
  --matches data/examples/synthetic_matches.csv \
  --output models/elo_poisson_v1.json

worldcup-predictor rankings \
  --model models/elo_poisson_v1.json

worldcup-predictor predict \
  --model models/elo_poisson_v1.json \
  --home Atlas --away Comet --neutral \
  --device auto

worldcup-predictor backtest \
  --matches data/raw/international_results.csv \
  --cutoff 2024-01-01

# simulate 默认先执行 catch-up：下载最新比赛数据、回填小组赛比分、重训模型，
# 并把已经踢完的淘汰赛"钉"为真实胜者，只对未踢的比赛做蒙特卡洛。
# --offline 跳过下载与重训；--no-condition-knockouts 关闭真实淘汰赛结果条件化。
worldcup-predictor simulate \
  --simulations 10000 \
  --device cuda \
  --output data/processed/simulation_2026.csv

# 也可以单独补齐数据与模型（不模拟）：
worldcup-predictor catch-up

worldcup-predictor benchmark-prediction \
  --model models/elo_poisson_current.json \
  --home Argentina --away France --iterations 10000

worldcup-predictor export-parquet \
  --matches data/raw/international_results.csv \
  --output data/processed/matches.parquet

worldcup-predictor train-dixon-coles \
  --matches data/raw/international_results.csv \
  --since 2018-01-01 \
  --max-iterations 200 \
  --output models/dixon_coles_current.json

# 公平动态对照：Elo 逐场更新 vs Dixon-Coles 滚动重拟合
worldcup-predictor compare-dixon-coles \
  --matches data/raw/international_results.csv \
  --cutoff 2024-01-01

# 整届回测：用赛前数据训练，模拟整届世界杯，对照真实晋级结果给各轮概率打分
worldcup-predictor backtest-tournament \
  --matches data/raw/international_results.csv \
  --groups data/worldcup/groups_2018.csv \
  --fixtures data/worldcup/fixtures_2018.csv \
  --actual data/worldcup/actual_2018.csv \
  --format wc32 --train-before 2018-06-14 --label wc2018

worldcup-predictor train-bayesian \
  --matches data/raw/international_results.csv \
  --since 2018-01-01 \
  --max-iterations 200 \
  --posterior-draws 100 \
  --output models/bayesian_current.json

worldcup-predictor predict-player-adjusted \
  --model models/elo_poisson_v1.json \
  --players data/examples/synthetic_players.csv \
  --home Atlas --away Comet --neutral
```

### 实时球员数据

Step 12 使用 API-Football v3 作为真实数据供应商。配置 API key 后，`sync-api-football`
会同步世界杯 (`league=1`, `season=2026`) 的球队、当前阵容和伤病；传入已开赛的
`--fixture-id` 时还会同步确认首发和逐场球员评分。

```bash
export API_FOOTBALL_KEY='your-api-football-key'

worldcup-predictor sync-api-football \
  --database-url postgresql+psycopg://worldcup:worldcup@localhost:5432/worldcup

worldcup-predictor predict-player-adjusted \
  --model models/elo_poisson_current.json \
  --database-url postgresql+psycopg://worldcup:worldcup@localhost:5432/worldcup \
  --home Argentina --away France --neutral
```

Docker 环境中设置 `API_FOOTBALL_KEY` 与 `API_FOOTBALL_SYNC_ENABLED=true` 后，Celery
Beat 每 6 小时执行一次同步。没有 key 时任务会安全跳过，不会影响球队级预测。

下载用于 MVP 的 CC0 历史国家队比赛数据：

```bash
worldcup-predictor fetch-data \
  --output data/raw/international_results.csv
```

下载命令会同时生成 `.metadata.json`，记录来源 URL、SHA-256、字节数和下载时间。

运行测试：

```bash
pytest
```

启动 API 和 dashboard：

```bash
uvicorn worldcup_predictor.api.main:app --reload
streamlit run app/dashboard.py
```

Docker：

```bash
docker compose up --build
```

## 路线状态

| Step | 内容 | 状态 |
|---|---|---|
| 1 | CSV 跑通 Elo | 完成 |
| 2 | Elo + 泊松 `predict_match()` | 完成 |
| 3 | Backtest + RPS + calibration | 完成 |
| 4 | 世界杯模拟器 | 完成 |
| 5 | Parquet 固化 processed data | 完成 |
| 6 | PostgreSQL schema / storage | 存储层、schema 和测试完成；服务链路（API / 动态更新）仍以 JSON/CSV 持久化，尚未接入 PostgreSQL——目前只有 API-Football 球员同步写库 |
| 7 | FastAPI | 完成 |
| 8 | Streamlit dashboard | 完成 |
| 9 | Celery + Redis 动态更新 | 完成 workflow、fixture 更新、剩余赛程重预测和 Celery task |
| 10 | Dixon-Coles 升级 | 完成可辨识约束、解析梯度、ρ 修正、保存/加载、CLI 和真实 2018+ 数据收敛验证；滚动重拟合的公平对照在 RPS/LogLoss/Brier 上全面优于 Elo-泊松并修复平局低估——见 [docs/model_comparison.md](docs/model_comparison.md)。`simulate --model models/dixon_coles_current.json` 可直接用它跑模拟 |
| 11 | PyMC 贝叶斯升级 | 完成 Empirical-Bayes 与 PyMC MCMC 层级模型、后验抽样预测区间 |
| 12 | 球员级扩展 | 完成 CSV 与 API-Football 同步、PostgreSQL 落库、伤病/首发/评分更新和 team+player hybrid predictor |
| 13 | Docker / docker-compose / README | 完成 |

## 实验模块

以下模块是与主线并存的研究性实验，**没有**接入 CLI 模拟、API 或 dashboard
服务链路，也都尚未在回测中证明优于主线 Elo-泊松：

- `ratings/v2.py` + `models/rating_v2_poisson.py`——Glicko 风格评分，带
  每队不确定度与近期状态项；还欠一场与生产 Elo 的公平回测对照。
- `models/neural_outcome.py`——神经网络胜平负分类器，仅作对照基线（主线
  始终是比分分布模型）。
- `models/tournament_value.py` + `workflows/distilled_value.py`——从单次
  模拟器快照蒸馏出的价值网络；它几乎精确复现那一份快照，功能上等价于
  缓存查表，仅作为 GPU 训练练习保留。

## 数据说明

MVP 历史数据默认使用
[`martj42/international_results`](https://github.com/martj42/international_results)
的 `results.csv`。该数据采用 CC0-1.0，包含男子成年国家队比赛，并提供日期、球队、比分、赛事和中立场字段。

已知限制：数据中的全场比分可能包含加时赛，不包含点球大战。用于严格的 90 分钟进球模型前，需要进一步识别并处理进入加时赛的淘汰赛。

2026 Round of 32 中第三名球队的 495 种官方组合映射已从 FIFA Regulations
Annexe C 固化到 `data/worldcup/third_place_mapping_2026.csv`。
