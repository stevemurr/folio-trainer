"""Pydantic v2 config schema mirroring allocation_model_default_config.yaml."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Strategy profiles
# ---------------------------------------------------------------------------


class StrategyProfileConfig(BaseModel):
    """Parameter overrides for a named strategy profile."""

    lambda_turnover: float | None = None
    lambda_cost: float | None = None
    lambda_concentration: float | None = None
    distillation_temperature: float | None = None
    inference_temperature: float | None = None
    rebalance_band: float | None = None
    partial_rebalance_alpha: float | None = None


BUILTIN_STRATEGY_PROFILES: dict[str, StrategyProfileConfig] = {
    "aggressive": StrategyProfileConfig(
        lambda_turnover=0.05,
        lambda_cost=0.30,
        lambda_concentration=0.0,
        distillation_temperature=0.05,
        inference_temperature=0.20,
        rebalance_band=0.0025,
        partial_rebalance_alpha=0.75,
    ),
    "neutral": StrategyProfileConfig(
        lambda_turnover=0.20,
        lambda_cost=1.0,
        lambda_concentration=0.0,
        distillation_temperature=0.10,
        inference_temperature=0.30,
        rebalance_band=0.005,
        partial_rebalance_alpha=0.50,
    ),
    "conservative": StrategyProfileConfig(
        lambda_turnover=0.50,
        lambda_cost=2.0,
        lambda_concentration=0.30,
        distillation_temperature=0.15,
        inference_temperature=0.40,
        rebalance_band=0.02,
        partial_rebalance_alpha=0.25,
    ),
}


class UniverseConfig(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    include_cash: bool = True
    cash_ticker: str = "CASH"
    max_single_name_weight: float = 0.25
    long_only: bool = True
    leverage_allowed: bool = False
    shorting_allowed: bool = False


class CalendarConfig(BaseModel):
    exchange: str = "XNYS"
    timezone: str = "America/New_York"
    signal_cutoff_time: str = "18:00"


class ExecutionConfig(BaseModel):
    price_convention: str = "next_open"
    rebalance_band: float = 0.005
    partial_rebalance_alpha: float = 0.50


class HorizonsConfig(BaseModel):
    primary: int = 20
    optional: list[int] = Field(default_factory=lambda: [5, 60])

    @property
    def all_horizons(self) -> list[int]:
        return sorted({self.primary, *self.optional})


class FeaturesConfig(BaseModel):
    return_windows: list[int] = Field(default_factory=lambda: [1, 5, 20, 60, 120])
    vol_windows: list[int] = Field(default_factory=lambda: [5, 20, 60, 120])
    dd_windows: list[int] = Field(default_factory=lambda: [20, 60, 120])
    beta_windows: list[int] = Field(default_factory=lambda: [60, 120])
    corr_windows: list[int] = Field(default_factory=lambda: [20, 60])
    liquidity_windows: list[int] = Field(default_factory=lambda: [20, 60])
    pca_components: int = 3


class CandidateSearchConfig(BaseModel):
    dirichlet_candidates_per_day: int = 5000
    dirichlet_alpha_mix: list[float] = Field(default_factory=lambda: [0.05, 0.20, 1.0, 5.0])
    top_k: int = 5
    distillation_temperature: float = 0.10
    local_perturbations_per_seed: int = 100
    sparse_k_assets: int = 30
    deterministic_candidates: list[str] = Field(
        default_factory=lambda: [
            "prev_live",
            "prev_target",
            "equal_weight",
            "cash_only",
            "inverse_vol",
            "min_variance",
        ]
    )


class TeacherObjectiveConfig(BaseModel):
    lambda_turnover: float = 0.20
    lambda_cost: float = 1.0
    lambda_concentration: float = 0.0
    epsilon: float = 1e-8


class LiquidityBucket(BaseModel):
    adv_min: float
    adv_max: float
    spread_bps_proxy: float


class CostModelConfig(BaseModel):
    commission_bps: float = 0.0
    regulatory_bps: float = 0.0
    spread_bps_proxy_default: float = 2.0
    adv_floor: float = 100_000.0
    impact_coeff: float = 10.0
    liquidity_buckets: list[LiquidityBucket] = Field(
        default_factory=lambda: [
            LiquidityBucket(adv_min=0, adv_max=1_000_000, spread_bps_proxy=10.0),
            LiquidityBucket(adv_min=1_000_000, adv_max=10_000_000, spread_bps_proxy=5.0),
            LiquidityBucket(adv_min=10_000_000, adv_max=1e20, spread_bps_proxy=2.0),
        ]
    )

    def get_spread_bps(self, adv: float) -> float:
        """Look up spread proxy from liquidity buckets."""
        for bucket in self.liquidity_buckets:
            if bucket.adv_min <= adv < bucket.adv_max:
                return bucket.spread_bps_proxy
        return self.spread_bps_proxy_default


class SplitsConfig(BaseModel):
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    walkforward_train_years: int = 5
    walkforward_val_years: int = 1
    walkforward_test_years: int = 1
    purge_days: int = 120

    @model_validator(mode="after")
    def _check_fracs(self) -> SplitsConfig:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            msg = f"Split fractions must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


class DirectWeightModelConfig(BaseModel):
    kind: str = "gbm"
    learning_rate: float = 0.03
    max_depth: int = 6
    num_leaves: int = 64
    n_estimators: int = 500
    temperature: float = 0.30
    l2_weight_change_penalty: float = 0.01
    use_confidence_weights: bool = True


class UtilityModelConfig(BaseModel):
    enabled: bool = True
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.10


class ModelConfig(BaseModel):
    direct_weight_model: DirectWeightModelConfig = Field(default_factory=DirectWeightModelConfig)
    utility_model: UtilityModelConfig = Field(default_factory=UtilityModelConfig)


class TrainingConfig(BaseModel):
    early_stopping_rounds: int = 50
    objective_metric: str = "validation_net_sharpe"
    secondary_metrics: list[str] = Field(
        default_factory=lambda: [
            "validation_max_drawdown",
            "validation_turnover",
            "validation_kl",
        ]
    )


class ReportingConfig(BaseModel):
    generate_html_reports: bool = True
    generate_plots: bool = True


class PipelineConfig(BaseModel):
    """Top-level config for the allocation pipeline."""

    project_name: str = "allocation_model_v1"
    random_seed: int = 42
    strategy: str | None = None
    custom_profiles: dict[str, StrategyProfileConfig] = Field(default_factory=dict)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    horizons: HorizonsConfig = Field(default_factory=HorizonsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    candidate_search: CandidateSearchConfig = Field(default_factory=CandidateSearchConfig)
    teacher_objective: TeacherObjectiveConfig = Field(default_factory=TeacherObjectiveConfig)
    cost_model: CostModelConfig = Field(default_factory=CostModelConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    # Runtime paths (not in YAML, set programmatically)
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
