import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { invoke } from "@tauri-apps/api/core";

type ModelOutput = {
  model_id?: string;
  is_trained?: boolean;
  prediction?: string | number | null;
  probabilities?: Record<string, number>;
  confidence?: number | null;
  reason?: string | null;
  regime?: string | null;
  direction?: string | null;
  strategy?: string | null;
  iv_direction?: string | null;
  accuracy_30d?: number | null;
  anomaly_score?: number | null;
  is_anomalous?: boolean;
  data_quality_score?: number | null;
  recommended_size?: number | null;
  iv_forecast_3d?: number | null;
  mape_30d?: number | null;
};

type MLPrediction = {
  symbol?: string;
  computed_at?: number;
  composite_direction?: "UP" | "FLAT" | "DOWN" | null;
  composite_probabilities?: Record<string, number>;
  composite_confidence?: number | null;
  model_agreement?: number | null;
  ensemble_method?: string;
  component_outputs?: Record<string, ModelOutput>;
  action_recommendation?: {
    trade?: boolean;
    action?: "UNVALIDATED" | "STAY FLAT" | "CONSIDER LONG" | "CONSIDER SHORT" | string;
    strategy?: string | null;
    size?: number | null;
    reasoning?: string[];
    eligibility?: {
      eligible?: boolean;
      state?: "UNVALIDATED" | "VALIDATED_NON_ACTIONABLE" | "ACTIONABLE" | string;
      reasons?: string[];
    };
  };
  accuracy_tracking?: {
    ensemble_accuracy_30d?: number | null;
    ensemble_accuracy_7d?: number | null;
    best_model_30d?: string | null;
    worst_model_30d?: string | null;
  };
  has_symbol_model?: boolean;
  model_source?: "symbol_specific" | "nifty_proxy" | string;
  ohlc_forecast?: OHLCForecast | null;
  error?: string;
};

type OHLCForecast = {
  symbol: string;
  date: string;
  current_close?: number;
  forecast_open: number;
  forecast_high: number;
  forecast_low: number;
  forecast_close: number;
  expected_range: number;
  expected_range_pct: number;
  confidence: "high" | "medium" | "low" | string;
  method_outputs?: Record<string, { open?: number; high?: number; low?: number; close?: number } | null>;
  key_levels?: Record<string, number>;
  gap_prediction?: {
    direction?: string;
    expected_gap_pct?: number;
    expected_open?: number;
  };
  close_mae_last_10d?: number | null;
};

type TrainingStatus = {
  running?: boolean;
  stage?: string;
  current_symbol?: string | null;
  done?: number;
  total?: number;
  trained?: number;
  errors?: number;
  last_report?: {
    error?: string | null;
    symbol?: string;
    models_trained?: string[];
  } | null;
};

type TrainedSummary = {
  count?: number;
  total_fno?: number;
  symbols?: string[];
};

type ModelRow = {
  key: string;
  model: string;
  prediction: string;
  confidence: number | null;
  accuracy: number | null;
  trained?: boolean;
  warning?: string | null;
};

const percentFormatter = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
  style: "percent",
});

type MLPanelProps = {
  symbol?: string;
};

export function MLPanel({ symbol = "NIFTY" }: MLPanelProps) {
  const [prediction, setPrediction] = useState<MLPrediction | null>(null);
  const [forecast, setForecast] = useState<OHLCForecast | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [trainedSummary, setTrainedSummary] = useState<TrainedSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [trainMessage, setTrainMessage] = useState("");

  useEffect(() => {
    let active = true;

    const loadPrediction = async () => {
      try {
        const result = await invoke<MLPrediction>("get_ml_prediction", {
          symbol,
        });
        if (!active) {
          return;
        }
        setPrediction(result);
        setForecast(result.ohlc_forecast ?? null);
        setMessage(result.error ?? "");
      } catch (error) {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadPrediction();
    const timer = window.setInterval(loadPrediction, 60_000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [symbol]);

  useEffect(() => {
    let active = true;

    const loadForecast = async () => {
      try {
        const result = await invoke<OHLCForecast>("get_ohlc_forecast", { symbol });
        if (active) {
          setForecast(result);
        }
      } catch {
        if (active && !prediction?.ohlc_forecast) {
          setForecast(null);
        }
      }
    };

    loadForecast();
    const timer = window.setInterval(loadForecast, 60_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [prediction?.ohlc_forecast, symbol]);

  useEffect(() => {
    let active = true;

    const loadTraining = async () => {
      try {
        const [status, summary] = await Promise.all([
          invoke<TrainingStatus>("get_training_status"),
          invoke<TrainedSummary>("get_trained_symbols"),
        ]);
        if (!active) {
          return;
        }
        setTrainingStatus(status);
        setTrainedSummary(summary);
      } catch {
        if (active) {
          setTrainingStatus(null);
        }
      }
    };

    loadTraining();
    const timer = window.setInterval(loadTraining, 5_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const startSingleTraining = async () => {
    setTrainMessage(`Training ${symbol}...`);
    try {
      const result = await invoke<{ status?: string }>("train_single_symbol", {
        force: false,
        symbol,
      });
      setTrainMessage(result.status === "already_running" ? "Training already running." : `Queued ${symbol}.`);
    } catch (error) {
      setTrainMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const startAllTraining = async () => {
    setTrainMessage("Queued all FNO stocks.");
    try {
      const result = await invoke<{ status?: string }>("train_all_stocks", { force: false });
      setTrainMessage(result.status === "already_running" ? "Training already running." : "Training all FNO stocks.");
    } catch (error) {
      setTrainMessage(error instanceof Error ? error.message : String(error));
    }
  };

  const rows = useMemo(() => buildRows(prediction), [prediction]);
  const components = prediction?.component_outputs ?? {};
  const direction = prediction?.composite_direction ?? null;
  const confidence = asNumber(prediction?.composite_confidence);
  const agreement = asNumber(prediction?.model_agreement);
  const probabilities = prediction?.composite_probabilities ?? {};
  const action = prediction?.action_recommendation;
  const actionEligible = action?.trade === true && action?.eligibility?.eligible === true;
  const anomaly = components.m05_anomaly;
  const lastUpdated = formatTimestamp(prediction?.computed_at);
  const needsSymbolTraining =
    symbol !== "NIFTY" && prediction?.has_symbol_model === false;

  return (
    <section style={wrapperStyle} aria-label="ML prediction panel">
      <OHLCForecastCard forecast={forecast} />

      {needsSymbolTraining || trainingStatus?.running ? (
        <TrainingCard
          needsSymbolTraining={needsSymbolTraining}
          onTrainAll={startAllTraining}
          onTrainSymbol={startSingleTraining}
          status={trainingStatus}
          summary={trainedSummary}
          symbol={symbol}
          trainMessage={trainMessage}
        />
      ) : null}

      <div style={cardsGridStyle}>
        <div style={ensembleCardStyle}>
          <div style={topLineStyle}>
            <span style={labelStyle}>
              ML Ensemble {prediction?.model_source === "nifty_proxy" ? "(NIFTY proxy)" : ""}
            </span>
            <span style={timestampStyle}>
              {loading && prediction === null ? "Training..." : lastUpdated}
            </span>
          </div>

          {anomaly?.is_anomalous ? (
            <div style={anomalyBannerStyle}>
              Anomaly circuit breaker active. Confidence suppressed.
            </div>
          ) : null}

          <div style={directionRowStyle}>
            <div style={{ ...directionStyle, color: predictionColor(direction) }}>
              {direction ?? "--"}
            </div>
            <div style={confidenceBlockStyle}>
              <div style={metricLabelStyle}>Confidence</div>
              <div style={{ ...metricValueStyle, color: confidenceColor(confidence) }}>
                {formatPercent(confidence)}
              </div>
            </div>
            <div style={confidenceBlockStyle}>
              <div style={metricLabelStyle}>Agreement</div>
              <div style={{ ...metricValueStyle, color: confidenceColor(agreement) }}>
                {formatPercent(agreement)}
              </div>
            </div>
          </div>

          <ProbabilityBar probabilities={probabilities} />

          <div style={actionGridStyle}>
            <Metric
              label="Backend Action"
              value={actionEligible ? action?.action ?? "NON-ACTIONABLE" : "NON-ACTIONABLE"}
              color={actionEligible ? "#22c55e" : "#facc15"}
            />
            <Metric label="Validation" value={action?.eligibility?.state ?? "UNVALIDATED"} />
            <Metric label="Strategy" value={actionEligible ? action?.strategy ?? "--" : "--"} />
            <Metric label="Size" value={actionEligible ? formatSize(action?.size) : "--"} />
          </div>

          <div style={reasonListStyle}>
            {(action?.reasoning ?? ["Waiting for model agreement."]).slice(0, 3).map((reason) => (
              <div key={reason} style={reasonStyle}>
                {reason}
              </div>
            ))}
          </div>

          {message ? <div style={messageStyle}>{message}</div> : null}
        </div>

        <div style={gridCardStyle}>
          <div style={tableHeaderStyle}>
            <span>Model</span>
            <span>Prediction</span>
            <span>Confidence</span>
            <span>30d Accuracy</span>
          </div>
          <div style={tableBodyStyle}>
            {rows.map((row) => (
              <div key={row.key} style={tableRowStyle} title={row.warning ?? undefined}>
                <span style={modelNameStyle}>{row.model}</span>
                <span style={{ ...predictionCellStyle, color: predictionColor(row.prediction) }}>
                  {row.trained === false && row.prediction === "--"
                    ? statusText(row.warning)
                    : row.prediction}
                </span>
                <span style={{ color: confidenceColor(row.confidence) }}>
                  {formatPercent(row.confidence)}
                </span>
                <span style={{ color: accuracyColor(row.accuracy) }}>
                  {formatPercent(row.accuracy)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function OHLCForecastCard({ forecast }: { forecast: OHLCForecast | null }) {
  const levels = forecast?.key_levels ?? {};
  const methods = forecast?.method_outputs ?? {};
  return (
    <div style={forecastCardStyle}>
      <div style={forecastHeaderStyle}>
        <div>
          <span style={labelStyle}>Next Day Forecast</span>
          <strong style={forecastDateStyle}>{forecast?.date ?? "--"}</strong>
        </div>
        <span style={{ ...confidencePillStyle, color: confidenceColor(confidenceScore(forecast?.confidence)) }}>
          Confidence: {(forecast?.confidence ?? "--").toUpperCase()}
        </span>
      </div>

      <div style={ohlcGridStyle}>
        <ForecastMetric label="OPEN" value={formatPrice(forecast?.forecast_open)} />
        <ForecastMetric label="HIGH" value={formatPrice(forecast?.forecast_high)} color="#22c55e" />
        <ForecastMetric label="LOW" value={formatPrice(forecast?.forecast_low)} color="#ef5350" />
        <ForecastMetric
          label="CLOSE"
          value={formatPrice(forecast?.forecast_close)}
          color={closeForecastColor(forecast)}
        />
      </div>

      <div style={rangeRowStyle}>
        <span>
          Expected Range: {formatPrice(forecast?.expected_range)} pts ({formatPctPoints(forecast?.expected_range_pct)})
        </span>
        <div style={rangeTrackStyle}>
          <div style={{ ...rangeFillStyle, width: `${rangeWidth(forecast?.expected_range_pct)}%` }} />
        </div>
        <span>
          Gap: {gapText(forecast)} {"->"} Opens near {formatPrice(forecast?.gap_prediction?.expected_open)}
        </span>
      </div>

      <div style={levelsStyle}>
        <span>R3: {formatPrice(levels.r3)}</span>
        <span>R2: {formatPrice(levels.r2)}</span>
        <span>R1: {formatPrice(levels.r1)}</span>
        <strong style={pivotStyle}>PIVOT: {formatPrice(levels.pivot)}</strong>
        <span>S1: {formatPrice(levels.s1)}</span>
        <span>S2: {formatPrice(levels.s2)}</span>
        <span>S3: {formatPrice(levels.s3)}</span>
      </div>

      <div style={methodsStyle}>
        <MethodLine label="ATR method" method={methods.atr_method} />
        <MethodLine label="IV method" method={methods.iv_method} />
        <MethodLine label="Gap method" method={methods.gap_method} />
        <span style={accuracyLineStyle}>
          10d measured close MAE: {formatCurrency(forecast?.close_mae_last_10d)}
        </span>
      </div>
    </div>
  );
}

function TrainingCard({
  needsSymbolTraining,
  onTrainAll,
  onTrainSymbol,
  status,
  summary,
  symbol,
  trainMessage,
}: {
  needsSymbolTraining: boolean;
  onTrainAll: () => void;
  onTrainSymbol: () => void;
  status: TrainingStatus | null;
  summary: TrainedSummary | null;
  symbol: string;
  trainMessage: string;
}) {
  const progress = progressPct(status);
  return (
    <div style={trainingCardStyle}>
      <div style={trainingTopStyle}>
        <strong style={trainingTitleStyle}>
          {needsSymbolTraining ? `No ML model for ${symbol}` : trainingLabel(status)}
        </strong>
        <span style={timestampStyle}>
          Trained: {summary?.count ?? 0}/{summary?.total_fno ?? "--"} stocks
        </span>
      </div>
      <div style={trainingTextStyle}>
        {status?.running
          ? `Training ${status.current_symbol ?? symbol}...`
          : needsSymbolTraining
            ? `Train ${symbol} model for real stock-specific predictions.`
            : "Stock model training ready."}
      </div>
      <div style={trainingProgressTrackStyle}>
        <div style={{ ...trainingProgressFillStyle, width: `${progress}%` }} />
      </div>
      <div style={trainingActionsStyle}>
        <button onClick={onTrainSymbol} style={trainingButtonStyle} type="button">
          Train {symbol} Model
        </button>
        <button onClick={onTrainAll} style={trainingButtonStyle} type="button">
          Train All FNO Stocks
        </button>
        <span style={trainingTextStyle}>{trainMessage || "Single stock takes about 5-30 seconds."}</span>
      </div>
    </div>
  );
}

function ForecastMetric({
  color = "#f8fafc",
  label,
  value,
}: {
  color?: string;
  label: string;
  value: string;
}) {
  return (
    <div style={forecastMetricStyle}>
      <span style={metricLabelStyle}>{label}</span>
      <strong style={{ ...forecastValueStyle, color }}>{value}</strong>
    </div>
  );
}

function MethodLine({
  label,
  method,
}: {
  label: string;
  method: { high?: number; low?: number; close?: number } | null | undefined;
}) {
  return (
    <span>
      {label}: H:{formatPrice(method?.high)} L:{formatPrice(method?.low)} C:{formatPrice(method?.close)}
    </span>
  );
}

function ProbabilityBar({ probabilities }: { probabilities: Record<string, number> }) {
  const up = clamp01(asNumber(probabilities.UP) ?? 0);
  const flat = clamp01(asNumber(probabilities.FLAT) ?? 0);
  const down = clamp01(asNumber(probabilities.DOWN) ?? 0);
  return (
    <div style={probabilityWrapStyle}>
      <div style={probabilityTrackStyle}>
        <div style={{ ...probabilityFillStyle, background: "#22c55e", width: `${up * 100}%` }} />
        <div style={{ ...probabilityFillStyle, background: "#facc15", width: `${flat * 100}%` }} />
        <div style={{ ...probabilityFillStyle, background: "#ef5350", width: `${down * 100}%` }} />
      </div>
      <div style={probabilityLabelsStyle}>
        <span>UP {formatPercent(up)}</span>
        <span>FLAT {formatPercent(flat)}</span>
        <span>DOWN {formatPercent(down)}</span>
      </div>
    </div>
  );
}

function Metric({ color = "#f8fafc", label, value }: { color?: string; label: string; value: string }) {
  return (
    <div style={metricBoxStyle}>
      <div style={metricLabelStyle}>{label}</div>
      <div style={{ ...metricValueStyle, color }}>{value}</div>
    </div>
  );
}

function buildRows(prediction: MLPrediction | null): ModelRow[] {
  const components = prediction?.component_outputs ?? {};
  const row = (
    key: string,
    model: string,
    output: ModelOutput | undefined,
    pick: (output: ModelOutput | undefined) => string,
    confidence?: number | null,
    accuracy?: number | null,
  ): ModelRow => ({
    key,
    model,
    prediction: pick(output),
    confidence: confidence ?? asNumber(output?.confidence),
    accuracy: accuracy ?? asNumber(output?.accuracy_30d),
    trained: output?.is_trained,
    warning: output?.reason ?? null,
  });

  return [
    row("m01", "Vol Regime", components.m01_vol_regime, (item) => item?.regime ?? textValue(item?.prediction)),
    row("m02", "Direction", components.m02_direction, (item) => item?.direction ?? textValue(item?.prediction)),
    row("m03", "Strategy", components.m03_strategy, (item) => item?.strategy ?? textValue(item?.prediction)),
    row("m04", "IV Reversion", components.m04_iv_reversion, (item) => item?.iv_direction ?? textValue(item?.prediction)),
    row("m05", "Anomaly", components.m05_anomaly, (item) => (item?.is_anomalous ? "Anomalous" : "Normal"), asNumber(components.m05_anomaly?.anomaly_score), null),
    row("m06", "Regime Return", components.m06_regime_return, (item) => item?.direction ?? textValue(item?.prediction)),
    row("m07", "IV Forecast", components.m07_iv_forecast, (item) => formatForecast(item), null, asNumber(components.m07_iv_forecast?.mape_30d)),
    row("m08", "GP Regime", components.m08_gp_regime, (item) => textValue(item?.prediction), qualityAsConfidence(components.m08_gp_regime), null),
    row("m09", "RL Sizer", components.m09_rl_sizer, (item) => formatSize(item?.recommended_size), asNumber(components.m09_rl_sizer?.confidence), null),
    {
      key: "m10",
      model: "ENSEMBLE",
      prediction: prediction?.composite_direction ?? "--",
      confidence: asNumber(prediction?.composite_confidence),
      accuracy: asNumber(prediction?.accuracy_tracking?.ensemble_accuracy_30d),
      trained: true,
      warning: null,
    },
  ];
}

function asNumber(value: number | null | undefined) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function textValue(value: string | number | null | undefined) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  return String(value);
}

function formatPercent(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return percentFormatter.format(clamp01(number));
}

function formatSize(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  if (number <= 0) {
    return "0";
  }
  return `${(number * 100).toFixed(1)}% max`;
}

function formatForecast(output: ModelOutput | undefined) {
  const value = asNumber(output?.iv_forecast_3d);
  if (value === null) {
    return "--";
  }
  const pct = value <= 1 ? value * 100 : value;
  return `${pct.toFixed(1)}% in 3d`;
}

function formatTimestamp(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return new Date(number * 1000).toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatPrice(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  }).format(number);
}

function formatCurrency(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return `Rs ${formatPrice(number)}`;
}

function formatPctPoints(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "--";
  }
  return `${number.toFixed(2)}%`;
}

function qualityAsConfidence(output: ModelOutput | undefined) {
  const quality = asNumber(output?.data_quality_score);
  return quality === null ? null : quality / 100;
}

function statusText(reason: string | null | undefined) {
  if (!reason) {
    return "Training...";
  }
  return reason.toLowerCase().includes("insufficient") || reason.toLowerCase().includes("empty")
    ? "Insufficient data"
    : "Training...";
}

function predictionColor(value: string | null | undefined) {
  const text = String(value ?? "").toUpperCase();
  if (["UP", "LOW", "BUY", "BULL", "NORMAL"].some((item) => text.includes(item))) {
    return "#22c55e";
  }
  if (["DOWN", "HIGH", "BEAR", "ANOMALOUS"].some((item) => text.includes(item))) {
    return "#ef5350";
  }
  if (["FLAT", "MEDIUM", "STABLE", "STAY"].some((item) => text.includes(item))) {
    return "#facc15";
  }
  return "#f8fafc";
}

function confidenceColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "#8a94a6";
  }
  if (number > 0.6) {
    return "#22c55e";
  }
  if (number >= 0.4) {
    return "#facc15";
  }
  return "#ef5350";
}

function accuracyColor(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return "#8a94a6";
  }
  if (number > 0.6) {
    return "#22c55e";
  }
  if (number >= 0.5) {
    return "#facc15";
  }
  return "#ef5350";
}

function confidenceScore(value: string | null | undefined) {
  const text = String(value ?? "").toLowerCase();
  if (text === "high") {
    return 0.75;
  }
  if (text === "medium") {
    return 0.5;
  }
  if (text === "low") {
    return 0.2;
  }
  return null;
}

function closeForecastColor(forecast: OHLCForecast | null) {
  const close = asNumber(forecast?.forecast_close);
  const current = asNumber(forecast?.current_close);
  if (close === null || current === null) {
    return "#f8fafc";
  }
  if (close > current) {
    return "#22c55e";
  }
  if (close < current) {
    return "#ef5350";
  }
  return "#f8fafc";
}

function rangeWidth(value: number | null | undefined) {
  const number = asNumber(value);
  if (number === null) {
    return 0;
  }
  return Math.max(8, Math.min(100, (number / 4) * 100));
}

function gapText(forecast: OHLCForecast | null) {
  const direction = String(forecast?.gap_prediction?.direction ?? "flat")
    .replace("_", " ")
    .toUpperCase();
  return `${direction} (${formatPctPoints(forecast?.gap_prediction?.expected_gap_pct)})`;
}

function progressPct(status: TrainingStatus | null) {
  const done = asNumber(status?.done);
  const total = asNumber(status?.total);
  if (done === null || total === null || total <= 0) {
    return status?.running ? 8 : 0;
  }
  return Math.max(0, Math.min(100, (done / total) * 100));
}

function trainingLabel(status: TrainingStatus | null) {
  if (!status?.running) {
    return "Training idle";
  }
  return status.stage === "single_symbol" ? "Single stock training" : "All-stock training";
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

const wrapperStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 10,
  minWidth: 0,
  width: "100%",
} satisfies CSSProperties;

const cardsGridStyle = {
  display: "grid",
  gap: 10,
  gridTemplateColumns: "minmax(300px, 0.85fr) minmax(560px, 1.15fr)",
  minWidth: 0,
  overflowX: "auto",
  width: "100%",
} satisfies CSSProperties;

const forecastCardStyle = {
  background: "#171b2a",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  boxSizing: "border-box",
  color: "#d1d4dc",
  display: "grid",
  gap: 10,
  minWidth: 0,
  padding: "12px 14px",
} satisfies CSSProperties;

const forecastHeaderStyle = {
  alignItems: "center",
  display: "flex",
  justifyContent: "space-between",
  minWidth: 0,
} satisfies CSSProperties;

const forecastDateStyle = {
  color: "#f8fafc",
  display: "block",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 13,
  lineHeight: 1.2,
  marginTop: 4,
} satisfies CSSProperties;

const confidencePillStyle = {
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  fontWeight: 700,
  lineHeight: 1.2,
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const ohlcGridStyle = {
  display: "grid",
  gap: 8,
  gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
  minWidth: 0,
} satisfies CSSProperties;

const forecastMetricStyle = {
  borderBottom: "1px solid #23283a",
  minWidth: 0,
  paddingBottom: 8,
} satisfies CSSProperties;

const forecastValueStyle = {
  display: "block",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 22,
  fontWeight: 800,
  lineHeight: 1,
  marginTop: 5,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const rangeRowStyle = {
  color: "#cbd5e1",
  display: "grid",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
  gap: 8,
  gridTemplateColumns: "minmax(210px, 0.7fr) minmax(160px, 1fr) minmax(220px, 0.8fr)",
  lineHeight: 1.35,
  minWidth: 0,
} satisfies CSSProperties;

const rangeTrackStyle = {
  alignSelf: "center",
  background: "#0f172a",
  border: "1px solid #23283a",
  height: 8,
  overflow: "hidden",
} satisfies CSSProperties;

const rangeFillStyle = {
  background: "linear-gradient(90deg, #ef5350, #facc15, #22c55e)",
  height: "100%",
} satisfies CSSProperties;

const levelsStyle = {
  color: "#8a94a6",
  display: "grid",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
  gap: 8,
  gridTemplateColumns: "repeat(7, minmax(0, 1fr))",
  lineHeight: 1.25,
  minWidth: 0,
} satisfies CSSProperties;

const pivotStyle = {
  color: "#facc15",
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const methodsStyle = {
  color: "#9ca3af",
  display: "grid",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
  lineHeight: 1.3,
  minWidth: 0,
} satisfies CSSProperties;

const accuracyLineStyle = {
  color: "#facc15",
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const trainingCardStyle = {
  background: "#121827",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  boxSizing: "border-box",
  color: "#d1d4dc",
  display: "grid",
  gap: 8,
  padding: "12px 14px",
} satisfies CSSProperties;

const trainingTopStyle = {
  alignItems: "center",
  display: "flex",
  justifyContent: "space-between",
  minWidth: 0,
} satisfies CSSProperties;

const trainingTitleStyle = {
  color: "#f8fafc",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 13,
  lineHeight: 1.2,
} satisfies CSSProperties;

const trainingTextStyle = {
  color: "#9ca3af",
  fontSize: 12,
  lineHeight: 1.35,
} satisfies CSSProperties;

const trainingProgressTrackStyle = {
  background: "#0f172a",
  border: "1px solid #23283a",
  height: 8,
  overflow: "hidden",
} satisfies CSSProperties;

const trainingProgressFillStyle = {
  background: "#4488ff",
  height: "100%",
  transition: "width 180ms ease",
} satisfies CSSProperties;

const trainingActionsStyle = {
  alignItems: "center",
  display: "grid",
  gap: 8,
  gridTemplateColumns: "160px 170px minmax(180px, 1fr)",
} satisfies CSSProperties;

const trainingButtonStyle = {
  background: "#1a1f33",
  border: "1px solid #35405f",
  borderRadius: 6,
  color: "#f8fafc",
  cursor: "pointer",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 12,
  minHeight: 30,
  padding: "0 10px",
} satisfies CSSProperties;

const ensembleCardStyle = {
  background: "#171b2a",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  boxSizing: "border-box",
  color: "#d1d4dc",
  display: "flex",
  flexDirection: "column",
  gap: 12,
  minHeight: 274,
  minWidth: 300,
  padding: "14px",
} satisfies CSSProperties;

const gridCardStyle = {
  background: "#171b2a",
  border: "1px solid #2a2e39",
  borderRadius: 8,
  boxSizing: "border-box",
  color: "#d1d4dc",
  minHeight: 274,
  minWidth: 560,
  padding: "12px",
} satisfies CSSProperties;

const topLineStyle = {
  alignItems: "center",
  display: "flex",
  gap: 10,
  justifyContent: "space-between",
} satisfies CSSProperties;

const labelStyle = {
  color: "#8a94a6",
  fontSize: 11,
  letterSpacing: 0,
  lineHeight: 1.2,
  textTransform: "uppercase",
} satisfies CSSProperties;

const timestampStyle = {
  color: "#9ca3af",
  fontSize: 11,
  lineHeight: 1.2,
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const anomalyBannerStyle = {
  background: "rgba(239, 83, 80, 0.12)",
  border: "1px solid rgba(239, 83, 80, 0.55)",
  borderRadius: 6,
  color: "#fecaca",
  fontSize: 12,
  lineHeight: 1.35,
  padding: "8px 10px",
} satisfies CSSProperties;

const directionRowStyle = {
  alignItems: "center",
  display: "grid",
  gap: 12,
  gridTemplateColumns: "minmax(90px, 1fr) minmax(82px, auto) minmax(82px, auto)",
} satisfies CSSProperties;

const directionStyle = {
  fontSize: 40,
  fontWeight: 800,
  letterSpacing: 0,
  lineHeight: 1,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const confidenceBlockStyle = {
  minWidth: 82,
} satisfies CSSProperties;

const metricLabelStyle = {
  color: "#8a94a6",
  fontSize: 10,
  letterSpacing: 0,
  lineHeight: 1.2,
  textTransform: "uppercase",
} satisfies CSSProperties;

const metricValueStyle = {
  color: "#f8fafc",
  fontSize: 15,
  fontWeight: 700,
  lineHeight: 1.25,
  minHeight: 19,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const probabilityWrapStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 6,
} satisfies CSSProperties;

const probabilityTrackStyle = {
  background: "#0f172a",
  borderRadius: 6,
  display: "flex",
  height: 10,
  overflow: "hidden",
  width: "100%",
} satisfies CSSProperties;

const probabilityFillStyle = {
  height: "100%",
  transition: "width 160ms ease",
} satisfies CSSProperties;

const probabilityLabelsStyle = {
  color: "#9ca3af",
  display: "flex",
  fontSize: 11,
  justifyContent: "space-between",
  lineHeight: 1.2,
} satisfies CSSProperties;

const actionGridStyle = {
  display: "grid",
  gap: 8,
  gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
} satisfies CSSProperties;

const metricBoxStyle = {
  background: "#101827",
  border: "1px solid #23283a",
  borderRadius: 6,
  minHeight: 50,
  minWidth: 0,
  padding: "8px",
} satisfies CSSProperties;

const reasonListStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
  minHeight: 58,
} satisfies CSSProperties;

const reasonStyle = {
  color: "#cbd5e1",
  fontSize: 11,
  lineHeight: 1.35,
  overflowWrap: "anywhere",
} satisfies CSSProperties;

const messageStyle = {
  color: "#facc15",
  fontSize: 12,
  lineHeight: 1.3,
} satisfies CSSProperties;

const tableHeaderStyle = {
  color: "#8a94a6",
  display: "grid",
  fontSize: 11,
  gap: 8,
  gridTemplateColumns: "1.2fr 1.15fr 0.75fr 0.75fr",
  letterSpacing: 0,
  lineHeight: 1.2,
  padding: "0 8px 8px",
  textTransform: "uppercase",
} satisfies CSSProperties;

const tableBodyStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
} satisfies CSSProperties;

const tableRowStyle = {
  alignItems: "center",
  background: "#101827",
  border: "1px solid #23283a",
  borderRadius: 6,
  display: "grid",
  fontSize: 12,
  gap: 8,
  gridTemplateColumns: "1.2fr 1.15fr 0.75fr 0.75fr",
  lineHeight: 1.25,
  minHeight: 30,
  padding: "6px 8px",
} satisfies CSSProperties;

const modelNameStyle = {
  color: "#d1d4dc",
  fontWeight: 700,
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;

const predictionCellStyle = {
  fontWeight: 700,
  minWidth: 0,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
} satisfies CSSProperties;
