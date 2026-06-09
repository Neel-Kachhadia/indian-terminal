import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  type CandlestickData,
  type HistogramData,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type LineWidth,
} from "lightweight-charts";

type OHLCV = {
  symbol: string;
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type IndicatorValue = number | null;

type IndicatorResponse = {
  timestamps: string[];
  indicators: Record<string, IndicatorValue[]>;
  error?: string;
};

type IndicatorKey =
  | "EMA9"
  | "EMA21"
  | "EMA50"
  | "EMA200"
  | "RSI"
  | "MACD"
  | "BB";

type IndicatorState = Record<IndicatorKey, boolean>;

type IndicatorSeriesRefs = {
  ema9: ISeriesApi<"Line">;
  ema21: ISeriesApi<"Line">;
  ema50: ISeriesApi<"Line">;
  ema200: ISeriesApi<"Line">;
  rsi: ISeriesApi<"Line">;
  macdLine: ISeriesApi<"Line">;
  macdSignal: ISeriesApi<"Line">;
  macdHist: ISeriesApi<"Histogram">;
  bbUpper: ISeriesApi<"Line">;
  bbMiddle: ISeriesApi<"Line">;
  bbLower: ISeriesApi<"Line">;
};

const INDICATOR_BUTTONS: { key: IndicatorKey; label: string }[] = [
  { key: "EMA9", label: "EMA9" },
  { key: "EMA21", label: "EMA21" },
  { key: "EMA50", label: "EMA50" },
  { key: "EMA200", label: "EMA200" },
  { key: "RSI", label: "RSI" },
  { key: "MACD", label: "MACD" },
  { key: "BB", label: "BB" },
];

const DEFAULT_ACTIVE: IndicatorState = {
  BB: false,
  EMA9: false,
  EMA21: true,
  EMA50: true,
  EMA200: false,
  MACD: false,
  RSI: true,
};

const INDICATOR_REQUEST = [
  { name: "EMA", params: { period: 9 } },
  { name: "EMA", params: { period: 21 } },
  { name: "EMA", params: { period: 50 } },
  { name: "EMA", params: { period: 200 } },
  { name: "RSI", params: { period: 14 } },
  { name: "MACD", params: { fast: 12, signal: 9, slow: 26 } },
  { name: "BBANDS", params: { period: 20, std: 2.0 } },
];

export function ChartPanel() {
  const mainContainerRef = useRef<HTMLDivElement | null>(null);
  const rsiContainerRef = useRef<HTMLDivElement | null>(null);
  const macdContainerRef = useRef<HTMLDivElement | null>(null);
  const chartRefs = useRef<IChartApi[]>([]);
  const seriesRefs = useRef<IndicatorSeriesRefs | null>(null);
  const [activeIndicators, setActiveIndicators] =
    useState<IndicatorState>(DEFAULT_ACTIVE);
  const [message, setMessage] = useState("Loading chart...");

  useEffect(() => {
    const mainContainer = mainContainerRef.current;
    const rsiContainer = rsiContainerRef.current;
    const macdContainer = macdContainerRef.current;
    if (!mainContainer || !rsiContainer || !macdContainer) {
      return;
    }

    let active = true;
    const mainChart = createPanelChart(mainContainer, 500, "#0a0a0a");
    const rsiChart = createPanelChart(rsiContainer, 120, "#0d0d0d");
    const macdChart = createPanelChart(macdContainer, 100, "#0d0d0d");
    chartRefs.current = [mainChart, rsiChart, macdChart];

    const candleSeries = mainChart.addSeries(CandlestickSeries, {
      borderDownColor: "#ef5350",
      borderUpColor: "#26a69a",
      downColor: "#ef5350",
      upColor: "#26a69a",
      wickDownColor: "#ef5350",
      wickUpColor: "#26a69a",
    });
    const volumeSeries = mainChart.addSeries(HistogramSeries, {
      color: "#334155",
      priceFormat: { type: "volume" },
      priceScaleId: "",
    });

    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        bottom: 0,
        top: 0.82,
      },
    });

    const lineWidth15 = 1.5 as unknown as LineWidth;
    const indicatorSeries: IndicatorSeriesRefs = {
      bbLower: mainChart.addSeries(LineSeries, {
        color: "#546e7a",
        lineStyle: LineStyle.Dashed,
        lineWidth: 1,
        visible: activeIndicators.BB,
      }),
      bbMiddle: mainChart.addSeries(LineSeries, {
        color: "#78909c",
        lineWidth: 1,
        visible: activeIndicators.BB,
      }),
      bbUpper: mainChart.addSeries(LineSeries, {
        color: "#546e7a",
        lineStyle: LineStyle.Dashed,
        lineWidth: 1,
        visible: activeIndicators.BB,
      }),
      ema9: mainChart.addSeries(LineSeries, {
        color: "#ff9800",
        lineWidth: 1,
        visible: activeIndicators.EMA9,
      }),
      ema21: mainChart.addSeries(LineSeries, {
        color: "#2196f3",
        lineWidth: 1,
        visible: activeIndicators.EMA21,
      }),
      ema50: mainChart.addSeries(LineSeries, {
        color: "#4caf50",
        lineWidth: lineWidth15,
        visible: activeIndicators.EMA50,
      }),
      ema200: mainChart.addSeries(LineSeries, {
        color: "#f44336",
        lineWidth: 2,
        visible: activeIndicators.EMA200,
      }),
      macdHist: macdChart.addSeries(HistogramSeries, {
        base: 0,
        color: "#26a69a",
        visible: activeIndicators.MACD,
      }),
      macdLine: macdChart.addSeries(LineSeries, {
        color: "#26a69a",
        lineWidth: 1,
        visible: activeIndicators.MACD,
      }),
      macdSignal: macdChart.addSeries(LineSeries, {
        color: "#ef5350",
        lineWidth: 1,
        visible: activeIndicators.MACD,
      }),
      rsi: rsiChart.addSeries(LineSeries, {
        color: "#b39ddb",
        lineWidth: 1,
        visible: activeIndicators.RSI,
      }),
    };
    indicatorSeries.rsi.createPriceLine({
      axisLabelVisible: true,
      color: "#6b7280",
      lineStyle: LineStyle.Dashed,
      lineWidth: 1,
      price: 70,
      title: "70",
    });
    indicatorSeries.rsi.createPriceLine({
      axisLabelVisible: true,
      color: "#6b7280",
      lineStyle: LineStyle.Dashed,
      lineWidth: 1,
      price: 30,
      title: "30",
    });
    seriesRefs.current = indicatorSeries;

    const resizeObserver = new ResizeObserver(([entry]) => {
      const width = Math.max(320, Math.floor(entry.contentRect.width));
      mainChart.applyOptions({ height: 500, width });
      rsiChart.applyOptions({ height: 120, width });
      macdChart.applyOptions({ height: 100, width });
    });

    resizeObserver.observe(mainContainer);

    const loadData = async () => {
      try {
        const [rows, indicatorResponse] = await Promise.all([
          invoke<OHLCV[]>("get_ohlcv", { symbol: "NIFTY" }),
          invoke<IndicatorResponse>("get_indicators", {
            indicators: INDICATOR_REQUEST,
            symbol: "NIFTY",
            timeframe: "1d",
          }),
        ]);
        if (!active) {
          return;
        }
        if (indicatorResponse.error) {
          throw new Error(indicatorResponse.error);
        }

        const candles: CandlestickData[] = rows.map((row) => ({
          close: row.close,
          high: row.high,
          low: row.low,
          open: row.open,
          time: row.date,
        }));
        const volumes: HistogramData[] = rows.map((row) => ({
          color: row.close >= row.open ? "#1f9d8a66" : "#d94b5b66",
          time: row.date,
          value: row.volume,
        }));

        candleSeries.setData(candles);
        volumeSeries.setData(volumes);
        setIndicatorData(indicatorSeries, indicatorResponse);
        chartRefs.current.forEach((chart) => chart.timeScale().fitContent());
        setMessage(rows.length > 0 ? "" : "No historical candles yet");
      } catch (error) {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      }
    };

    loadData();

    return () => {
      active = false;
      resizeObserver.disconnect();
      chartRefs.current.forEach((chart) => chart.remove());
      chartRefs.current = [];
      seriesRefs.current = null;
    };
  }, []);

  useEffect(() => {
    const series = seriesRefs.current;
    if (!series) {
      return;
    }

    series.ema9.applyOptions({ visible: activeIndicators.EMA9 });
    series.ema21.applyOptions({ visible: activeIndicators.EMA21 });
    series.ema50.applyOptions({ visible: activeIndicators.EMA50 });
    series.ema200.applyOptions({ visible: activeIndicators.EMA200 });
    series.rsi.applyOptions({ visible: activeIndicators.RSI });
    series.macdLine.applyOptions({ visible: activeIndicators.MACD });
    series.macdSignal.applyOptions({ visible: activeIndicators.MACD });
    series.macdHist.applyOptions({ visible: activeIndicators.MACD });
    series.bbUpper.applyOptions({ visible: activeIndicators.BB });
    series.bbMiddle.applyOptions({ visible: activeIndicators.BB });
    series.bbLower.applyOptions({ visible: activeIndicators.BB });
  }, [activeIndicators]);

  const toggleIndicator = (key: IndicatorKey) => {
    setActiveIndicators((current) => ({
      ...current,
      [key]: !current[key],
    }));
  };

  return (
    <section
      style={{
        minWidth: 0,
        width: "100%",
      }}
    >
      <div
        aria-label="Indicator toggles"
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 8,
          marginBottom: 10,
        }}
      >
        {INDICATOR_BUTTONS.map((item) => {
          const isActive = activeIndicators[item.key];
          return (
            <button
              key={item.key}
              onClick={() => toggleIndicator(item.key)}
              style={{
                background: isActive ? "#1f2937" : "#101010",
                border: `1px solid ${isActive ? "#4b5563" : "#262626"}`,
                borderRadius: 6,
                color: isActive ? "#ffffff" : "#8a94a6",
                cursor: "pointer",
                fontFamily: "monospace",
                fontSize: 12,
                height: 30,
                padding: "0 10px",
              }}
              type="button"
            >
              {item.label}
            </button>
          );
        })}
      </div>
      <div
        ref={mainContainerRef}
        style={{
          height: 500,
          position: "relative",
          width: "100%",
        }}
      />
      <div
        ref={rsiContainerRef}
        style={{
          height: activeIndicators.RSI ? 120 : 0,
          marginTop: activeIndicators.RSI ? 8 : 0,
          overflow: "hidden",
          position: "relative",
          transition: "height 120ms ease",
          width: "100%",
        }}
      />
      <div
        ref={macdContainerRef}
        style={{
          height: activeIndicators.MACD ? 100 : 0,
          marginTop: activeIndicators.MACD ? 8 : 0,
          overflow: "hidden",
          position: "relative",
          transition: "height 120ms ease",
          width: "100%",
        }}
      />
      {message ? (
        <div
          style={{
            color: "#8a94a6",
            fontSize: 12,
            marginTop: 8,
            minHeight: 16,
          }}
        >
          {message}
        </div>
      ) : null}
    </section>
  );
}

function createPanelChart(
  container: HTMLDivElement,
  height: number,
  background: string,
) {
  return createChart(container, {
    autoSize: false,
    height,
    layout: {
      background: { type: ColorType.Solid, color: background },
      textColor: "#d1d4dc",
    },
    grid: {
      horzLines: { color: "#1f2933" },
      vertLines: { color: "#1f2933" },
    },
    rightPriceScale: {
      borderColor: "#2a2e39",
    },
    timeScale: {
      borderColor: "#2a2e39",
      timeVisible: false,
    },
    width: container.clientWidth,
  });
}

function setIndicatorData(
  series: IndicatorSeriesRefs,
  response: IndicatorResponse,
) {
  const { indicators, timestamps } = response;
  series.ema9.setData(toLineData(timestamps, indicators.EMA_9));
  series.ema21.setData(toLineData(timestamps, indicators.EMA_21));
  series.ema50.setData(toLineData(timestamps, indicators.EMA_50));
  series.ema200.setData(toLineData(timestamps, indicators.EMA_200));
  series.rsi.setData(toLineData(timestamps, indicators.RSI_14));
  series.macdLine.setData(toLineData(timestamps, indicators.MACD_line));
  series.macdSignal.setData(toLineData(timestamps, indicators.MACD_signal));
  series.macdHist.setData(toHistogramData(timestamps, indicators.MACD_hist));
  series.bbUpper.setData(toLineData(timestamps, indicators.BB_upper));
  series.bbMiddle.setData(toLineData(timestamps, indicators.BB_middle));
  series.bbLower.setData(toLineData(timestamps, indicators.BB_lower));
}

function toLineData(
  timestamps: string[],
  values: IndicatorValue[] | undefined,
): LineData[] {
  if (!values) {
    return [];
  }

  return timestamps.flatMap((time, index) => {
    const value = values[index];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return [];
    }

    return [{ time, value }];
  });
}

function toHistogramData(
  timestamps: string[],
  values: IndicatorValue[] | undefined,
): HistogramData[] {
  if (!values) {
    return [];
  }

  return timestamps.flatMap((time, index) => {
    const value = values[index];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return [];
    }

    return [
      {
        color: value >= 0 ? "#26a69a" : "#ef5350",
        time,
        value,
      },
    ];
  });
}
