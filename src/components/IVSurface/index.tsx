import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import Plotly from "plotly.js-dist-min";
import type { Config, Data, Layout } from "plotly.js";

type NullableNumber = number | null;

type IVSurfaceData = {
  error?: string;
  surface: {
    strikes: number[];
    expiries: string[];
    dtes: number[];
    call_iv_grid: NullableNumber[][];
    put_iv_grid: NullableNumber[][];
  };
  skew: {
    expiry: string;
    strikes: number[];
    call_ivs: NullableNumber[];
    put_ivs: NullableNumber[];
    atm_strike?: NullableNumber;
  };
  term_structure: {
    expiries: string[];
    dtes: number[];
    atm_call_iv: NullableNumber[];
    atm_put_iv: NullableNumber[];
  };
};

const plotConfig: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
};

const baseLayout: Partial<Layout> = {
  autosize: true,
  font: { color: "#d1d4dc", family: "monospace" },
  margin: { b: 48, l: 56, r: 22, t: 18 },
  paper_bgcolor: "#0d0d0d",
  plot_bgcolor: "#0d0d0d",
};

export function IVSurface() {
  const [surfaceData, setSurfaceData] = useState<IVSurfaceData | null>(null);
  const [message, setMessage] = useState("Building IV surface...");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    const loadSurface = async () => {
      try {
        const result = await invoke<IVSurfaceData>("get_iv_surface", {
          symbol: "NIFTY",
        });
        if (!active) {
          return;
        }

        if (result.error) {
          setSurfaceData(null);
          setMessage("Insufficient data for surface");
          return;
        }

        setSurfaceData(result);
        setMessage("");
      } catch (error) {
        if (active) {
          setSurfaceData(null);
          setMessage(error instanceof Error ? error.message : String(error));
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    loadSurface();
    const timer = window.setInterval(loadSurface, 30_000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const hasSkew = (surfaceData?.skew.strikes.length ?? 0) > 0;
  const hasSurface =
    (surfaceData?.surface.expiries.length ?? 0) >= 2 &&
    (surfaceData?.surface.strikes.length ?? 0) >= 3;
  const hasTermStructure =
    (surfaceData?.term_structure.expiries.length ?? 0) > 0;

  const surfaceTraces = useMemo<Data[]>(() => {
    if (!surfaceData) {
      return [];
    }

    return [
      {
        colorscale: "RdYlGn",
        name: "Call IV Surface",
        reversescale: true,
        showscale: true,
        type: "surface",
        x: surfaceData.surface.strikes,
        y: surfaceData.surface.dtes,
        z: transposeGridAsPercent(surfaceData.surface.call_iv_grid),
      } as Data,
    ];
  }, [surfaceData]);

  const surfaceLayout = useMemo<Partial<Layout>>(
    () => ({
      ...baseLayout,
      height: 400,
      scene: {
        bgcolor: "#0d0d0d",
        xaxis: axis3d("Strike"),
        yaxis: axis3d("DTE"),
        zaxis: axis3d("IV%"),
      },
    }),
    [],
  );

  const skewTraces = useMemo<Data[]>(() => {
    if (!surfaceData) {
      return [];
    }

    return [
      {
        line: { color: "#26a69a", width: 2 },
        mode: "lines+markers",
        name: "Call IV",
        type: "scatter",
        x: surfaceData.skew.strikes,
        y: toPercentArray(surfaceData.skew.call_ivs),
      } as Data,
      {
        line: { color: "#ef5350", width: 2 },
        mode: "lines+markers",
        name: "Put IV",
        type: "scatter",
        x: surfaceData.skew.strikes,
        y: toPercentArray(surfaceData.skew.put_ivs),
      } as Data,
    ];
  }, [surfaceData]);

  const skewLayout = useMemo<Partial<Layout>>(() => {
    const atmStrike = surfaceData?.skew.atm_strike;
    return {
      ...baseLayout,
      height: 320,
      shapes:
        atmStrike === null || atmStrike === undefined
          ? []
          : [
              {
                line: { color: "#f59e0b", dash: "dot", width: 1 },
                type: "line",
                x0: atmStrike,
                x1: atmStrike,
                y0: 0,
                y1: 1,
                yref: "paper",
              },
            ],
      xaxis: axis2d("Strike"),
      yaxis: axis2d("IV%"),
    };
  }, [surfaceData]);

  const termStructureTraces = useMemo<Data[]>(() => {
    if (!surfaceData) {
      return [];
    }

    return [
      {
        line: { color: "#26a69a", width: 2 },
        mode: "lines+markers",
        name: "ATM Call IV",
        type: "scatter",
        x: surfaceData.term_structure.dtes,
        y: toPercentArray(surfaceData.term_structure.atm_call_iv),
      } as Data,
      {
        line: { color: "#ef5350", width: 2 },
        mode: "lines+markers",
        name: "ATM Put IV",
        type: "scatter",
        x: surfaceData.term_structure.dtes,
        y: toPercentArray(surfaceData.term_structure.atm_put_iv),
      } as Data,
    ];
  }, [surfaceData]);

  const termStructureLayout = useMemo<Partial<Layout>>(
    () => ({
      ...baseLayout,
      height: 320,
      xaxis: axis2d("DTE"),
      yaxis: axis2d("IV%"),
    }),
    [],
  );

  if (loading && !surfaceData) {
    return <StatusBlock message="Building IV surface..." />;
  }

  if (!surfaceData) {
    return <StatusBlock message={message || "Insufficient data for surface"} />;
  }

  return (
    <section
      style={{
        color: "#d1d4dc",
        display: "flex",
        flexDirection: "column",
        gap: 18,
        minWidth: 0,
        width: "100%",
      }}
    >
      <ChartBlock title="IV Surface">
        {hasSurface ? (
          <PlotlyChart
            data={surfaceTraces}
            layout={surfaceLayout}
            style={{ height: 400 }}
          />
        ) : (
          <InlineMessage message="Insufficient data for surface" />
        )}
      </ChartBlock>

      <ChartBlock title={`IV Skew - ${surfaceData.skew.expiry || "nearest expiry"}`}>
        {hasSkew ? (
          <PlotlyChart
            data={skewTraces}
            layout={skewLayout}
            style={{ height: 320 }}
          />
        ) : (
          <InlineMessage message="No skew data yet" />
        )}
      </ChartBlock>

      <ChartBlock title="Term Structure">
        {hasTermStructure ? (
          <PlotlyChart
            data={termStructureTraces}
            layout={termStructureLayout}
            style={{ height: 320 }}
          />
        ) : (
          <InlineMessage message="No term structure data yet" />
        )}
      </ChartBlock>
    </section>
  );
}

function PlotlyChart({
  data,
  layout,
  style,
}: {
  data: Data[];
  layout: Partial<Layout>;
  style: React.CSSProperties;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    Plotly.react(container, data, layout, plotConfig);

    const resizeObserver = new ResizeObserver(() => {
      Plotly.Plots.resize(container);
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      Plotly.purge(container);
    };
  }, [data, layout]);

  return (
    <div
      ref={containerRef}
      style={{
        minWidth: 0,
        overflow: "hidden",
        width: "100%",
        ...style,
      }}
    />
  );
}

function ChartBlock({
  children,
  title,
}: {
  children: React.ReactNode;
  title: string;
}) {
  return (
    <div
      style={{
        background: "#0d0d0d",
        border: "1px solid #2a2e39",
        minWidth: 0,
        padding: "12px 12px 8px",
        width: "100%",
      }}
    >
      <h2
        style={{
          color: "#f8fafc",
          fontSize: 15,
          fontWeight: 700,
          letterSpacing: 0,
          margin: "0 0 8px",
        }}
      >
        {title}
      </h2>
      {children}
    </div>
  );
}

function StatusBlock({ message }: { message: string }) {
  return (
    <section
      style={{
        background: "#0d0d0d",
        border: "1px solid #2a2e39",
        color: "#8a94a6",
        fontSize: 12,
        padding: 16,
        width: "100%",
      }}
    >
      {message}
    </section>
  );
}

function InlineMessage({ message }: { message: string }) {
  return (
    <div
      style={{
        alignItems: "center",
        color: "#8a94a6",
        display: "flex",
        fontSize: 12,
        height: 120,
      }}
    >
      {message}
    </div>
  );
}

function axis2d(title: string) {
  return {
    gridcolor: "#1f2933",
    linecolor: "#2a2e39",
    tickfont: { color: "#d1d4dc" },
    title: { font: { color: "#d1d4dc" }, text: title },
    zerolinecolor: "#2a2e39",
  };
}

function axis3d(title: string) {
  return {
    backgroundcolor: "#0d0d0d",
    gridcolor: "#1f2933",
    linecolor: "#2a2e39",
    tickfont: { color: "#d1d4dc" },
    title: { font: { color: "#d1d4dc" }, text: title },
    zerolinecolor: "#2a2e39",
  };
}

function transposeGridAsPercent(grid: NullableNumber[][]) {
  const columnCount = grid.reduce(
    (largest, row) => Math.max(largest, row.length),
    0,
  );

  return Array.from({ length: columnCount }, (_, columnIndex) =>
    grid.map((row) => toPercent(row[columnIndex] ?? null)),
  );
}

function toPercentArray(values: NullableNumber[]) {
  return values.map(toPercent);
}

function toPercent(value: NullableNumber) {
  if (value === null || !Number.isFinite(value)) {
    return null;
  }
  return value * 100;
}
