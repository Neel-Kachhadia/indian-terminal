import type Plotly from "plotly.js";

type PlotlyApi = typeof Plotly;

declare global {
  interface Window {
    Plotly?: PlotlyApi;
  }
}

let plotlyPromise: Promise<PlotlyApi> | null = null;

export function loadPlotly(): Promise<PlotlyApi> {
  if (window.Plotly) {
    return Promise.resolve(window.Plotly);
  }
  if (plotlyPromise) {
    return plotlyPromise;
  }
  const pending = new Promise<PlotlyApi>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/plotly.min.js";
    script.async = true;
    script.dataset.bharatPlotly = "true";
    script.addEventListener("load", () => {
      if (window.Plotly) {
        resolve(window.Plotly);
      } else {
        reject(new Error("Local Plotly asset loaded without API"));
      }
    });
    script.addEventListener("error", () => reject(new Error("Local Plotly asset failed to load")));
    document.head.appendChild(script);
  }).catch((error): never => {
    plotlyPromise = null;
    throw error;
  });
  plotlyPromise = pending;
  return pending;
}
