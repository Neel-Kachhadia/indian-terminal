import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { downloadDir, join } from "@tauri-apps/api/path";
import { save } from "@tauri-apps/plugin-dialog";
import { writeFile } from "@tauri-apps/plugin-fs";
import { open } from "@tauri-apps/plugin-shell";

let pdfLibraryPromise;

function loadPdfLibrary() {
  if (!pdfLibraryPromise) {
    const moduleUrl = new URL("/pdf.min.mjs", window.location.href).href;
    pdfLibraryPromise = import(/* @vite-ignore */ moduleUrl).then((library) => {
      library.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
      return library;
    });
  }
  return pdfLibraryPromise;
}

export function ContentViewer({ item, onClose }) {
  const contentType = detectType(item);
  const externalUrl = item?.pdf_url || item?.url || "";
  const title = item?.title ?? "Equity content";
  const [pdfBytes, setPdfBytes] = useState(null);
  const [pdfError, setPdfError] = useState("");

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    let active = true;
    setPdfBytes(null);
    setPdfError("");
    if (contentType !== "pdf" || !item?.pdf_url) {
      return () => {
        active = false;
      };
    }
    invoke("fetch_equity_pdf", { url: item.pdf_url })
      .then((encoded) => {
        if (active) {
          setPdfBytes(decodeBase64(encoded));
        }
      })
      .catch((error) => {
        if (active) {
          setPdfError(error instanceof Error ? error.message : String(error));
        }
      });
    return () => {
      active = false;
    };
  }, [contentType, item?.pdf_url]);

  const savePdf = async () => {
    if (!pdfBytes) {
      return;
    }
    const defaultPath = await join(await downloadDir(), `${safeFileName(title)}.pdf`);
    const filePath = await save({
      defaultPath,
      filters: [{ name: "PDF", extensions: ["pdf"] }],
    });
    if (!filePath) {
      return;
    }
    await writeFile(filePath, pdfBytes);
  };

  return (
    <div style={overlayStyle}>
      <div style={headerStyle}>
        <button onClick={onClose} style={iconButtonStyle} title="Back" type="button">
          ←
        </button>
        <div style={titleStyle}>{truncate(title, 60)}</div>
        {contentType === "pdf" ? (
          <button disabled={!pdfBytes} onClick={savePdf} style={actionButtonStyle} type="button">
            ⬇ Save
          </button>
        ) : null}
        <button
          disabled={!externalUrl}
          onClick={() => externalUrl && open(externalUrl)}
          style={actionButtonStyle}
          type="button"
        >
          ↗ Browser
        </button>
        <button onClick={onClose} style={iconButtonStyle} title="Close" type="button">
          ×
        </button>
      </div>
      <div style={contentStyle}>
        {contentType === "pdf" ? <PdfViewer data={pdfBytes} error={pdfError} /> : null}
        {contentType === "text" ? <TextViewer item={item} /> : null}
        {contentType === "web" ? <WebViewer item={item} /> : null}
      </div>
    </div>
  );
}

function PdfViewer({ data, error }) {
  const canvasRef = useRef(null);
  const documentRef = useRef(null);
  const renderTaskRef = useRef(null);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [zoom, setZoom] = useState(1);
  const [message, setMessage] = useState("Loading PDF...");

  useEffect(() => {
    let active = true;
    setMessage(error || "Loading PDF...");
    setPage(1);
    setTotal(0);
    documentRef.current = null;

    if (!data) {
      return () => {
        active = false;
      };
    }

    loadPdfLibrary()
      .then((pdfjsLib) => pdfjsLib.getDocument({ data: data.slice() }).promise)
      .then((pdf) => {
        if (!active) {
          return;
        }
        documentRef.current = pdf;
        setTotal(pdf.numPages);
        setMessage("");
      })
      .catch((error) => {
        if (active) {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      });

    return () => {
      active = false;
      renderTaskRef.current?.cancel?.();
      documentRef.current?.destroy?.();
    };
  }, [data, error]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const pdf = documentRef.current;
    if (!canvas || !pdf) {
      return;
    }

    let active = true;
    renderTaskRef.current?.cancel?.();
    pdf.getPage(page)
      .then((pdfPage) => {
        if (!active) {
          return;
        }
        const viewport = pdfPage.getViewport({ scale: zoom });
        const context = canvas.getContext("2d");
        canvas.height = viewport.height;
        canvas.width = viewport.width;
        renderTaskRef.current = pdfPage.render({ canvasContext: context, viewport });
        return renderTaskRef.current.promise;
      })
      .catch((error) => {
        if (String(error?.name) !== "RenderingCancelledException") {
          setMessage(error instanceof Error ? error.message : String(error));
        }
      });

    return () => {
      active = false;
      renderTaskRef.current?.cancel?.();
    };
  }, [page, total, zoom]);

  return (
    <div style={pdfShellStyle}>
      <div style={canvasWrapStyle}>
        <canvas ref={canvasRef} style={canvasStyle} />
        {message ? <div style={pdfMessageStyle}>{message}</div> : null}
      </div>
      <div style={pdfControlsStyle}>
        <button disabled={page <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))} style={controlButtonStyle} type="button">
          ← Prev
        </button>
        <span style={pageStyle}>Page {page} of {total || "--"}</span>
        <button disabled={!total || page >= total} onClick={() => setPage((value) => Math.min(total, value + 1))} style={controlButtonStyle} type="button">
          Next →
        </button>
        <button onClick={() => setZoom((value) => Math.max(0.6, value - 0.1))} style={controlButtonStyle} type="button">
          −
        </button>
        <span style={pageStyle}>{Math.round(zoom * 100)}%</span>
        <button onClick={() => setZoom((value) => Math.min(2.2, value + 0.1))} style={controlButtonStyle} type="button">
          +
        </button>
      </div>
    </div>
  );
}

function TextViewer({ item }) {
  return (
    <article style={textPanelStyle}>
      <h2 style={textTitleStyle}>{item.title}</h2>
      <div style={textMetaStyle}>{formatDate(item.date)}</div>
      <span style={sourceBadgeStyle}>{item.source ?? "NSE"}</span>
      <p style={bodyTextStyle}>{item.body}</p>
    </article>
  );
}

function WebViewer({ item }) {
  const url = item?.url || item?.pdf_url || "";
  return <WebFallback url={url} />;
}

function WebFallback({ url }) {
  return (
    <div style={fallbackStyle}>
      <div style={fallbackTitleStyle}>This article requires opening in browser</div>
      <button disabled={!url} onClick={() => url && open(url)} style={fallbackButtonStyle} type="button">
        Open in Browser
      </button>
      <input readOnly style={urlStyle} value={url || "No URL available"} />
    </div>
  );
}

function detectType(item) {
  const pdfUrl = item?.pdf_url ?? "";
  if (/\.pdf($|\?)/i.test(pdfUrl)) {
    return "pdf";
  }
  if (item?.type === "announcement" && item?.body) {
    return "text";
  }
  return "web";
}

function truncate(value, limit) {
  const text = String(value ?? "");
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

function safeFileName(value) {
  const cleaned = String(value ?? "document").replace(/[^a-z0-9._-]+/gi, "_");
  return cleaned.slice(0, 80) || "document";
}

function decodeBase64(value) {
  const binary = window.atob(String(value));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function formatDate(dateValue) {
  const timestamp = Date.parse(dateValue ?? "");
  if (!Number.isFinite(timestamp)) {
    return "Date unavailable";
  }
  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(timestamp));
}

const overlayStyle = {
  background: "#000000ee",
  display: "grid",
  gridTemplateRows: "48px minmax(0, 1fr)",
  inset: 0,
  position: "fixed",
  zIndex: 9999,
};

const headerStyle = {
  alignItems: "center",
  background: "#111111",
  borderBottom: "1px solid #262626",
  display: "grid",
  gap: 8,
  gridTemplateColumns: "38px minmax(0, 1fr) auto auto 38px",
  padding: "0 10px",
};

const titleStyle = {
  color: "#ffffff",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 13,
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap",
};

const iconButtonStyle = {
  background: "#181818",
  border: "1px solid #333333",
  color: "#ffffff",
  cursor: "pointer",
  fontSize: 16,
  height: 30,
};

const actionButtonStyle = {
  background: "#181818",
  border: "1px solid #333333",
  color: "#ffffff",
  cursor: "pointer",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  height: 30,
  padding: "0 10px",
};

const contentStyle = {
  minHeight: 0,
  overflow: "hidden",
};

const pdfShellStyle = {
  display: "grid",
  gridTemplateRows: "minmax(0, 1fr) 44px",
  height: "100%",
};

const canvasWrapStyle = {
  alignItems: "start",
  display: "flex",
  justifyContent: "center",
  minHeight: 0,
  overflow: "auto",
  padding: 18,
  position: "relative",
};

const canvasStyle = {
  background: "#ffffff",
  boxShadow: "0 12px 40px rgba(0,0,0,0.45)",
  maxWidth: "100%",
};

const pdfMessageStyle = {
  color: "#d4aa00",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 13,
  left: 24,
  position: "absolute",
  top: 24,
};

const pdfControlsStyle = {
  alignItems: "center",
  background: "#111111",
  borderTop: "1px solid #262626",
  display: "flex",
  gap: 10,
  justifyContent: "center",
};

const controlButtonStyle = {
  background: "#181818",
  border: "1px solid #333333",
  color: "#ffffff",
  cursor: "pointer",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  height: 28,
  padding: "0 9px",
};

const pageStyle = {
  color: "#cccccc",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
};

const textPanelStyle = {
  boxSizing: "border-box",
  color: "#cccccc",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  height: "100%",
  lineHeight: 1.6,
  overflow: "auto",
  padding: 24,
};

const textTitleStyle = {
  color: "#ffffff",
  fontSize: 20,
  lineHeight: 1.3,
  margin: 0,
};

const textMetaStyle = {
  color: "#888888",
  fontSize: 12,
  marginTop: 8,
};

const sourceBadgeStyle = {
  background: "#1a3a5c",
  color: "#4d9de0",
  display: "inline-block",
  fontSize: 11,
  marginTop: 14,
  padding: "4px 7px",
};

const bodyTextStyle = {
  color: "#cccccc",
  fontSize: 14,
  marginTop: 18,
  maxWidth: 980,
  whiteSpace: "pre-wrap",
};

const fallbackStyle = {
  alignItems: "center",
  color: "#cccccc",
  display: "grid",
  gap: 14,
  height: "100%",
  justifyItems: "center",
  padding: 24,
};

const fallbackTitleStyle = {
  color: "#ffffff",
  fontFamily: "Inter, system-ui, sans-serif",
  fontSize: 16,
};

const fallbackButtonStyle = {
  background: "#00d4aa",
  border: "1px solid #00d4aa",
  color: "#00130f",
  cursor: "pointer",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 12,
  fontWeight: 700,
  height: 36,
  padding: "0 16px",
};

const urlStyle = {
  background: "#111111",
  border: "1px solid #333333",
  color: "#cccccc",
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  fontSize: 11,
  maxWidth: 760,
  padding: "8px 10px",
  width: "80%",
};
