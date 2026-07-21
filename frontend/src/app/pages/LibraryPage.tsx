import { useState, useEffect, useCallback } from "react";
import { api } from "../services/api";
import { useAuth } from "../context/AuthContext";
import type { OutputItem } from "../services/api";

type OutputType = OutputItem["type"];

const TYPE_LABELS: Record<OutputType, string> = {
  analyze:  "Analyze",
  inspect:  "Inspect",
  snapshot: "Snapshot",
  autoscan: "Auto-Scan",
  record:   "Record",
  flag:     "Flagged",
};

const TYPE_COLORS: Record<OutputType, string> = {
  analyze:  "badge-blue",
  inspect:  "badge-amber",
  snapshot: "badge-green",
  autoscan: "badge-purple",
  record:   "badge-red",
  flag:     "badge-orange",
};

// Only these types offer a download button
const DOWNLOADABLE: Set<OutputType> = new Set(["snapshot", "record", "flag"]);

// These types show the image + Q&A side panel in the preview
const HAS_QA: Set<OutputType> = new Set(["analyze", "inspect", "autoscan"]);

function formatDate(ts: string): string {
  return new Date(ts).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function formatTime(ts: string): string {
  return new Date(ts).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function dateKey(ts: string): string {
  return ts.slice(0, 10);
}

// ── Grouping helpers ──────────────────────────────────────────────────────────

interface DateGroup {
  dk: string;
  dateLabel: string;
  byType: Partial<Record<OutputType, OutputItem[]>>;
}

interface UserGroup {
  userId: number;
  username: string;
  dates: DateGroup[];
}

function groupByDate(items: OutputItem[]): DateGroup[] {
  const map = new Map<string, DateGroup>();
  for (const item of items) {
    const dk = dateKey(item.timestamp);
    if (!map.has(dk)) map.set(dk, { dk, dateLabel: formatDate(item.timestamp), byType: {} });
    const g = map.get(dk)!;
    if (!g.byType[item.type]) g.byType[item.type] = [];
    g.byType[item.type]!.push(item);
  }
  return Array.from(map.values()).sort((a, b) => b.dk.localeCompare(a.dk));
}

function groupByUser(items: OutputItem[]): UserGroup[] {
  const map = new Map<number, { username: string; items: OutputItem[] }>();
  for (const item of items) {
    if (!map.has(item.user_id)) map.set(item.user_id, { username: item.username ?? String(item.user_id), items: [] });
    map.get(item.user_id)!.items.push(item);
  }
  return Array.from(map.entries())
    .map(([userId, { username, items }]) => ({ userId, username, dates: groupByDate(items) }))
    .sort((a, b) => a.username.localeCompare(b.username));
}

// ── Preview modal ─────────────────────────────────────────────────────────────

function PreviewModal({ item, onClose }: { item: OutputItem; onClose: () => void }) {
  const viewUrl = api.libraryViewUrl(item.id);
  const dlUrl   = api.libraryDownloadUrl(item.id);
  const isVideo = item.type === "record";
  const hasQA   = HAS_QA.has(item.type);
  const canDownload = DOWNLOADABLE.has(item.type);

  // Escape closes the preview. On tablets the visible close button is hidden
  // (tapping the backdrop is the natural gesture there), so this keeps a
  // non-pointer way out for keyboard and assistive-technology users.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="preview-overlay" onClick={onClose}>
      <div
        className={`preview-modal ${hasQA ? "preview-modal-split" : "preview-modal-single"}`}
        onClick={e => e.stopPropagation()}
      >
        {/* Images can take an overlaid close button. Video cannot: Safari draws
            its own controls over the video surface and we cannot move them, so
            the video layout gets a dedicated header bar instead. */}
        {!isVideo && (
          <button className="preview-close" onClick={onClose} aria-label="Close preview">
            ✕
          </button>
        )}

        {isVideo ? (
          /* ── Video recording ─────────────────────────────────── */
          <div className="preview-video-box">
            <div className="preview-video-header">
              <button
                className="preview-close preview-close-static"
                onClick={onClose}
                aria-label="Close preview"
              >
                ✕
              </button>
            </div>
            <video
              className="preview-video-player"
              src={viewUrl}
              controls
              preload="metadata"
            />
            <div className="preview-video-footer">
              <span className="preview-ts">{formatDate(item.timestamp)} · {formatTime(item.timestamp)}</span>
              <a className="btn-preview-download" href={dlUrl} download>↓ Download MP4</a>
            </div>
          </div>
        ) : hasQA ? (
          /* ── Analyze / Inspect / Auto-Scan ── image + Q&A panel ─ */
          <>
            <div className="preview-img-side">
              <img className="preview-img" src={viewUrl} alt={TYPE_LABELS[item.type]} />
            </div>
            <div className="preview-qa-side">
              <div className="preview-qa-header">
                <span className={`lib-badge ${TYPE_COLORS[item.type]}`}>{TYPE_LABELS[item.type]}</span>
                <span className="preview-ts">{formatDate(item.timestamp)} · {formatTime(item.timestamp)}</span>
              </div>
              {item.prompt && (
                <div className="preview-qa-block">
                  <span className="preview-qa-label preview-q-label">Q</span>
                  <p className="preview-qa-text">{item.prompt}</p>
                </div>
              )}
              {item.response && (
                <div className="preview-qa-block">
                  <span className="preview-qa-label preview-a-label">A</span>
                  <p className="preview-qa-text">{item.response}</p>
                </div>
              )}
              {item.tokens != null && item.elapsed_s != null && (
                <p className="preview-meta">{item.elapsed_s}s · {item.tokens} tokens</p>
              )}
            </div>
          </>
        ) : (
          /* ── Snapshot / Flag ── full image ─────────────────────── */
          <div className="preview-single-box">
            <img className="preview-img-full" src={viewUrl} alt={TYPE_LABELS[item.type]} />
            <div className="preview-single-footer">
              <span className={`lib-badge ${TYPE_COLORS[item.type]}`}>{TYPE_LABELS[item.type]}</span>
              <span className="preview-ts">{formatDate(item.timestamp)} · {formatTime(item.timestamp)}</span>
              {item.response && <p className="preview-flag-response">{item.response}</p>}
              {canDownload && (
                <a className="btn-preview-download" href={dlUrl} download>↓ Download</a>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Media card ────────────────────────────────────────────────────────────────

function MediaCard({ item, onPreview }: { item: OutputItem; onPreview: (item: OutputItem) => void }) {
  const isVideo    = item.type === "record";
  const canDownload = DOWNLOADABLE.has(item.type);
  const viewUrl    = api.libraryViewUrl(item.id);
  const dlUrl      = api.libraryDownloadUrl(item.id);

  return (
    <div className="lib-card" onClick={() => onPreview(item)}>
      <div className="lib-card-media">
        {isVideo ? (
          <video
            className="lib-card-video"
            src={viewUrl}
            preload="metadata"
            muted
          />
        ) : (
          <img
            className="lib-card-img"
            src={viewUrl}
            alt={TYPE_LABELS[item.type]}
            loading="lazy"
            onError={e => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
          />
        )}
      </div>
      <div className="lib-card-body">
        <div className="lib-card-time">{formatTime(item.timestamp)}</div>
        {item.response && <p className="lib-card-response">{item.response}</p>}
        {item.tokens != null && item.elapsed_s != null && (
          <div className="lib-card-meta">{item.elapsed_s}s · {item.tokens} tokens</div>
        )}
        {canDownload && (
          <a
            className="btn-lib-download"
            href={dlUrl}
            download
            onClick={e => e.stopPropagation()}
          >
            ↓ Download
          </a>
        )}
      </div>
    </div>
  );
}

// ── Sidebar sub-items (type list under a date) ────────────────────────────────

function TypeList({
  date,
  selected,
  onSelect,
}: {
  date: DateGroup;
  selected: { dk: string; type: OutputType; userId?: number } | null;
  onSelect: (dk: string, type: OutputType) => void;
}) {
  const types = Object.keys(date.byType) as OutputType[];
  return (
    <div className="lib-type-list">
      {types.map(type => {
        const count    = date.byType[type]?.length ?? 0;
        const isActive = selected?.dk === date.dk && selected?.type === type;
        return (
          <button
            key={type}
            className={`lib-type-btn ${isActive ? "lib-type-btn-active" : ""}`}
            onClick={() => onSelect(date.dk, type)}
          >
            <span className={`lib-type-dot ${TYPE_COLORS[type]}`} />
            <span className="lib-type-name">{TYPE_LABELS[type]}</span>
            <span className="lib-type-count">{count}</span>
          </button>
        );
      })}
    </div>
  );
}

// ── Date group (collapsible) ──────────────────────────────────────────────────

function DateGroupBlock({
  date,
  selected,
  expanded,
  onToggle,
  onSelect,
}: {
  date: DateGroup;
  selected: { dk: string; type: OutputType; userId?: number } | null;
  expanded: boolean;
  onToggle: () => void;
  onSelect: (dk: string, type: OutputType) => void;
}) {
  return (
    <div className="lib-date-group">
      <button
        className={`lib-date-btn ${expanded ? "lib-date-btn-open" : ""}`}
        onClick={onToggle}
      >
        <span className="lib-date-chevron">{expanded ? "▾" : "▸"}</span>
        <span className="lib-date-label">{date.dateLabel}</span>
      </button>
      {expanded && (
        <TypeList date={date} selected={selected} onSelect={onSelect} />
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

interface Selection {
  userId?: number;
  dk: string;
  type: OutputType;
}

export function LibraryPage() {
  const { user } = useAuth();
  const isAdmin  = user?.is_admin ?? false;

  const [outputs,       setOutputs]       = useState<OutputItem[]>([]);
  const [loading,       setLoading]       = useState(true);
  const [error,         setError]         = useState<string | null>(null);
  const [expandedDates, setExpandedDates] = useState<Set<string>>(new Set());
  const [expandedUsers, setExpandedUsers] = useState<Set<number>>(new Set());
  const [selected,      setSelected]      = useState<Selection | null>(null);
  const [preview,       setPreview]       = useState<OutputItem | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    const fetch = isAdmin ? api.libraryAdminOutputs() : api.libraryOutputs();
    fetch
      .then(r => { setOutputs(r.outputs); setLoading(false); })
      .catch(e => { setError(e instanceof Error ? e.message : "Failed to load library."); setLoading(false); });
  }, [isAdmin]);

  useEffect(() => { load(); }, [load]);

  // Auto-expand and auto-select on first load
  useEffect(() => {
    if (outputs.length === 0 || selected) return;

    if (isAdmin) {
      const groups = groupByUser(outputs);
      if (!groups.length) return;
      const firstUser = groups[0];
      setExpandedUsers(new Set([firstUser.userId]));
      if (firstUser.dates.length) {
        const firstDate = firstUser.dates[0];
        setExpandedDates(new Set([firstDate.dk]));
        const firstType = (Object.keys(firstDate.byType) as OutputType[])[0];
        if (firstType) setSelected({ userId: firstUser.userId, dk: firstDate.dk, type: firstType });
      }
    } else {
      const dates = groupByDate(outputs);
      if (!dates.length) return;
      const firstDate = dates[0];
      setExpandedDates(new Set([firstDate.dk]));
      const firstType = (Object.keys(firstDate.byType) as OutputType[])[0];
      if (firstType) setSelected({ dk: firstDate.dk, type: firstType });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [outputs]);

  function toggleDate(dk: string) {
    setExpandedDates(prev => {
      const next = new Set(prev);
      next.has(dk) ? next.delete(dk) : next.add(dk);
      return next;
    });
  }

  function toggleUser(userId: number) {
    setExpandedUsers(prev => {
      const next = new Set(prev);
      next.has(userId) ? next.delete(userId) : next.add(userId);
      return next;
    });
  }

  function selectType(dk: string, type: OutputType, userId?: number) {
    setSelected({ dk, type, userId });
    setExpandedDates(prev => new Set([...prev, dk]));
    if (userId != null) setExpandedUsers(prev => new Set([...prev, userId]));
  }

  // Derive current items and groups
  const userGroups  = isAdmin ? groupByUser(outputs) : [];
  const dateFlatGroups = isAdmin ? [] : groupByDate(outputs);

  let currentItems: OutputItem[] = [];
  let currentDateLabel = "";

  if (selected) {
    if (isAdmin) {
      const ug = userGroups.find(g => g.userId === selected.userId);
      const dg = ug?.dates.find(d => d.dk === selected.dk);
      currentItems      = dg?.byType[selected.type] ?? [];
      currentDateLabel  = dg?.dateLabel ?? "";
    } else {
      const dg = dateFlatGroups.find(d => d.dk === selected.dk);
      currentItems      = dg?.byType[selected.type] ?? [];
      currentDateLabel  = dg?.dateLabel ?? "";
    }
  }

  if (loading) return <div className="lib-root"><div className="lib-loading">Loading library…</div></div>;
  if (error)   return <div className="lib-root"><div className="result-error">{error}</div></div>;

  return (
    <div className="lib-root">
      {/* ── Preview modal ───────────────────────────────────────────────── */}
      {preview && <PreviewModal item={preview} onClose={() => setPreview(null)} />}

      {/* ── Left sidebar ────────────────────────────────────────────────── */}
      <aside className="lib-sidebar">
        <div className="lib-sidebar-header">
          <span className="section-label blue">LIBRARY</span>
          <button className="lib-refresh-btn" onClick={load} title="Refresh">↺</button>
        </div>

        {outputs.length === 0 ? (
          <div className="lib-empty-sidebar">No outputs yet</div>
        ) : isAdmin ? (
          /* Admin: username → date → type */
          <div className="lib-date-list">
            {userGroups.map(ug => {
              const isUserOpen = expandedUsers.has(ug.userId);
              return (
                <div key={ug.userId} className="lib-user-group">
                  <button
                    className={`lib-user-btn ${isUserOpen ? "lib-user-btn-open" : ""}`}
                    onClick={() => toggleUser(ug.userId)}
                  >
                    <span className="lib-date-chevron">{isUserOpen ? "▾" : "▸"}</span>
                    <span className="lib-user-name">{ug.username}</span>
                  </button>
                  {isUserOpen && (
                    <div className="lib-user-dates">
                      {ug.dates.map(date => (
                        <DateGroupBlock
                          key={date.dk}
                          date={date}
                          selected={selected?.userId === ug.userId ? selected : null}
                          expanded={expandedDates.has(date.dk)}
                          onToggle={() => toggleDate(date.dk)}
                          onSelect={(dk, type) => selectType(dk, type, ug.userId)}
                        />
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          /* User: date → type */
          <div className="lib-date-list">
            {dateFlatGroups.map(date => (
              <DateGroupBlock
                key={date.dk}
                date={date}
                selected={selected}
                expanded={expandedDates.has(date.dk)}
                onToggle={() => toggleDate(date.dk)}
                onSelect={(dk, type) => selectType(dk, type)}
              />
            ))}
          </div>
        )}
      </aside>

      {/* ── Main content ────────────────────────────────────────────────── */}
      <main className="lib-main">
        {!selected ? (
          <div className="lib-placeholder">
            <span className="lib-placeholder-icon">🗂</span>
            <span className="lib-placeholder-text">Select a category from the sidebar</span>
          </div>
        ) : currentItems.length === 0 ? (
          <div className="lib-placeholder">
            <span className="lib-placeholder-text">No items found</span>
          </div>
        ) : (
          <>
            <div className="lib-main-header">
              <span className={`lib-badge ${TYPE_COLORS[selected.type]}`}>
                {TYPE_LABELS[selected.type]}
              </span>
              {isAdmin && selected.userId != null && (
                <span className="lib-main-user">
                  {userGroups.find(g => g.userId === selected.userId)?.username}
                </span>
              )}
              <span className="lib-main-date">{currentDateLabel}</span>
              <span className="lib-main-count">
                {currentItems.length} item{currentItems.length !== 1 ? "s" : ""}
              </span>
            </div>
            <div className="lib-grid">
              {currentItems.map(item => (
                <MediaCard key={item.id} item={item} onPreview={setPreview} />
              ))}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
