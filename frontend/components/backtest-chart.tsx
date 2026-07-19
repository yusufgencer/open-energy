"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Skeleton } from "@/components/ui/skeleton";
import { Empty, EmptyHeader, EmptyTitle, EmptyMedia, EmptyDescription } from "@/components/ui/empty";
import { HistoryIcon } from "lucide-react";
import { getBacktest } from "@/lib/api";
import type { BacktestRow, PlantKind } from "@/lib/types";

const ACCENT: Record<PlantKind, string> = {
  wind: "#0EA5E9",
  solar: "#F59E0B",
};

interface Props {
  plantId: string;
  horizon: number;
  kind: PlantKind;
}

interface ChartRow {
  time: string;
  actual: number;
  p50: number;
  p10: number | null;
  p90: number | null;
}

function formatTime(isoTime: string): string {
  const d = new Date(isoTime);
  const dd = String(d.getDate()).padStart(2, "0");
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  return `${dd}/${mm} ${hh}:00`;
}

export default function BacktestChart({ plantId, horizon, kind }: Props) {
  const [rows, setRows] = useState<BacktestRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getBacktest(plantId, horizon)
      .then((r) => {
        setRows(r);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setLoading(false);
      });
  }, [plantId, horizon]);

  if (loading) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-48" />
        <Skeleton className="h-72 w-full rounded-xl" />
      </div>
    );
  }
  if (error) return <p className="text-sm text-destructive">{error}</p>;
  if (rows.length === 0) {
    return (
      <Empty className="border min-h-72">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <HistoryIcon />
          </EmptyMedia>
          <EmptyTitle>Backtest verisi yok</EmptyTitle>
          <EmptyDescription>
            Bu horizon için henüz şampiyon model yok. Önce bir deney çalıştır.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    );
  }

  return <BacktestView rows={rows} kind={kind} />;
}

/** Saf sunum bileşeni — test penceresi gerçek üretim vs tahmin (p10/p50/p90). */
export function BacktestView({ rows, kind }: { rows: BacktestRow[]; kind: PlantKind }) {
  const accent = ACCENT[kind];
  const gradientId = `btGradient-${kind}`;

  const data: ChartRow[] = useMemo(
    () =>
      rows.map((r) => ({
        time: formatTime(r.valid_time),
        actual: r.actual,
        p50: r.p50,
        p10: r.p10,
        p90: r.p90,
      })),
    [rows]
  );

  const metrics = useMemo(() => {
    let ae = 0;
    let covN = 0;
    let cov = 0;
    for (const r of rows) {
      ae += Math.abs(r.actual - r.p50);
      if (r.p10 != null && r.p90 != null) {
        covN += 1;
        if (r.actual >= r.p10 && r.actual <= r.p90) cov += 1;
      }
    }
    return {
      mae: rows.length ? ae / rows.length : null,
      coverage: covN ? cov / covN : null,
      n: rows.length,
    };
  }, [rows]);

  const hasBand = rows[0]?.p10 != null;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs">
        <Metric label="MAE" value={metrics.mae != null ? `${metrics.mae.toFixed(1)} MW` : "—"} />
        <Metric
          label="Coverage (p10–p90)"
          value={metrics.coverage != null ? `${(metrics.coverage * 100).toFixed(0)}%` : "—"}
        />
        <Metric label="Test noktası" value={String(metrics.n)} />
      </div>

      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={accent} stopOpacity={0.26} />
              <stop offset="95%" stopColor={accent} stopOpacity={0.04} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            interval="preserveStartEnd"
            minTickGap={48}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={false}
            width={40}
            label={{
              value: "MW",
              angle: -90,
              position: "insideLeft",
              style: { fontSize: 10, fill: "var(--muted-foreground)" },
            }}
          />
          <Tooltip
            cursor={{ stroke: "var(--border)", strokeWidth: 1 }}
            content={({ active, label, payload }) => (
              <BacktestTooltip
                active={active}
                label={label as string}
                row={(payload?.[0]?.payload as ChartRow) ?? undefined}
                accent={accent}
              />
            )}
          />

          {hasBand && (
            <>
              <Area
                type="monotone"
                dataKey="p90"
                fill={`url(#${gradientId})`}
                stroke="none"
                name="p90"
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="p10"
                fill="var(--background)"
                stroke="none"
                fillOpacity={1}
                name="p10"
                isAnimationActive={false}
              />
            </>
          )}
          {/* Tahmin p50 — accent renk, kesikli (forecast konvansiyonu) */}
          <Line
            type="monotone"
            dataKey="p50"
            stroke={accent}
            strokeWidth={2}
            strokeDasharray="6 3"
            dot={false}
            name="p50"
            isAnimationActive={false}
          />
          {/* Gerçek üretim — düz ink çizgi (ground truth) */}
          <Line
            type="monotone"
            dataKey="actual"
            stroke="var(--foreground)"
            strokeWidth={2}
            dot={false}
            name="actual"
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0 w-4 border-t-2" style={{ borderColor: "var(--foreground)" }} />
          Gerçek üretim
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0 w-4 border-t-2 border-dashed" style={{ borderColor: accent }} />
          Tahmin p50
        </span>
        {hasBand && (
          <span className="flex items-center gap-1.5">
            <span className="inline-block size-3 rounded-sm" style={{ backgroundColor: accent, opacity: 0.22 }} />
            p10–p90 bandı
          </span>
        )}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-muted-foreground">{label}</span>
      <span className="num font-medium text-foreground">{value}</span>
    </span>
  );
}

const TT_ROWS: { key: "actual" | "p50"; label: string }[] = [
  { key: "actual", label: "Gerçek" },
  { key: "p50", label: "Tahmin p50" },
];

function BacktestTooltip({
  active,
  label,
  row,
  accent,
}: {
  active?: boolean;
  label?: string;
  row?: ChartRow;
  accent: string;
}) {
  if (!active || !row) return null;
  return (
    <div className="rounded-md border border-border/60 bg-background px-3 py-2 text-xs shadow-xl">
      <p className="mb-1.5 font-medium">{label}</p>
      <div className="flex flex-col gap-0.5">
        {TT_ROWS.map((r) => (
          <div key={r.key} className="flex items-center justify-between gap-6">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              <span
                className="size-2 rounded-full"
                style={{ backgroundColor: r.key === "actual" ? "var(--foreground)" : accent }}
              />
              {r.label}
            </span>
            <span className="num font-medium text-foreground">{row[r.key].toFixed(2)} MW</span>
          </div>
        ))}
        {row.p10 != null && row.p90 != null && (
          <div className="flex items-center justify-between gap-6">
            <span className="text-muted-foreground">p10–p90</span>
            <span className="num text-foreground">
              {row.p10.toFixed(1)} – {row.p90.toFixed(1)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
