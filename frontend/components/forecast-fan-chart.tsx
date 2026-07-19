"use client";

import { useEffect, useState } from "react";
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Empty,
  EmptyHeader,
  EmptyTitle,
  EmptyMedia,
} from "@/components/ui/empty";
import { TrendingUpIcon } from "lucide-react";
import { getForecasts } from "@/lib/api";
import type { ForecastRow, PlantKind } from "@/lib/types";

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
  p10: number;
  p50: number;
  p90: number;
}

function formatTime(isoTime: string): string {
  const d = new Date(isoTime);
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  return `${dd} ${hh}:00`;
}

const TT_ROWS: { key: "p90" | "p50" | "p10"; label: string; dim: boolean }[] = [
  { key: "p90", label: "p90 (üst)", dim: true },
  { key: "p50", label: "p50 medyan", dim: false },
  { key: "p10", label: "p10 (alt)", dim: true },
];

function FanTooltip({
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
                style={{
                  backgroundColor: accent,
                  opacity: r.dim ? 0.35 : 1,
                }}
              />
              {r.label}
            </span>
            <span className="num font-medium text-foreground">
              {row[r.key].toFixed(2)} MW
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ForecastFanChart({ plantId, horizon, kind }: Props) {
  const [data, setData] = useState<ChartRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getForecasts(plantId, horizon)
      .then((rows: ForecastRow[]) => {
        setData(
          rows.map((r) => ({
            time: formatTime(r.valid_time),
            p10: r.p10,
            p50: r.p50,
            p90: r.p90,
          }))
        );
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
        <Skeleton className="h-64 w-full rounded-xl" />
      </div>
    );
  }

  if (error) {
    return (
      <p className="text-sm text-destructive">{error}</p>
    );
  }

  if (data.length === 0) {
    return (
      <Empty className="border min-h-64">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <TrendingUpIcon />
          </EmptyMedia>
          <EmptyTitle>Tahmin verisi yok</EmptyTitle>
        </EmptyHeader>
      </Empty>
    );
  }

  return <FanChartView data={data} kind={kind} />;
}

/** Saf sunum bileşeni — veri çekmeden p10/p50/p90 fan grafiğini çizer. */
export function FanChartView({ data, kind }: { data: ChartRow[]; kind: PlantKind }) {
  const accent = ACCENT[kind];
  const gradientId = `fanGradient-${kind}`;
  // "Şimdi" çizgisi: tüm seri ileriye dönük tahmin olduğundan ilk zaman damgasında.
  const nowLabel = data[0]?.time;

  return (
    <div className="flex flex-col gap-2">
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={accent} stopOpacity={0.28} />
              <stop offset="95%" stopColor={accent} stopOpacity={0.04} />
            </linearGradient>
          </defs>
          {/* ONS spec: ince yatay gridline, dikey gridline yok */}
          <CartesianGrid stroke="var(--border)" vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={{ stroke: "var(--border)" }}
            interval="preserveStartEnd"
            minTickGap={32}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
            tickLine={false}
            axisLine={false}
            width={40}
            tickFormatter={(v: number) => `${v}`}
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
              <FanTooltip
                active={active}
                label={label as string}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                row={(payload?.[0]?.payload as ChartRow) ?? undefined}
                accent={accent}
              />
            )}
          />

          {/* "Şimdi" referans çizgisi — tahmin başlangıcı */}
          {nowLabel && (
            <ReferenceLine
              x={nowLabel}
              stroke="var(--muted-foreground)"
              strokeDasharray="4 3"
              strokeWidth={1}
              label={{
                value: "Şimdi",
                position: "insideTopLeft",
                fontSize: 10,
                fill: "var(--muted-foreground)",
              }}
            />
          )}

          {/* p90 fan area — top of the band */}
          <Area
            type="monotone"
            dataKey="p90"
            fill={`url(#${gradientId})`}
            stroke="none"
            name="p90"
            isAnimationActive={false}
          />
          {/* p10 eraser — fills below p10 with background color */}
          <Area
            type="monotone"
            dataKey="p10"
            fill="var(--background)"
            stroke="none"
            fillOpacity={1}
            name="p10"
            isAnimationActive={false}
          />
          {/* p50 median — dashed: tahmin (forecast) konvansiyonu */}
          <Line
            type="monotone"
            dataKey="p50"
            stroke={accent}
            strokeWidth={2.25}
            strokeDasharray="6 3"
            dot={false}
            name="p50"
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend / band açıklaması */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-0 w-4 border-t-2 border-dashed"
            style={{ borderColor: accent }}
          />
          p50 medyan
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block size-3 rounded-sm"
            style={{ backgroundColor: accent, opacity: 0.22 }}
          />
          p10–p90 (%80 güven aralığı)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-0 w-4 border-t border-dashed border-muted-foreground" />
          Şimdi
        </span>
      </div>
    </div>
  );
}
