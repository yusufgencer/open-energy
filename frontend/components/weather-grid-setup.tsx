"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  Grid3x3Icon,
  DownloadIcon,
  Loader2Icon,
  CheckCircle2Icon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, FieldLabel, FieldDescription } from "@/components/ui/field";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { createWeatherGrid, ingestWeather, getJob } from "@/lib/api";
import type { Job, WeatherGridResponse } from "@/lib/types";

interface Props {
  plantId: string;
}

// Open-Meteo previous-runs NWP sources. The grid is built for one geometry
// (source), but ingest can fill it from several models — the source_policy axis.
const SOURCE_OPTIONS = [
  { id: "icon_eu", label: "ICON-EU" },
  { id: "ecmwf_ifs025", label: "ECMWF IFS" },
  { id: "gfs_global", label: "GFS" },
] as const;

const SIZE_OPTIONS = ["3", "5", "7"] as const;

export default function WeatherGridSetup({ plantId }: Props) {
  const [source, setSource] = useState<string>("icon_eu");
  const [size, setSize] = useState<string>("5");
  const [upwind, setUpwind] = useState<string>("");
  const [building, setBuilding] = useState(false);
  const [grid, setGrid] = useState<WeatherGridResponse | null>(null);

  // Ingest sources (multi-select): the source_policy axis the search explores.
  const [sources, setSources] = useState<string[]>(["icon_eu"]);
  const [pastDays, setPastDays] = useState<string>("92");
  const [ingesting, setIngesting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [rows, setRows] = useState<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const handleBuild = async () => {
    setBuilding(true);
    setGrid(null);
    setRows(null);
    try {
      const res = await createWeatherGrid(plantId, {
        source,
        size: parseInt(size, 10),
        upwind_deg: upwind.trim() === "" ? null : Number(upwind),
      });
      setGrid(res);
      // Default the ingest source list to the grid's own source.
      setSources((prev) => (prev.includes(source) ? prev : [source]));
      toast.success(`${res.count} noktalı grid oluşturuldu (${res.grid_id})`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Grid oluşturulamadı";
      toast.error(
        msg.includes("400") && msg.includes("representative")
          ? "Önce haritadan bir nokta ekle (grid onun etrafında kurulur)"
          : msg
      );
    } finally {
      setBuilding(false);
    }
  };

  const pollJob = (jobId: number) => {
    intervalRef.current = setInterval(async () => {
      try {
        const job: Job = await getJob(jobId);
        setProgress(job.progress);
        if (job.status === "done") {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setRows(Number(job.detail));
          setIngesting(false);
          toast.success(`${job.detail} satır eğitim havası çekildi`);
        } else if (job.status === "failed") {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setIngesting(false);
          toast.error(`Çekme başarısız: ${job.detail ?? "bilinmeyen hata"}`);
        }
      } catch (err) {
        if (intervalRef.current) clearInterval(intervalRef.current);
        setIngesting(false);
        toast.error(err instanceof Error ? err.message : "Polling hatası");
      }
    }, 1500);
  };

  const handleIngest = async () => {
    if (!grid || sources.length === 0) return;
    setIngesting(true);
    setProgress(0);
    setRows(null);
    try {
      const { job_id } = await ingestWeather(plantId, {
        grid_id: grid.grid_id,
        sources,
        past_days: parseInt(pastDays, 10),
      });
      toast.info("Eğitim havası çekme başlatıldı");
      pollJob(job_id);
    } catch (err) {
      setIngesting(false);
      toast.error(err instanceof Error ? err.message : "Çekme başlatılamadı");
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Grid3x3Icon className="size-4" />
          Hava Gridi (eğitim verisi)
        </CardTitle>
        <CardDescription>
          Santral etrafında bir NxN hava gridi kur ve previous-runs eğitim
          havasını doldur. Grid, arama sırasında en iyi <em>uzamsal nokta</em> ve{" "}
          <em>kaynak</em> kombinasyonunun bulunmasını sağlar.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {/* ── Grid geometrisi ── */}
        <div className="grid gap-3 sm:grid-cols-3">
          <Field>
            <FieldLabel htmlFor="grid-source">Kaynak (geometri)</FieldLabel>
            <select
              id="grid-source"
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="h-9 rounded-md border bg-transparent px-2 text-sm"
            >
              {SOURCE_OPTIONS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </Field>
          <Field>
            <FieldLabel>Boyut</FieldLabel>
            <ToggleGroup
              type="single"
              value={size}
              onValueChange={(v) => v && setSize(v)}
              className="flex gap-1"
            >
              {SIZE_OPTIONS.map((s) => (
                <ToggleGroupItem key={s} value={s} size="sm">
                  {s}×{s}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </Field>
          <Field>
            <FieldLabel htmlFor="grid-upwind">Rüzgâr yönü (°)</FieldLabel>
            <Input
              id="grid-upwind"
              type="number"
              min="0"
              max="360"
              value={upwind}
              onChange={(e) => setUpwind(e.target.value)}
              placeholder="opsiyonel"
            />
            <FieldDescription>Grid o yöne kayar (upwind lean).</FieldDescription>
          </Field>
        </div>

        <Button onClick={handleBuild} disabled={building}>
          {building ? (
            <Loader2Icon data-icon className="animate-spin" />
          ) : (
            <Grid3x3Icon data-icon />
          )}
          {building ? "Oluşturuluyor..." : "Grid Oluştur"}
        </Button>

        {/* ── Grid hazır: çek ── */}
        {grid && (
          <div className="flex flex-col gap-3 rounded-lg border bg-muted/30 p-3">
            <div className="flex items-center justify-between">
              <p className="text-sm">
                Grid:{" "}
                <span className="font-mono font-medium">{grid.grid_id}</span>
              </p>
              <Badge variant="outline" className="num text-[10px]">
                {grid.count} nokta
              </Badge>
            </div>

            <Field>
              <FieldLabel>Kaynaklar (source_policy)</FieldLabel>
              <ToggleGroup
                type="multiple"
                value={sources}
                onValueChange={(v) => v.length > 0 && setSources(v)}
                className="flex flex-wrap gap-1"
              >
                {SOURCE_OPTIONS.map((s) => (
                  <ToggleGroupItem key={s.id} value={s.id} size="sm">
                    {s.label}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
              <FieldDescription>
                Birden fazla model seç → arama kaynak kombinasyonlarını dener.
              </FieldDescription>
            </Field>

            <Field>
              <FieldLabel htmlFor="grid-past">Geçmiş gün</FieldLabel>
              <Input
                id="grid-past"
                type="number"
                min="1"
                max="92"
                value={pastDays}
                onChange={(e) => setPastDays(e.target.value)}
              />
            </Field>

            <Button onClick={handleIngest} disabled={ingesting || sources.length === 0}>
              {ingesting ? (
                <Loader2Icon data-icon className="animate-spin" />
              ) : (
                <DownloadIcon data-icon />
              )}
              {ingesting ? "Çekiliyor..." : "Eğitim Havasını Çek"}
            </Button>
            {ingesting && (
              <Progress value={Math.round(progress * 100)} className="h-1.5" />
            )}
            {rows !== null && !ingesting && (
              <p className="num flex items-center gap-1 text-sm text-[var(--positive)]">
                <CheckCircle2Icon className="size-4" />
                {rows} satır eğitim havası yüklendi
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
