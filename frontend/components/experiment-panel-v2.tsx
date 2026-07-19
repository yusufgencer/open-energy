"use client";

import { useState, useRef, useEffect } from "react";
import { PlayIcon, FlaskConicalIcon } from "lucide-react";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Empty,
  EmptyHeader,
  EmptyTitle,
  EmptyDescription,
  EmptyMedia,
  EmptyContent,
} from "@/components/ui/empty";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import AccuracyCard from "@/components/accuracy-card";
import { startExperiment, getJob, getResults } from "@/lib/api";
import type { Plant, PlantPoint, TrialResult, Job } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  plant: Plant;
  points: PlantPoint[];
}

const HORIZON_OPTIONS = ["1", "6", "24", "48"] as const;

export default function ExperimentPanelV2({ plant, points }: Props) {
  const [open, setOpen] = useState(false);
  const [pointId, setPointId] = useState<string>(
    String(points[0]?.point_id ?? "")
  );
  const [horizons, setHorizons] = useState<string[]>(["24"]);
  const [nTrials, setNTrials] = useState("10");
  const nwpSources = "icon";

  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<TrialResult[] | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const pollJob = (jobId: number) => {
    intervalRef.current = setInterval(async () => {
      try {
        const job: Job = await getJob(jobId);
        setJobStatus(job.status);
        setProgress(job.progress);

        if (job.status === "done") {
          stopPolling();
          const experimentId = Number(job.detail);
          if (!isNaN(experimentId)) {
            const res = await getResults(experimentId);
            setResults(res);
            toast.success("Experiment tamamlandı");
          } else {
            toast.error("Experiment ID alınamadı");
          }
          setLoading(false);
        } else if (job.status === "failed") {
          stopPolling();
          toast.error(`Experiment başarısız: ${job.detail ?? "bilinmeyen hata"}`);
          setLoading(false);
        }
      } catch (err) {
        stopPolling();
        toast.error(err instanceof Error ? err.message : "Polling hatası");
        setLoading(false);
      }
    }, 1500);
  };

  const handleStart = async () => {
    if (!pointId || horizons.length === 0) return;
    setLoading(true);
    setResults(null);
    setJobStatus("pending");
    setProgress(0);
    setOpen(false);

    const horizonList = horizons.map((h) => parseInt(h, 10));
    const nwpList = nwpSources.split(",").map((s) => s.trim()).filter(Boolean);

    try {
      const { job_id } = await startExperiment({
        plant_id: plant.plant_id,
        point_id: Number(pointId),
        horizons: horizonList,
        kind: plant.kind,
        capacity_mw: plant.capacity_mw,
        nwp_sources: nwpList,
        n_trials: parseInt(nTrials, 10),
      });
      toast.info("Experiment başlatıldı");
      pollJob(job_id);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Experiment başlatılamadı");
      setLoading(false);
    }
  };

  const groupedByHorizon = results
    ? Object.entries(
        results.reduce<Record<number, TrialResult[]>>((acc, r) => {
          acc[r.horizon_hours] = acc[r.horizon_hours] ?? [];
          acc[r.horizon_hours].push(r);
          return acc;
        }, {})
      ).sort(([a], [b]) => Number(a) - Number(b))
    : [];

  // suppress unused variable warning
  void groupedByHorizon;

  const champions = results?.filter((r) => r.is_champion) ?? [];

  return (
    <div className="flex flex-col gap-6">
      {/* Config dialog trigger */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Experiment&apos;ler</h2>
          <p className="text-sm text-muted-foreground">
            Farklı model ailelerini dene, en iyisini bul
          </p>
        </div>
        <Dialog open={open} onOpenChange={setOpen}>
          <DialogTrigger asChild>
            <Button disabled={loading || points.length === 0}>
              <PlayIcon data-icon />
              {loading ? "Çalışıyor..." : "Yeni Experiment"}
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Experiment Yapılandır</DialogTitle>
              <DialogDescription>
                Eğitim parametrelerini ayarla ve başlat.
              </DialogDescription>
            </DialogHeader>
            <FieldGroup>
              <Field>
                <FieldLabel>Nokta</FieldLabel>
                <Select value={pointId} onValueChange={setPointId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Nokta seç" />
                  </SelectTrigger>
                  <SelectContent>
                    {points
                      .filter((p) => p.point_id != null)
                      .map((p) => (
                        <SelectItem key={p.point_id!} value={String(p.point_id!)}>
                          {p.point_type} ({p.latitude.toFixed(3)},{" "}
                          {p.longitude.toFixed(3)})
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </Field>
              <Field>
                <FieldLabel>Horizon (saat)</FieldLabel>
                <ToggleGroup
                  type="multiple"
                  value={horizons}
                  onValueChange={(v) => v.length > 0 && setHorizons(v)}
                  className="flex flex-wrap gap-1"
                >
                  {HORIZON_OPTIONS.map((h) => (
                    <ToggleGroupItem key={h} value={h} size="sm">
                      {h}h
                    </ToggleGroupItem>
                  ))}
                </ToggleGroup>
              </Field>
              <Field>
                <FieldLabel>Deneme Sayısı</FieldLabel>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={nTrials}
                  onChange={(e) => setNTrials(e.target.value)}
                />
              </Field>
            </FieldGroup>
            <DialogFooter>
              <Button variant="outline" onClick={() => setOpen(false)}>
                İptal
              </Button>
              <Button onClick={handleStart} disabled={!pointId || horizons.length === 0}>
                Başlat
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {/* Job progress */}
      {loading && (
        <div className="flex flex-col gap-2 rounded-lg border p-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Çalışıyor...</span>
            <Badge variant="secondary">
              {jobStatus ?? "pending"}
            </Badge>
          </div>
          <Progress value={Math.round(progress * 100)} className="h-2" />
          <span className="text-xs text-muted-foreground">
            {Math.round(progress * 100)}% tamamlandı
          </span>
        </div>
      )}

      {/* Empty state */}
      {!loading && results === null && (
        <Empty className="border min-h-48">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <FlaskConicalIcon />
            </EmptyMedia>
            <EmptyTitle>Henüz experiment yok</EmptyTitle>
          </EmptyHeader>
          <EmptyContent>
            <EmptyDescription>
              {points.length === 0
                ? "Önce haritadan bir nokta ekle, ardından experiment başlat."
                : "Yeni experiment başlatmak için yukarıdaki butona tıkla."}
            </EmptyDescription>
          </EmptyContent>
        </Empty>
      )}

      {/* Results */}
      {results !== null && (
        <Tabs defaultValue="cards">
          <TabsList>
            <TabsTrigger value="cards">Özet Kartlar</TabsTrigger>
            <TabsTrigger value="table">Detay Tablosu</TabsTrigger>
          </TabsList>

          <TabsContent value="cards" className="mt-4">
            {champions.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                Sonuç bulunamadı (yeterli veri olmayabilir).
              </p>
            ) : (
              <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
                {champions.map((r, i) => (
                  <AccuracyCard key={i} result={r} />
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="table" className="mt-4">
            {/* Yoğun terminal tablosu: 32px satır, yapışkan başlık, sayısal hizalama */}
            <div className="max-h-[28rem] overflow-auto rounded-lg border">
              <Table className="text-xs">
                <TableHeader className="sticky top-0 z-10 bg-muted/95 backdrop-blur supports-[backdrop-filter]:bg-muted/80">
                  <TableRow className="[&>th]:h-8 [&>th]:px-2 [&>th]:text-[11px] [&>th]:uppercase [&>th]:tracking-wide">
                    <TableHead>Horizon</TableHead>
                    <TableHead>Model</TableHead>
                    <TableHead>NWP</TableHead>
                    <TableHead className="text-right">nRMSE</TableHead>
                    <TableHead className="text-right">Skill</TableHead>
                    <TableHead>Kazanan nokta×kaynak</TableHead>
                    <TableHead>Feature Blocks</TableHead>
                    <TableHead className="text-center">★</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody className="[&>tr>td]:h-8 [&>tr>td]:py-1">
                  {results.map((r, i) => {
                    const blocks: string[] = (() => {
                      try { return JSON.parse(r.feature_blocks) as string[]; }
                      catch { return []; }
                    })();
                    const isWarn = r.skill_score < 0;
                    return (
                      <TableRow
                        key={i}
                        className={cn(
                          isWarn && "bg-destructive/5",
                          r.is_champion && "bg-primary/5"
                        )}
                      >
                        <TableCell className="num tabular-nums">{r.horizon_hours}h</TableCell>
                        <TableCell className="font-mono">
                          {r.model_family}
                        </TableCell>
                        <TableCell className="font-mono text-muted-foreground">
                          {r.nwp_source}
                        </TableCell>
                        <TableCell className="num text-right font-medium">
                          {r.test_nrmse !== null
                            ? `${(r.test_nrmse * 100).toFixed(1)}%`
                            : "—"}
                        </TableCell>
                        <TableCell
                          className={cn(
                            "num text-right font-medium",
                            isWarn
                              ? "text-[var(--negative)]"
                              : r.skill_score >= 0.1 && "text-[var(--positive)]"
                          )}
                        >
                          {r.skill_score >= 0 ? "▲" : "▼"} {r.skill_score.toFixed(3)}
                        </TableCell>
                        <TableCell className="font-mono text-muted-foreground">
                          {r.point_policy || r.source_policy ? (
                            <>
                              {r.point_policy ?? "single_point"}
                              <span className="text-muted-foreground/50"> × </span>
                              {r.source_policy ?? "single_source"}
                            </>
                          ) : (
                            "—"
                          )}
                        </TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {blocks.map((b) => (
                              <Badge key={b} variant="outline" className="text-[10px] font-mono px-1 py-0">
                                {b}
                              </Badge>
                            ))}
                          </div>
                        </TableCell>
                        <TableCell className="text-center">
                          {r.is_champion && (
                            <span className="text-primary" title="Şampiyon">★</span>
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
