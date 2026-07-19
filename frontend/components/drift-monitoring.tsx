"use client";

import { useEffect, useState } from "react";
import { ActivityIcon, RefreshCwIcon, WrenchIcon, ZapIcon } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getDrift,
  startDriftCheck,
  startPostprocessorFit,
  startReforecast,
  startRetrain,
} from "@/lib/api";
import { driftSeverity } from "@/lib/drift";
import type { DriftStatus, Plant } from "@/lib/types";

export { driftSeverity } from "@/lib/drift";

interface Props {
  plant: Plant;
  pointId: number | null;
}

const HORIZONS = [24, 48];

export default function DriftMonitoring({ plant, pointId }: Props) {
  const [rows, setRows] = useState<DriftStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState<string | null>(null);

  const refresh = () => {
    setLoading(true);
    getDrift(plant.plant_id)
      .then(setRows)
      .catch((error: Error) => toast.error(error.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    let active = true;
    getDrift(plant.plant_id)
      .then((nextRows) => {
        if (active) setRows(nextRows);
      })
      .catch((error: Error) => toast.error(error.message))
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [plant.plant_id]);

  const run = async (label: string, operation: () => Promise<{ job_id: number }>) => {
    setRunning(label);
    try {
      const job = await operation();
      toast.success(`${label} işi #${job.job_id} kuyruğa alındı`);
      refresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : `${label} başlatılamadı`);
    } finally {
      setRunning(null);
    }
  };

  const disabled = pointId == null || running != null;
  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">Drift & İzleme</h2>
        <p className="text-sm text-muted-foreground">
          Veri tazeliğini ve şampiyon modelin saha performansını izle; alarmı doğrudan aksiyona bağla.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button
          disabled={disabled}
          onClick={() => run("Drift kontrolü", () => startDriftCheck(plant.plant_id, {
            point_id: pointId!, horizons: HORIZONS, kind: plant.kind,
          }))}
        >
          <ActivityIcon data-icon /> Drift Kontrolü
        </Button>
        <Button
          variant="outline"
          disabled={disabled}
          onClick={() => run("Yeniden eğitim", () => startRetrain(plant.plant_id, {
            plant_id: plant.plant_id, point_id: pointId!, horizons: HORIZONS,
            kind: plant.kind, capacity_mw: plant.capacity_mw,
          }))}
        >
          <RefreshCwIcon data-icon /> Yeniden Eğit
        </Button>
        <Button
          variant="outline"
          disabled={disabled}
          onClick={() => run("Yeniden tahmin", () => startReforecast(plant.plant_id, {
            point_id: pointId!, horizon_hours: 24, kind: plant.kind,
          }))}
        >
          <ZapIcon data-icon /> Yeniden Tahmin
        </Button>
        <Button
          variant="outline"
          disabled={running != null}
          onClick={() => run("Kalibrasyon düzeltmesi", () => startPostprocessorFit(
            plant.plant_id, { horizons: HORIZONS }
          ))}
        >
          <WrenchIcon data-icon /> Postprocessor Fit
        </Button>
      </div>
      {pointId == null && (
        <p className="text-sm text-destructive">Manuel kontroller için önce bir santral noktası ekle.</p>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        {loading ? (
          <p className="text-sm text-muted-foreground">Drift durumu yükleniyor…</p>
        ) : rows.length === 0 ? (
          <Card>
            <CardHeader><CardTitle className="text-base">Henüz drift ölçümü yok</CardTitle></CardHeader>
            <CardContent className="text-sm text-muted-foreground">
              İlk ölçümü başlatmak için “Drift Kontrolü” düğmesini kullan.
            </CardContent>
          </Card>
        ) : rows.map((row) => {
          const severity = driftSeverity(row);
          return (
            <Card key={row.horizon_hours}>
              <CardHeader className="flex-row items-center justify-between">
                <CardTitle className="text-base">{row.horizon_hours} saat</CardTitle>
                <Badge variant={severity === "critical" ? "destructive" : "secondary"}>
                  {severity}
                </Badge>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-3 text-sm">
                <Value label="Güncel nRMSE" value={row.recent_nrmse} />
                <Value label="Şampiyon nRMSE" value={row.champion_nrmse} />
                <Value label="Oran" value={row.ratio} />
                <Value label="Ardışık alarm" value={row.consecutive_breaches} />
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}

function Value({ label, value }: { label: string; value?: number | null }) {
  return (
    <div>
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="num font-medium">{value == null ? "—" : Number(value).toFixed(3)}</p>
    </div>
  );
}
