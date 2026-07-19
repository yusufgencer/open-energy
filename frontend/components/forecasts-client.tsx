"use client";

import { useState } from "react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import ForecastFanChart from "@/components/forecast-fan-chart";
import BacktestChart from "@/components/backtest-chart";
import type { PlantKind } from "@/lib/types";

const HORIZON_OPTIONS = [
  { value: 1, label: "1h" },
  { value: 6, label: "6h" },
  { value: 24, label: "24h" },
  { value: 48, label: "48h" },
];

interface Props {
  plantId: string;
  kind: PlantKind;
}

export default function ForecastsClient({ plantId, kind }: Props) {
  const [horizon, setHorizon] = useState(24);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h2 className="text-lg font-semibold">Forecast&apos;ler</h2>
          <p className="text-sm text-muted-foreground">
            Olasılıksal üretim tahmini (p10/p50/p90 quantile)
          </p>
        </div>
        <ToggleGroup
          type="single"
          value={String(horizon)}
          onValueChange={(v) => v && setHorizon(Number(v))}
        >
          {HORIZON_OPTIONS.map((h) => (
            <ToggleGroupItem key={h.value} value={String(h.value)}>
              {h.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>

      <Tabs defaultValue="forecast" className="gap-4">
        <TabsList>
          <TabsTrigger value="forecast">İleriye Dönük</TabsTrigger>
          <TabsTrigger value="backtest">Backtest (test)</TabsTrigger>
        </TabsList>

        <TabsContent value="forecast">
          <ForecastFanChart plantId={plantId} horizon={horizon} kind={kind} />
        </TabsContent>

        <TabsContent value="backtest">
          <div className="flex flex-col gap-2">
            <p className="text-sm text-muted-foreground">
              Şampiyon modelin ayrılmış test penceresindeki tahmini vs gerçek üretim
              — modelin geçmiş performansı.
            </p>
            <BacktestChart plantId={plantId} horizon={horizon} kind={kind} />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
