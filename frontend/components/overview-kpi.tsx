"use client";

import Link from "next/link";
import {
  MapPinIcon,
  DatabaseIcon,
  FlaskConicalIcon,
  TrendingUpIcon,
  CheckCircleIcon,
  CircleDashedIcon,
  ArrowRightIcon,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import type { Plant, PlantPoint } from "@/lib/types";

interface Props {
  plant: Plant;
  points: PlantPoint[];
}

interface StepStatus {
  label: string;
  description: string;
  done: boolean;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
}

export default function OverviewKpi({ plant, points }: Props) {
  const isWind = plant.kind === "wind";
  const accentColor = isWind ? "#0EA5E9" : "#F59E0B";

  const steps: StepStatus[] = [
    {
      label: "Harita Noktaları",
      description: "Türbin / panel konumlarını ekle",
      done: points.length > 0,
      href: `/plants/${plant.plant_id}/map`,
      icon: MapPinIcon,
    },
    {
      label: "Geçmiş Veri",
      description: "Üretim CSV'sini yükle",
      done: false, // would need a separate API call; placeholder
      href: `/plants/${plant.plant_id}/data`,
      icon: DatabaseIcon,
    },
    {
      label: "Experiment",
      description: "Model eğit ve karşılaştır",
      done: false, // would need experiment list API; placeholder
      href: `/plants/${plant.plant_id}/experiments`,
      icon: FlaskConicalIcon,
    },
    {
      label: "Forecast",
      description: "Üretim tahmini görüntüle",
      done: false,
      href: `/plants/${plant.plant_id}/forecasts`,
      icon: TrendingUpIcon,
    },
  ];

  const nextStep = steps.find((s) => !s.done);
  const completedCount = steps.filter((s) => s.done).length;

  const kpis: {
    label: string;
    value: string;
    unit: string;
    accent?: string;
  }[] = [
    { label: "Kapasite", value: String(plant.capacity_mw), unit: "MW kurulu" },
    { label: "Nokta Sayısı", value: String(points.length), unit: "konum" },
    {
      label: "Santral Türü",
      value: plant.kind === "wind" ? "Rüzgar" : "Güneş",
      unit: "enerji türü",
      accent: accentColor,
    },
    {
      label: "Kurulum",
      value: `${completedCount}/${steps.length}`,
      unit: "adım tamam",
    },
  ];

  return (
    <div className="flex flex-col gap-4">
      {/* KPI şeridi — terminal tarzı: küçük büyük-harf etiket, sol aksan çubuğu */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {kpis.map((k) => (
          <Card
            key={k.label}
            className="relative overflow-hidden border-l-4 py-3 gap-1"
            style={{ borderLeftColor: k.accent ?? "var(--border)" }}
          >
            <CardContent className="px-4">
              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                {k.label}
              </p>
              <p
                className="num mt-1 text-2xl font-semibold leading-none"
                style={k.accent ? { color: k.accent } : undefined}
              >
                {k.value}
              </p>
              <p className="mt-1 text-[11px] text-muted-foreground">{k.unit}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Next step guidance card */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Sıradaki adım</CardTitle>
          <CardDescription>
            Kurulum ilerlemeniz: {completedCount}/{steps.length} adım tamamlandı
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {steps.map((step, i) => (
            <div key={step.label}>
              {i > 0 && <Separator className="mb-3" />}
              <div className="flex items-center gap-3">
                <div
                  className={cn(
                    "flex size-8 shrink-0 items-center justify-center rounded-full",
                    step.done
                      ? "bg-green-100 text-green-600"
                      : "bg-muted text-muted-foreground"
                  )}
                >
                  {step.done ? (
                    <CheckCircleIcon className="size-4" />
                  ) : (
                    <CircleDashedIcon className="size-4" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p
                    className={cn(
                      "text-sm font-medium",
                      step.done && "line-through text-muted-foreground"
                    )}
                  >
                    {step.label}
                  </p>
                  <p className="text-xs text-muted-foreground truncate">
                    {step.description}
                  </p>
                </div>
                {!step.done && nextStep?.label === step.label && (
                  <Button size="sm" asChild>
                    <Link href={step.href}>
                      Başla
                      <ArrowRightIcon data-icon />
                    </Link>
                  </Button>
                )}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
