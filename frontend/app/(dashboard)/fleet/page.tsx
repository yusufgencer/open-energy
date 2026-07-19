import Link from "next/link";
import { ActivityIcon, ArrowRightIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getFleetHealth } from "@/lib/api";

export default async function FleetPage() {
  const rows = await getFleetHealth().catch(() => []);
  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-2xl font-bold">Filo Sağlığı</h1>
        <p className="text-sm text-muted-foreground">
          Tahmin, üretim, drift ve aktif iş sinyallerinin tek görünümü.
        </p>
      </div>
      <div className="grid gap-3">
        {rows.map((row) => (
          <Card key={row.plant_id}>
            <CardHeader className="flex-row items-center justify-between">
              <div>
                <CardTitle className="text-base">{row.name}</CardTitle>
                <p className="text-xs text-muted-foreground">{row.plant_id} · {row.capacity_mw} MW</p>
              </div>
              <Badge variant={row.status === "critical" ? "destructive" : "secondary"}>
                {row.status}
              </Badge>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-4">
              <Signal label="Son tahmin" value={formatDate(row.latest_forecast_at)} />
              <Signal label="Son üretim" value={formatDate(row.latest_production_at)} />
              <Signal label="Drift" value={row.drift_status} />
              <Signal label="Aktif iş" value={String(row.active_jobs)} />
              <Button variant="ghost" size="sm" asChild className="sm:col-start-4">
                <Link href={`/plants/${row.plant_id}/monitoring`}>
                  İzlemeyi aç <ArrowRightIcon data-icon />
                </Link>
              </Button>
            </CardContent>
          </Card>
        ))}
        {rows.length === 0 && (
          <div className="flex min-h-48 flex-col items-center justify-center gap-2 rounded-xl border text-muted-foreground">
            <ActivityIcon className="size-6" />
            <p className="text-sm">Filo sağlık verisi yok.</p>
          </div>
        )}
      </div>
    </div>
  );
}

function Signal({ label, value }: { label: string; value: string }) {
  return <div><p className="text-xs text-muted-foreground">{label}</p><p className="num text-sm font-medium">{value}</p></div>;
}

function formatDate(value: string | null): string {
  return value ? new Date(value).toLocaleString("tr-TR") : "—";
}
