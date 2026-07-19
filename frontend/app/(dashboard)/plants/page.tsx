import Link from "next/link";
import { WindIcon, SunIcon, ArrowRightIcon, ActivityIcon } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import {
  Empty,
  EmptyHeader,
  EmptyTitle,
  EmptyDescription,
  EmptyMedia,
  EmptyContent,
} from "@/components/ui/empty";
import { listPlants } from "@/lib/api";
import CreatePlantDialog from "@/components/create-plant-dialog";
import type { Plant } from "@/lib/types";

function PlantCard({ plant }: { plant: Plant }) {
  const isWind = plant.kind === "wind";
  return (
    <Card className="flex flex-col">
      <CardHeader className="flex-row items-start gap-3">
        <div
          className="flex size-10 shrink-0 items-center justify-center rounded-lg text-white"
          style={{ backgroundColor: isWind ? "#0EA5E9" : "#F59E0B" }}
        >
          {isWind ? <WindIcon className="size-5" /> : <SunIcon className="size-5" />}
        </div>
        <div className="flex-1 min-w-0">
          <CardTitle className="text-base truncate">{plant.name}</CardTitle>
          <CardDescription className="truncate">{plant.plant_id}</CardDescription>
        </div>
      </CardHeader>
      <CardContent className="flex-1">
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">
            {plant.capacity_mw} MW
          </Badge>
          <Badge variant="outline">{plant.timezone}</Badge>
        </div>
      </CardContent>
      <CardFooter>
        <Button variant="ghost" size="sm" asChild className="w-full">
          <Link href={`/plants/${plant.plant_id}`}>
            Aç
            <ArrowRightIcon data-icon />
          </Link>
        </Button>
      </CardFooter>
    </Card>
  );
}

export default async function PlantsPage() {
  let plants: Plant[] = [];
  let fetchError: string | null = null;

  try {
    plants = await listPlants();
  } catch (err) {
    fetchError = err instanceof Error ? err.message : "Santraller yüklenemedi";
  }

  return (
    <div className="flex flex-col">
      <header className="flex h-14 shrink-0 items-center gap-2 border-b bg-background px-4">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-2 h-4" />
        <span className="text-sm font-medium">Santrallerim</span>
      </header>
      <div className="flex flex-col gap-6 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Santrallerim</h1>
            <p className="text-sm text-muted-foreground">
              {plants.length > 0
                ? `${plants.length} santral kayıtlı`
                : "Henüz santral yok"}
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" asChild>
              <Link href="/fleet"><ActivityIcon data-icon /> Filo Sağlığı</Link>
            </Button>
            <CreatePlantDialog />
          </div>
        </div>

        {fetchError && (
          <p className="text-sm text-destructive">{fetchError}</p>
        )}

        {plants.length === 0 && !fetchError ? (
          <Empty className="border min-h-64">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <WindIcon />
              </EmptyMedia>
              <EmptyTitle>Henüz santral yok</EmptyTitle>
            </EmptyHeader>
            <EmptyContent>
              <EmptyDescription>
                İlk santralini oluşturarak tahmin sürecini başlat.
              </EmptyDescription>
              <CreatePlantDialog />
            </EmptyContent>
          </Empty>
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {plants.map((plant) => (
              <PlantCard key={plant.plant_id} plant={plant} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
