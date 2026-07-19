import { notFound } from "next/navigation";
import { getPlant } from "@/lib/api";
import ProductionUpload from "@/components/production-upload";
import EpiasIntegration from "@/components/epias-integration";
import WeatherGridSetup from "@/components/weather-grid-setup";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { InfoIcon } from "lucide-react";

interface PageProps {
  params: Promise<{ plantId: string }>;
}

export default async function DataPage(props: PageProps) {
  const { plantId } = await props.params;

  let plant, points;
  try {
    const data = await getPlant(plantId);
    plant = data.plant;
    points = data.points;
  } catch {
    notFound();
  }

  const hasPoints = points.length > 0;

  return (
    <div className="flex flex-col gap-4 max-w-2xl">
      <div>
        <h2 className="text-lg font-semibold">Geçmiş Veri</h2>
        <p className="text-sm text-muted-foreground">
          Gerçek üretim verisini getir — model eğitimi için kullanılır
        </p>
      </div>

      {!hasPoints && (
        <Alert>
          <InfoIcon className="size-4" />
          <AlertTitle>Önce nokta ekle</AlertTitle>
          <AlertDescription>
            Veri yüklemeden önce haritadan en az bir nokta eklemen gerekiyor.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="epias">
        <TabsList>
          <TabsTrigger value="epias">EPİAŞ&apos;tan Çek</TabsTrigger>
          <TabsTrigger value="csv">CSV Yükle</TabsTrigger>
          <TabsTrigger value="grid">Hava Gridi</TabsTrigger>
        </TabsList>
        <TabsContent value="epias" className="mt-4">
          <EpiasIntegration plantId={plantId} plantName={plant.name} />
        </TabsContent>
        <TabsContent value="csv" className="mt-4">
          <ProductionUpload plantId={plantId} />
        </TabsContent>
        <TabsContent value="grid" className="mt-4">
          {hasPoints ? (
            <WeatherGridSetup plantId={plantId} />
          ) : (
            <Alert>
              <InfoIcon className="size-4" />
              <AlertTitle>Önce nokta ekle</AlertTitle>
              <AlertDescription>
                Grid, santralin temsili noktası etrafında kurulur — önce
                haritadan bir nokta ekle.
              </AlertDescription>
            </Alert>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
