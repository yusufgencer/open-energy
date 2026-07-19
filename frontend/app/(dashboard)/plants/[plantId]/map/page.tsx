import { notFound } from "next/navigation";
import { getPlant } from "@/lib/api";
import MapPanelWrapper from "@/components/map-panel-wrapper";

interface PageProps {
  params: Promise<{ plantId: string }>;
}

export default async function MapPage(props: PageProps) {
  const { plantId } = await props.params;

  let points;
  try {
    const data = await getPlant(plantId);
    points = data.points;
  } catch {
    notFound();
  }

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold">Harita & Noktalar</h2>
        <p className="text-sm text-muted-foreground">
          Türbin, panel veya hava istasyonu konumlarını ekle
        </p>
      </div>
      <MapPanelWrapper plantId={plantId} initialPoints={points} />
    </div>
  );
}
