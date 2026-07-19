import { notFound } from "next/navigation";
import { getPlant } from "@/lib/api";
import ForecastsClient from "@/components/forecasts-client";

interface PageProps {
  params: Promise<{ plantId: string }>;
}

export default async function ForecastsPage(props: PageProps) {
  const { plantId } = await props.params;

  let plant;
  try {
    const data = await getPlant(plantId);
    plant = data.plant;
  } catch {
    notFound();
  }

  return <ForecastsClient plantId={plantId} kind={plant.kind} />;
}
