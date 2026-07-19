import { notFound } from "next/navigation";
import { getPlant } from "@/lib/api";
import OverviewKpi from "@/components/overview-kpi";

interface PageProps {
  params: Promise<{ plantId: string }>;
}

export default async function PlantOverviewPage(props: PageProps) {
  const { plantId } = await props.params;

  let plant, points;
  try {
    const data = await getPlant(plantId);
    plant = data.plant;
    points = data.points;
  } catch {
    notFound();
  }

  return <OverviewKpi plant={plant} points={points} />;
}
