import { notFound } from "next/navigation";
import DriftMonitoring from "@/components/drift-monitoring";
import { getPlant } from "@/lib/api";

export default async function MonitoringPage({
  params,
}: PageProps<"/plants/[plantId]/monitoring">) {
  const { plantId } = await params;
  let data;
  try {
    data = await getPlant(plantId);
  } catch {
    notFound();
  }
  return <DriftMonitoring plant={data.plant} pointId={data.points[0]?.point_id ?? null} />;
}
