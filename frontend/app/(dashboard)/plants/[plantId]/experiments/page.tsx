import { notFound } from "next/navigation";
import { getPlant } from "@/lib/api";
import ExperimentPanelV2 from "@/components/experiment-panel-v2";
import PolicyWinner from "@/components/policy-winner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { InfoIcon } from "lucide-react";

interface PageProps {
  params: Promise<{ plantId: string }>;
}

export default async function ExperimentsPage(props: PageProps) {
  const { plantId } = await props.params;

  let plant, points;
  try {
    const data = await getPlant(plantId);
    plant = data.plant;
    points = data.points;
  } catch {
    notFound();
  }

  return (
    <div className="flex flex-col gap-4">
      {points.length === 0 && (
        <Alert>
          <InfoIcon className="size-4" />
          <AlertTitle>Nokta gerekli</AlertTitle>
          <AlertDescription>
            Experiment başlatmadan önce haritadan en az bir nokta ekle.
          </AlertDescription>
        </Alert>
      )}
      <ExperimentPanelV2 plant={plant} points={points} />
      <PolicyWinner plantId={plant.plant_id} />
    </div>
  );
}
