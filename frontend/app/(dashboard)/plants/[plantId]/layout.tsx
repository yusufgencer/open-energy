import { notFound } from "next/navigation";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Badge } from "@/components/ui/badge";
import PlantNav from "@/components/plant-nav";
import { getPlant } from "@/lib/api";

interface PlantLayoutProps {
  params: Promise<{ plantId: string }>;
  children: React.ReactNode;
}

export default async function PlantLayout(props: PlantLayoutProps) {
  const { plantId } = await props.params;

  let plant;
  try {
    const data = await getPlant(plantId);
    plant = data.plant;
  } catch {
    notFound();
  }

  const isWind = plant.kind === "wind";

  return (
    <div className="flex flex-col gap-0 -m-4">
      <header className="flex h-14 shrink-0 items-center gap-2 border-b bg-background px-4">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-2 h-4" />
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href="/plants">Santrallerim</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage className="flex items-center gap-2">
                {plant.name}
                <Badge
                  variant="outline"
                  className="text-xs"
                  style={{
                    color: isWind ? "#0EA5E9" : "#F59E0B",
                    borderColor: isWind ? "#0EA5E9" : "#F59E0B",
                  }}
                >
                  {plant.capacity_mw} MW
                </Badge>
              </BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </header>
      <div className="border-b px-4">
        <PlantNav plantId={plantId} />
      </div>
      <div className="flex flex-1 flex-col gap-4 p-4">
        {props.children}
      </div>
    </div>
  );
}
