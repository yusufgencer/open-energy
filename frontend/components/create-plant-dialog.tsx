"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { PlusIcon } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { createPlant } from "@/lib/api";
import type { Plant, PlantKind } from "@/lib/types";

export default function CreatePlantDialog() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState<Plant>({
    plant_id: "",
    name: "",
    kind: "wind",
    capacity_mw: 0,
    timezone: "Europe/Istanbul",
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await createPlant(form);
      toast.success(`${form.name} santrali oluşturuldu`);
      setOpen(false);
      router.refresh();
      router.push(`/plants/${form.plant_id}`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Santral oluşturulamadı");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <PlusIcon data-icon />
          Yeni Santral
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Yeni Santral Oluştur</DialogTitle>
          <DialogDescription>
            Rüzgar veya güneş santralini tanımla.
          </DialogDescription>
        </DialogHeader>
        <form id="create-plant-form" onSubmit={handleSubmit}>
          <FieldGroup>
            <Field>
              <FieldLabel>Santral ID</FieldLabel>
              <Input
                placeholder="orn-wf-01"
                value={form.plant_id}
                onChange={(e) => setForm((f) => ({ ...f, plant_id: e.target.value }))}
                required
              />
            </Field>
            <Field>
              <FieldLabel>Ad</FieldLabel>
              <Input
                placeholder="Ege Rüzgar Santrali"
                value={form.name}
                onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                required
              />
            </Field>
            <Field>
              <FieldLabel>Tür</FieldLabel>
              <Select
                value={form.kind}
                onValueChange={(v) => setForm((f) => ({ ...f, kind: v as PlantKind }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="wind">Rüzgar</SelectItem>
                  <SelectItem value="solar">Güneş</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            <Field>
              <FieldLabel>Kapasite (MW)</FieldLabel>
              <Input
                type="number"
                step="0.1"
                min="0"
                value={form.capacity_mw}
                onChange={(e) =>
                  setForm((f) => ({ ...f, capacity_mw: Number(e.target.value) }))
                }
                required
              />
            </Field>
            <Field>
              <FieldLabel>Zaman Dilimi</FieldLabel>
              <Input
                value={form.timezone}
                onChange={(e) => setForm((f) => ({ ...f, timezone: e.target.value }))}
                required
              />
            </Field>
          </FieldGroup>
        </form>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>
            İptal
          </Button>
          <Button type="submit" form="create-plant-form" disabled={loading}>
            {loading ? "Oluşturuluyor..." : "Oluştur"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
