"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  PlugZapIcon,
  SearchIcon,
  DownloadIcon,
  CheckCircle2Icon,
  LinkIcon,
  Loader2Icon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, FieldGroup, FieldLabel, FieldDescription } from "@/components/ui/field";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  epiasStatus,
  epiasConnect,
  epiasDisconnect,
  epiasSearchPlants,
  ingestEpias,
  getJob,
} from "@/lib/api";
import type { EpiasStatus, EpiasPowerplant, Job } from "@/lib/types";

interface Props {
  plantId: string;
  plantName: string;
}

function isoDaysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 10);
}

export default function EpiasIntegration({ plantId, plantName }: Props) {
  const [status, setStatus] = useState<EpiasStatus | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [connecting, setConnecting] = useState(false);

  const [query, setQuery] = useState(plantName);
  const [results, setResults] = useState<EpiasPowerplant[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [selected, setSelected] = useState<EpiasPowerplant | null>(null);

  const [startDate, setStartDate] = useState(isoDaysAgo(92));
  const [endDate, setEndDate] = useState(isoDaysAgo(0));
  const [field, setField] = useState<"total" | "wind">("total");

  const [pulling, setPulling] = useState(false);
  const [progress, setProgress] = useState(0);
  const [rows, setRows] = useState<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    epiasStatus().then(setStatus).catch(() => setStatus({ connected: false, username: null }));
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const handleConnect = async () => {
    if (!username || !password) return;
    setConnecting(true);
    try {
      const s = await epiasConnect(username, password);
      setStatus(s);
      setPassword("");
      toast.success("EPİAŞ'a bağlanıldı");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Bağlantı hatası";
      toast.error(
        msg.includes("401") ? "Kullanıcı adı veya şifre hatalı" : `Bağlanılamadı: ${msg}`
      );
    } finally {
      setConnecting(false);
    }
  };

  const handleDisconnect = async () => {
    try {
      const s = await epiasDisconnect();
      setStatus(s);
      setResults(null);
      setSelected(null);
      toast.info("EPİAŞ bağlantısı kesildi");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Hata");
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    setResults(null);
    try {
      const r = await epiasSearchPlants(query.trim());
      setResults(r);
      if (r.length === 0) toast.info("Eşleşen santral bulunamadı");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Arama hatası");
    } finally {
      setSearching(false);
    }
  };

  const pollJob = (jobId: number) => {
    intervalRef.current = setInterval(async () => {
      try {
        const job: Job = await getJob(jobId);
        setProgress(job.progress);
        if (job.status === "done") {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setRows(Number(job.detail));
          setPulling(false);
          toast.success(`${job.detail} satır üretim verisi çekildi`);
        } else if (job.status === "failed") {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setPulling(false);
          toast.error(`Çekme başarısız: ${job.detail ?? "bilinmeyen hata"}`);
        }
      } catch (err) {
        if (intervalRef.current) clearInterval(intervalRef.current);
        setPulling(false);
        toast.error(err instanceof Error ? err.message : "Polling hatası");
      }
    }, 1500);
  };

  const handlePull = async () => {
    if (!selected) return;
    setPulling(true);
    setProgress(0);
    setRows(null);
    try {
      const { job_id } = await ingestEpias(plantId, {
        power_plant_id: selected.id,
        start_date: startDate,
        end_date: endDate,
        field,
      });
      toast.info("EPİAŞ veri çekme başlatıldı");
      pollJob(job_id);
    } catch (err) {
      setPulling(false);
      toast.error(err instanceof Error ? err.message : "Çekme başlatılamadı");
    }
  };

  // ── Bağlı değil: kimlik formu ──────────────────────────────────────────────
  if (status && !status.connected) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <PlugZapIcon className="size-4" />
            EPİAŞ Şeffaflık Bağlantısı
          </CardTitle>
          <CardDescription>
            EPİAŞ hesabınla bağlan, santralinin gerçek üretim verisini otomatik çek.
            Ücretsiz hesap:{" "}
            <a
              href="https://kayit.epias.com.tr"
              target="_blank"
              rel="noreferrer"
              className="underline underline-offset-2"
            >
              kayit.epias.com.tr
            </a>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="epias-user">Kullanıcı adı (e-posta)</FieldLabel>
              <Input
                id="epias-user"
                type="email"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="ornek@firma.com"
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="epias-pass">Şifre</FieldLabel>
              <Input
                id="epias-pass"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleConnect()}
              />
              <FieldDescription>
                Kimlik bilgilerin yalnızca kendi yerel sunucunda saklanır, dışarı gönderilmez.
              </FieldDescription>
            </Field>
            <Button
              onClick={handleConnect}
              disabled={connecting || !username || !password}
            >
              {connecting ? (
                <Loader2Icon data-icon className="animate-spin" />
              ) : (
                <LinkIcon data-icon />
              )}
              {connecting ? "Bağlanıyor..." : "Bağlan"}
            </Button>
          </FieldGroup>
        </CardContent>
      </Card>
    );
  }

  // ── Bağlı: ara + çek ───────────────────────────────────────────────────────
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <CardTitle className="flex items-center gap-2 text-base">
              <CheckCircle2Icon className="size-4 text-[var(--positive)]" />
              EPİAŞ&apos;tan Veri Çek
            </CardTitle>
            <CardDescription className="mt-1">
              {status?.username ? (
                <>
                  Bağlı:{" "}
                  <span className="font-medium text-foreground">{status.username}</span>
                </>
              ) : (
                "Bağlı"
              )}
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm" onClick={handleDisconnect}>
            Bağlantıyı kes
          </Button>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {/* Santral ara */}
        <div className="flex flex-col gap-2">
          <FieldLabel htmlFor="epias-q">Santral ara</FieldLabel>
          <div className="flex gap-2">
            <Input
              id="epias-q"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="ör. BARES / BALIKESIR"
            />
            <Button variant="outline" onClick={handleSearch} disabled={searching}>
              {searching ? (
                <Loader2Icon data-icon className="animate-spin" />
              ) : (
                <SearchIcon data-icon />
              )}
              Ara
            </Button>
          </div>
        </div>

        {results && results.length > 0 && (
          <div className="flex flex-col gap-1 rounded-lg border p-1">
            {results.map((pp) => (
              <button
                key={pp.id}
                onClick={() => setSelected(pp)}
                className={cn(
                  "flex items-center justify-between rounded-md px-2 py-1.5 text-left text-sm transition-colors",
                  selected?.id === pp.id ? "bg-muted font-medium" : "hover:bg-muted/50"
                )}
              >
                <span className="truncate">{pp.name}</span>
                <Badge variant="outline" className="num ml-2 shrink-0 text-[10px]">
                  id {pp.id}
                </Badge>
              </button>
            ))}
          </div>
        )}

        {/* Çekme parametreleri */}
        {selected && (
          <div className="flex flex-col gap-3 rounded-lg border bg-muted/30 p-3">
            <p className="text-sm">
              Seçili: <span className="font-medium">{selected.name}</span>
            </p>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <Field>
                <FieldLabel htmlFor="epias-start">Başlangıç</FieldLabel>
                <Input
                  id="epias-start"
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="epias-end">Bitiş</FieldLabel>
                <Input
                  id="epias-end"
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                />
              </Field>
              <Field>
                <FieldLabel htmlFor="epias-field">Alan</FieldLabel>
                <select
                  id="epias-field"
                  value={field}
                  onChange={(e) => setField(e.target.value as "total" | "wind")}
                  className="h-9 rounded-md border bg-transparent px-2 text-sm"
                >
                  <option value="total">total (santral)</option>
                  <option value="wind">wind</option>
                </select>
              </Field>
            </div>
            <Button onClick={handlePull} disabled={pulling}>
              {pulling ? (
                <Loader2Icon data-icon className="animate-spin" />
              ) : (
                <DownloadIcon data-icon />
              )}
              {pulling ? "Çekiliyor..." : "Veriyi Çek"}
            </Button>
            {pulling && <Progress value={Math.round(progress * 100)} className="h-1.5" />}
            {rows !== null && !pulling && (
              <p className="num text-sm text-[var(--positive)]">
                ✓ {rows} satır üretim verisi yüklendi
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
