"use client";

import { useEffect, useState } from "react";
import { TrophyIcon, Loader2Icon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getPolicyLeaderboard } from "@/lib/api";
import type { PolicyLeaderboardRow } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  plantId: string;
}

function pct(x: number | null): string {
  return x != null ? `${(x * 100).toFixed(1)}%` : "—";
}

export default function PolicyWinner({ plantId }: Props) {
  const [rows, setRows] = useState<PolicyLeaderboardRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getPolicyLeaderboard(plantId)
      .then((r) => active && setRows(r))
      .catch((e) => active && setError(e instanceof Error ? e.message : "Hata"));
    return () => {
      active = false;
    };
  }, [plantId]);

  if (error) return null;

  const winner = rows && rows.length > 0 ? rows[0] : null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <TrophyIcon className="size-4" />
          Bu santral için kazanan nokta × kaynak
        </CardTitle>
        <CardDescription>
          {"{uzamsal nokta × NWP kaynağı}"} kombinasyonlarının kazanma oranı —
          bu santralin öğrenilmiş hafızası.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {rows === null ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2Icon className="size-4 animate-spin" /> Yükleniyor...
          </p>
        ) : rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Henüz kayıtlı experiment yok. Bir grid kur, çek ve experiment
            çalıştır — kazanan kombinasyon burada görünür.
          </p>
        ) : (
          <div className="flex flex-col gap-3">
            {winner && (
              <div className="flex flex-wrap items-center gap-2 rounded-lg border bg-primary/5 p-3 text-sm">
                <Badge variant="default">Lider</Badge>
                <span className="font-mono font-medium">
                  {winner.point_policy}
                </span>
                <span className="text-muted-foreground">×</span>
                <span className="font-mono font-medium">
                  {winner.source_policy}
                </span>
                <span className="text-muted-foreground">
                  kazandı — {(winner.win_rate * 100).toFixed(0)}% kazanma,
                  skill {winner.median_skill_score != null
                    ? winner.median_skill_score.toFixed(3)
                    : "—"}
                </span>
              </div>
            )}

            <div className="max-h-72 overflow-auto rounded-lg border">
              <Table className="text-xs">
                <TableHeader className="sticky top-0 z-10 bg-muted/95">
                  <TableRow className="[&>th]:h-8 [&>th]:px-2 [&>th]:text-[11px] [&>th]:uppercase [&>th]:tracking-wide">
                    <TableHead>Nokta</TableHead>
                    <TableHead>Kaynak</TableHead>
                    <TableHead className="text-right">Deneme</TableHead>
                    <TableHead className="text-right">Kazanç</TableHead>
                    <TableHead className="text-right">Kazanma %</TableHead>
                    <TableHead className="text-right">Med. skill</TableHead>
                    <TableHead className="text-right">Med. nRMSE</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody className="[&>tr>td]:h-8 [&>tr>td]:py-1">
                  {rows.map((r, i) => (
                    <TableRow
                      key={`${r.point_policy}-${r.source_policy}`}
                      className={cn(i === 0 && "bg-primary/5")}
                    >
                      <TableCell className="font-mono">{r.point_policy}</TableCell>
                      <TableCell className="font-mono text-muted-foreground">
                        {r.source_policy}
                      </TableCell>
                      <TableCell className="num text-right">{r.trials}</TableCell>
                      <TableCell className="num text-right">{r.wins}</TableCell>
                      <TableCell className="num text-right font-medium">
                        {(r.win_rate * 100).toFixed(0)}%
                      </TableCell>
                      <TableCell className="num text-right">
                        {r.median_skill_score != null
                          ? r.median_skill_score.toFixed(3)
                          : "—"}
                      </TableCell>
                      <TableCell className="num text-right">
                        {pct(r.median_nrmse)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
