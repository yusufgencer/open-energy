import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { listJobs } from "@/lib/api";

export default async function JobsPage() {
  const jobs = await listJobs(50).catch(() => []);
  return (
    <div className="flex flex-col gap-5">
      <div>
        <h1 className="text-2xl font-bold">İş Merkezi</h1>
        <p className="text-sm text-muted-foreground">En yeni 50 arka plan işi ve çalışma durumu.</p>
      </div>
      <div className="grid gap-3">
        {jobs.map((job) => (
          <Card key={job.job_id}>
            <CardHeader className="flex-row items-center justify-between">
              <div>
                <CardTitle className="text-base">#{job.job_id} · {job.kind}</CardTitle>
                <p className="text-xs text-muted-foreground">{formatDate(job.created_at)}</p>
              </div>
              <Badge variant={job.status === "dead" || job.status === "failed" ? "destructive" : "secondary"}>
                {job.status}
              </Badge>
            </CardHeader>
            <CardContent className="flex flex-col gap-2">
              <Progress value={Math.max(0, Math.min(100, job.progress * 100))} />
              <p className="text-xs text-muted-foreground">{job.detail ?? "Ayrıntı yok"}</p>
              {job.last_error && <p className="text-xs text-destructive">{job.last_error}</p>}
            </CardContent>
          </Card>
        ))}
        {jobs.length === 0 && <p className="text-sm text-muted-foreground">Henüz iş kaydı yok.</p>}
      </div>
    </div>
  );
}

function formatDate(value?: string): string {
  return value ? new Date(value).toLocaleString("tr-TR") : "—";
}
