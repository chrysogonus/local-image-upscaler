import type { JobSnapshot } from "../types";

interface ProgressStatusProps {
  job: JobSnapshot | null;
}

export function ProgressStatus({ job }: ProgressStatusProps) {
  // The live region has to exist before its text changes for a screen reader
  // to announce it, so the idle state keeps the element and says nothing.
  if (!job) return <div className="status idle" role="status" aria-live="polite" />;
  const terminalError = job.state === "failed" || job.state === "cancelled";
  return (
    <div
      className={`status ${terminalError ? "error" : job.state === "completed" ? "success" : "busy"}`}
      role="status"
      aria-live="polite"
    >
      <div className="status-line">
        <span>{job.message}</span>
        <span>{job.progress == null ? "" : `${Math.round(job.progress * 100)}%`}</span>
      </div>
      <div
        className={`progress-track ${job.progress == null ? "indeterminate" : ""}`}
        aria-hidden="true"
      >
        <span style={job.progress == null ? undefined : { width: `${job.progress * 100}%` }} />
      </div>
      {job.error ? <p>{job.error}</p> : null}
    </div>
  );
}
