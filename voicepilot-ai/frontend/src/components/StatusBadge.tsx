import { IN_PROGRESS } from "../services/api";
import type { LectureStatus } from "../services/api";

const LABELS: Record<LectureStatus, string> = {
  queued: "Queued",
  downloading: "Downloading audio…",
  processing: "Transcribing…",
  completed: "Ready",
  failed: "Failed",
};

interface Props {
  status: LectureStatus;
  /** Shown as "Transcribing… 43%" while processing, when known. */
  progress?: number | null;
}

export default function StatusBadge({ status, progress }: Props) {
  const busy = IN_PROGRESS.includes(status);
  const showPercent = status === "processing" && progress != null;
  return (
    <span className={`badge badge-${status}`}>
      {busy && <span className="spinner spinner-xs" aria-hidden="true" />}
      {LABELS[status] ?? status}
      {showPercent && ` ${Math.round(progress)}%`}
    </span>
  );
}
