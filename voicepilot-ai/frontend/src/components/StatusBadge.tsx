import { IN_PROGRESS } from "../services/api";
import type { LectureStatus } from "../services/api";

const LABELS: Record<LectureStatus, string> = {
  queued: "Queued",
  downloading: "Downloading audio…",
  processing: "Transcribing…",
  completed: "Ready",
  failed: "Failed",
};

export default function StatusBadge({ status }: { status: LectureStatus }) {
  const busy = IN_PROGRESS.includes(status);
  return (
    <span className={`badge badge-${status}`}>
      {busy && <span className="spinner spinner-xs" aria-hidden="true" />}
      {LABELS[status] ?? status}
    </span>
  );
}
