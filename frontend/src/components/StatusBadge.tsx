import { CircleCheck, CircleDashed, CircleX, MinusCircle } from 'lucide-react'
import { formatLabel } from '../utils/format'

function statusTone(status: string | null | undefined) {
  const normalized = status?.toUpperCase() ?? ''
  if (
    normalized === 'COMPLETED' ||
    normalized === 'AVAILABLE' ||
    normalized === 'SELECTED'
  ) return 'positive'
  if (
    normalized === 'REJECTED' ||
    normalized.includes('FAIL') ||
    normalized.includes('ERROR')
  ) return 'negative'
  if (
    normalized.includes('UNAVAILABLE') ||
    normalized.includes('NOT_REQUESTED') ||
    normalized.includes('INSUFFICIENT')
  ) return 'neutral'
  return 'pending'
}

export function StatusBadge({ status }: { status: string | null | undefined }) {
  const tone = statusTone(status)
  const Icon =
    tone === 'positive'
      ? CircleCheck
      : tone === 'negative'
        ? CircleX
        : tone === 'neutral'
          ? MinusCircle
          : CircleDashed

  return (
    <span className={`status-badge badge-${tone}`}>
      <Icon size={13} strokeWidth={1.9} aria-hidden="true" />
      {formatLabel(status)}
    </span>
  )
}
