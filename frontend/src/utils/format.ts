const unavailable = '—'

function finite(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export function formatPercent(value: number | null | undefined): string {
  return finite(value)
    ? new Intl.NumberFormat('en-GB', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(value)
    : unavailable
}

export function formatRatio(value: number | null | undefined): string {
  return finite(value)
    ? new Intl.NumberFormat('en-GB', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(value)
    : unavailable
}

export function formatCount(value: number | null | undefined): string {
  return finite(value)
    ? new Intl.NumberFormat('en-GB', { maximumFractionDigits: 0 }).format(value)
    : unavailable
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return unavailable
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return unavailable
  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short',
  }).format(date)
}

export function formatLabel(value: string | null | undefined): string {
  if (!value) return unavailable
  return value
    .toLowerCase()
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

export function truncateDigest(value: string, visible = 12): string {
  if (value.length <= visible * 2 + 1) return value
  return `${value.slice(0, visible)}…${value.slice(-visible)}`
}
