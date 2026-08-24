interface MetricCardProps {
  label: string
  value: string
  note?: string
  emphasis?: 'default' | 'oos'
}

export function MetricCard({
  label,
  value,
  note,
  emphasis = 'default',
}: MetricCardProps) {
  return (
    <article className={`research-metric metric-${emphasis}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      {note && <span>{note}</span>}
    </article>
  )
}
