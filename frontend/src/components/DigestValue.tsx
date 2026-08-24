import { Check, Copy } from 'lucide-react'
import { useState } from 'react'
import { truncateDigest } from '../utils/format'

export function DigestValue({ value }: { value: string }) {
  const [copied, setCopied] = useState(false)

  async function copyDigest() {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <button
      className="digest-button"
      type="button"
      onClick={copyDigest}
      title={value}
      aria-label={`Copy digest ${value}`}
    >
      <code>{truncateDigest(value)}</code>
      {copied ? <Check size={13} aria-hidden="true" /> : <Copy size={13} aria-hidden="true" />}
      <span className="sr-only">{copied ? 'Copied' : 'Copy full digest'}</span>
    </button>
  )
}
