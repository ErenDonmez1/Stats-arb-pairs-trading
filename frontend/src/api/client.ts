import type {
  ExperimentPage,
  ExperimentSummary,
  HealthResponse,
  MetaResponse,
} from '../types/api'
import {
  DEMO_HEALTH,
  DEMO_META,
  getDemoExperiment,
  getDemoExperimentPage,
} from '../data/demo'

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()
export const API_BASE_URL = (configuredBaseUrl || 'http://localhost:8000').replace(/\/$/, '')
export const IS_DEMO_MODE = import.meta.env.VITE_DEMO_MODE?.trim().toLowerCase() === 'true'

export class ApiError extends Error {
  readonly status: number | null

  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function requestJson<T>(path: string): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { Accept: 'application/json' },
    })
  } catch {
    throw new ApiError('Unable to connect to the research API.')
  }

  if (!response.ok) {
    throw new ApiError(`Research API request failed (${response.status}).`, response.status)
  }

  try {
    const payload: unknown = await response.json()
    if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
      throw new ApiError('Research API returned an unexpected response shape.', response.status)
    }
    return payload as T
  } catch {
    throw new ApiError('Research API returned an invalid JSON response.', response.status)
  }
}

export function getHealth(): Promise<HealthResponse> {
  if (IS_DEMO_MODE) return Promise.resolve(structuredClone(DEMO_HEALTH))
  return requestJson<HealthResponse>('/health')
}

export function getMeta(): Promise<MetaResponse> {
  if (IS_DEMO_MODE) return Promise.resolve(structuredClone(DEMO_META))
  return requestJson<MetaResponse>('/api/v1/meta')
}

export function getExperiments({
  limit,
  offset,
}: {
  limit: number
  offset: number
}): Promise<ExperimentPage> {
  if (IS_DEMO_MODE) return Promise.resolve(getDemoExperimentPage(limit, offset))

  const query = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  })
  return requestJson<ExperimentPage>(`/api/v1/experiments?${query}`)
}

export function getExperiment(runId: string): Promise<ExperimentSummary> {
  if (IS_DEMO_MODE) {
    const experiment = getDemoExperiment(runId)
    return experiment
      ? Promise.resolve(experiment)
      : Promise.reject(new ApiError('Synthetic demo experiment was not found.', 404))
  }

  return requestJson<ExperimentSummary>(
    `/api/v1/experiments/${encodeURIComponent(runId)}`,
  )
}
