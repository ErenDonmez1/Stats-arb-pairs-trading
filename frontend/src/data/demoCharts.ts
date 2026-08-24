export type DemoScreeningStatus = 'SELECTED' | 'REJECTED'

export interface DemoPairCandidate {
  pairId: string
  symbolY: string
  symbolX: string
  group: string
  screeningStatus: string
  halfLife: number
  hurst: number
  rank: number | null
  status: DemoScreeningStatus
  experimentRunId: string | null
}

export interface DemoResearchPoint {
  date: string
  equity: number
  drawdown: number
  rollingSharpe: number | null
  normalizedX: number
  normalizedY: number
  spread: number
  spreadMean: number
  zscore: number
  beta: number
  event: 'ENTER_LONG' | 'ENTER_SHORT' | 'EXIT' | null
  eventValue: number | null
}

export interface DemoFoldResult {
  fold: string
  return: number
  status: 'SELECTED' | 'NO_SELECTION'
}

export interface DemoResearchCharts {
  pairId: string
  points: DemoResearchPoint[]
  foldReturns: DemoFoldResult[]
  entryZ: number
  exitZ: number
  stopZ: number
  oosTradeCount: number
}

export interface DemoRobustnessCell {
  scenarioId: string
  entryZ: number
  formationDays: number
  sharpe: number
  totalReturn: number
  baseline: boolean
}

export interface DemoCostStressPoint {
  costBps: number
  sharpe: number
  totalReturn: number
}

export interface DemoBootstrapInterval {
  metric: string
  lower: number
  median: number
  upper: number
  observed: number
}

export const DEMO_PAIR_CANDIDATES: readonly DemoPairCandidate[] = [
  {
    pairId: 'SYNTH_A / SYNTH_B',
    symbolY: 'SYNTH_A',
    symbolX: 'SYNTH_B',
    group: 'Synthetic Industrials',
    screeningStatus: 'Cointegrated · FDR passed',
    halfLife: 8.6,
    hurst: 0.38,
    rank: 1,
    status: 'SELECTED',
    experimentRunId: 'demo-daily-baseline-2025-01',
  },
  {
    pairId: 'SYNTH_C / SYNTH_D',
    symbolY: 'SYNTH_C',
    symbolX: 'SYNTH_D',
    group: 'Synthetic Financials',
    screeningStatus: 'Cointegrated · FDR passed',
    halfLife: 13.2,
    hurst: 0.44,
    rank: 2,
    status: 'SELECTED',
    experimentRunId: 'demo-cost-stress-2024-09',
  },
  {
    pairId: 'SYNTH_E / SYNTH_F',
    symbolY: 'SYNTH_E',
    symbolX: 'SYNTH_F',
    group: 'Synthetic Utilities',
    screeningStatus: 'FDR threshold exceeded',
    halfLife: 18.9,
    hurst: 0.49,
    rank: null,
    status: 'REJECTED',
    experimentRunId: null,
  },
  {
    pairId: 'SYNTH_G / SYNTH_H',
    symbolY: 'SYNTH_G',
    symbolX: 'SYNTH_H',
    group: 'Synthetic Energy',
    screeningStatus: 'Half-life exceeded',
    halfLife: 42.7,
    hurst: 0.41,
    rank: null,
    status: 'REJECTED',
    experimentRunId: null,
  },
  {
    pairId: 'SYNTH_I / SYNTH_J',
    symbolY: 'SYNTH_I',
    symbolX: 'SYNTH_J',
    group: 'Synthetic Healthcare',
    screeningStatus: 'Hurst threshold exceeded',
    halfLife: 16.4,
    hurst: 0.58,
    rank: null,
    status: 'REJECTED',
    experimentRunId: null,
  },
  {
    pairId: 'SYNTH_K / SYNTH_L',
    symbolY: 'SYNTH_K',
    symbolX: 'SYNTH_L',
    group: 'Synthetic Technology',
    screeningStatus: 'Cointegration not significant',
    halfLife: 27.1,
    hurst: 0.52,
    rank: null,
    status: 'REJECTED',
    experimentRunId: null,
  },
]

const MONTHS = Array.from({ length: 36 }, (_, index) => {
  const year = 2022 + Math.floor(index / 12)
  const month = String((index % 12) + 1).padStart(2, '0')
  return `${year}-${month}-01`
})

const BASELINE_EQUITY = [
  100, 99.3, 100.1, 101, 100.4, 101.3, 102, 101.5, 102.7, 103.2, 102.8, 103.9,
  104.4, 103.5, 102.1, 100.8, 99.2, 97.7, 98.6, 99.8, 101, 100.2, 101.6, 102.3,
  103.1, 102.6, 103.5, 104.2, 103.8, 104.9, 105.3, 104.6, 105.7, 106.1, 105.2, 104.6,
]

const STRESS_EQUITY = [
  100, 99.5, 100.2, 100.7, 99.8, 100.4, 101.1, 100.2, 100.9, 101.4, 100.5, 101.2,
  100.4, 99.7, 98.9, 98.1, 97.3, 96.6, 97.2, 97.9, 98.5, 97.6, 98.2, 98.8,
  98.1, 97.5, 98.3, 98.9, 98.2, 98.7, 99.1, 98.4, 98.9, 99.3, 98.1, 97.2,
]

const BASELINE_ZSCORE = [
  0.1, 0.6, 1.1, 1.7, 2.25, 1.4, 0.35, -0.2, -0.8, -1.4, -0.9, 0.2,
  0.7, 1.2, 0.5, -0.6, -1.3, -2.2, -1.1, -0.25, 0.4, 1.0, 0.3, -0.5,
  -1.1, -0.4, 0.6, 1.3, 2.45, 1.2, 0.25, -0.4, -1.0, -1.7, -0.8, 0.1,
]

const STRESS_ZSCORE = [
  -0.2, 0.4, 1.0, 1.6, 2.1, 1.7, 0.8, 0.1, -0.6, -1.2, -1.8, -0.7,
  0.2, 0.8, 1.5, 2.35, 2.9, 3.2, 2.1, 0.9, 0.2, -0.5, -1.4, -2.15,
  -1.6, -0.8, 0.1, 0.9, 1.8, 2.3, 1.4, 0.4, -0.3, -1.1, -1.7, -0.6,
]

const BASELINE_ROLLING_SHARPE: Array<number | null> = [
  null, null, null, null, null, null, 0.12, 0.19, 0.31, 0.42, 0.36, 0.48,
  0.55, 0.43, 0.22, 0.08, -0.1, -0.24, -0.18, -0.05, 0.09, 0.02, 0.16, 0.21,
  0.28, 0.23, 0.31, 0.39, 0.34, 0.43, 0.47, 0.38, 0.45, 0.49, 0.35, 0.24,
]

function round(value: number, digits = 3): number {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

function drawdowns(equity: readonly number[]): number[] {
  let peak = equity[0]
  return equity.map((value) => {
    peak = Math.max(peak, value)
    return round(value / peak - 1, 4)
  })
}

function eventAt(index: number): DemoResearchPoint['event'] {
  const events: Record<number, DemoResearchPoint['event']> = {
    4: 'ENTER_SHORT',
    6: 'EXIT',
    17: 'ENTER_LONG',
    19: 'EXIT',
    28: 'ENTER_SHORT',
    30: 'EXIT',
  }
  return events[index] ?? null
}

function buildPoints(
  equity: readonly number[],
  zscores: readonly number[],
  sharpe: readonly (number | null)[],
  profileOffset: number,
): DemoResearchPoint[] {
  const drawdown = drawdowns(equity)
  const betaCycle = [0, 0.008, 0.014, 0.01, -0.004, -0.012]
  const xCycle = [0, 0.35, -0.15, 0.5, -0.3, 0.2]
  const yCycle = [0.2, -0.25, 0.45, -0.1, 0.55, -0.35]

  return MONTHS.map((date, index) => {
    const event = eventAt(index)
    const normalizedX = 100 + index * (0.52 + profileOffset * 0.02) + xCycle[index % 6]
    const normalizedY = 100 + index * (0.55 - profileOffset * 0.01) + yCycle[index % 6]
    const spreadMean = 0.003 * Math.sin(index / 4)
    const spread = spreadMean + zscores[index] * 0.018
    return {
      date,
      equity: equity[index],
      drawdown: drawdown[index],
      rollingSharpe: sharpe[index],
      normalizedX: round(normalizedX, 2),
      normalizedY: round(normalizedY, 2),
      spread: round(spread, 4),
      spreadMean: round(spreadMean, 4),
      zscore: zscores[index],
      beta: round(1.06 + profileOffset * -0.16 + betaCycle[index % 6], 3),
      event,
      eventValue: event ? zscores[index] : null,
    }
  })
}

const BASELINE_POINTS = buildPoints(
  BASELINE_EQUITY,
  BASELINE_ZSCORE,
  BASELINE_ROLLING_SHARPE,
  0,
)

const STRESS_POINTS = buildPoints(
  STRESS_EQUITY,
  STRESS_ZSCORE,
  BASELINE_ROLLING_SHARPE.map((value) => (value === null ? null : round(value - 0.38, 2))),
  1,
)

const CHARTS_BY_PAIR: Record<string, DemoResearchCharts> = {
  'SYNTH_A / SYNTH_B': {
    pairId: 'SYNTH_A / SYNTH_B',
    points: BASELINE_POINTS,
    entryZ: 2,
    exitZ: 0.5,
    stopZ: 3.5,
    oosTradeCount: 12,
    foldReturns: [
      { fold: 'F01', return: 0.012, status: 'SELECTED' },
      { fold: 'F02', return: -0.008, status: 'SELECTED' },
      { fold: 'F03', return: 0.017, status: 'SELECTED' },
      { fold: 'F04', return: 0, status: 'NO_SELECTION' },
      { fold: 'F05', return: -0.014, status: 'SELECTED' },
      { fold: 'F06', return: 0.011, status: 'SELECTED' },
      { fold: 'F07', return: 0.009, status: 'SELECTED' },
      { fold: 'F08', return: -0.006, status: 'SELECTED' },
      { fold: 'F09', return: 0.025, status: 'SELECTED' },
    ],
  },
  'SYNTH_C / SYNTH_D': {
    pairId: 'SYNTH_C / SYNTH_D',
    points: STRESS_POINTS,
    entryZ: 2,
    exitZ: 0.5,
    stopZ: 3.5,
    oosTradeCount: 10,
    foldReturns: [
      { fold: 'F01', return: 0.008, status: 'SELECTED' },
      { fold: 'F02', return: -0.012, status: 'SELECTED' },
      { fold: 'F03', return: 0, status: 'NO_SELECTION' },
      { fold: 'F04', return: -0.009, status: 'SELECTED' },
      { fold: 'F05', return: -0.016, status: 'SELECTED' },
      { fold: 'F06', return: 0.007, status: 'SELECTED' },
      { fold: 'F07', return: 0, status: 'NO_SELECTION' },
      { fold: 'F08', return: 0.005, status: 'SELECTED' },
      { fold: 'F09', return: -0.011, status: 'SELECTED' },
    ],
  },
}

export const DEMO_ROBUSTNESS_GRID: readonly DemoRobustnessCell[] = [
  { scenarioId: 's01', entryZ: 1.5, formationDays: 252, sharpe: 0.08, totalReturn: 0.012, baseline: false },
  { scenarioId: 's02', entryZ: 2, formationDays: 252, sharpe: 0.16, totalReturn: 0.028, baseline: false },
  { scenarioId: 's03', entryZ: 2.5, formationDays: 252, sharpe: 0.11, totalReturn: 0.019, baseline: false },
  { scenarioId: 's04', entryZ: 3, formationDays: 252, sharpe: -0.04, totalReturn: -0.006, baseline: false },
  { scenarioId: 's05', entryZ: 1.5, formationDays: 378, sharpe: 0.13, totalReturn: 0.022, baseline: false },
  { scenarioId: 'baseline', entryZ: 2, formationDays: 378, sharpe: 0.24, totalReturn: 0.046, baseline: true },
  { scenarioId: 's07', entryZ: 2.5, formationDays: 378, sharpe: 0.18, totalReturn: 0.033, baseline: false },
  { scenarioId: 's08', entryZ: 3, formationDays: 378, sharpe: 0.06, totalReturn: 0.009, baseline: false },
  { scenarioId: 's09', entryZ: 1.5, formationDays: 504, sharpe: 0.05, totalReturn: 0.008, baseline: false },
  { scenarioId: 's10', entryZ: 2, formationDays: 504, sharpe: 0.19, totalReturn: 0.036, baseline: false },
  { scenarioId: 's11', entryZ: 2.5, formationDays: 504, sharpe: 0.15, totalReturn: 0.026, baseline: false },
  { scenarioId: 's12', entryZ: 3, formationDays: 504, sharpe: 0.02, totalReturn: 0.003, baseline: false },
  { scenarioId: 's13', entryZ: 1.5, formationDays: 630, sharpe: -0.07, totalReturn: -0.011, baseline: false },
  { scenarioId: 's14', entryZ: 2, formationDays: 630, sharpe: 0.09, totalReturn: 0.014, baseline: false },
  { scenarioId: 's15', entryZ: 2.5, formationDays: 630, sharpe: 0.07, totalReturn: 0.011, baseline: false },
  { scenarioId: 's16', entryZ: 3, formationDays: 630, sharpe: -0.11, totalReturn: -0.018, baseline: false },
]

export const DEMO_COST_STRESS: readonly DemoCostStressPoint[] = [
  { costBps: 1, sharpe: 0.31, totalReturn: 0.058 },
  { costBps: 3, sharpe: 0.24, totalReturn: 0.046 },
  { costBps: 5, sharpe: 0.17, totalReturn: 0.034 },
  { costBps: 8, sharpe: 0.07, totalReturn: 0.017 },
  { costBps: 12, sharpe: -0.05, totalReturn: -0.004 },
]

export const DEMO_BOOTSTRAP_INTERVAL: DemoBootstrapInterval = {
  metric: 'OOS Sharpe',
  lower: -0.15,
  median: 0.22,
  upper: 0.58,
  observed: 0.24,
}

export function getDemoResearchCharts(pairId: string): DemoResearchCharts | null {
  const result = CHARTS_BY_PAIR[pairId]
  return result ? structuredClone(result) : null
}
