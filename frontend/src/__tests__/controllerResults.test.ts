import { describe, expect, it } from 'vitest'
import { METRICS, improvementPct, isExperimental, pairs, presentControllers, results, verdict, type ControllerId, type VsBaseline } from '../lib/controllerResults'

const vs = (diff: number, lo: number, hi: number, paired = 10): VsBaseline => ({ diff, ci95: [lo, hi], pct: null, p: null, better: 0, worse: 0, paired })

describe('verdict', () => {
  it('is unclear when the interval crosses zero', () => {
    expect(verdict(METRICS.mean_wait_s, vs(-5, -9, 0.1))).toBe('unclear')
  })
  it('respects metric direction', () => {
    expect(verdict(METRICS.mean_wait_s, vs(-5, -9, -1))).toBe('better')
    expect(verdict(METRICS.mean_wait_s, vs(5, 1, 9))).toBe('worse')
    expect(verdict(METRICS.throughput_vph, vs(20, 5, 40))).toBe('better')
    expect(verdict(METRICS.throughput_vph, vs(-20, -40, -5))).toBe('worse')
  })
  it('is unclear without a comparison, and never clear from one or two paired runs', () => {
    expect(verdict(METRICS.mean_wait_s, undefined)).toBe('unclear')
    expect(verdict(METRICS.mean_wait_s, vs(-5, -9, -1, 2))).toBe('unclear')
  })
})

describe('improvementPct', () => {
  it('is positive when the compared controller is better, for both directions', () => {
    expect(improvementPct(METRICS.mean_wait_s, 100, vs(-10, -15, -5))).toEqual({ mid: 10, lo: 5, hi: 15 })
    expect(improvementPct(METRICS.throughput_vph, 100, vs(10, 5, 15))).toEqual({ mid: 10, lo: 5, hi: 15 })
  })
})

describe('recorded results', () => {
  it('uses the new schema', () => {
    expect(results.schema).toBe('flowstate.controller-results/2')
    expect(results.baseline).toBe('fixed_time')
    expect(results.flowstate).toBe('rl_ppo')
    expect(results.network.id).toBe('district')
  })
  it('holds exactly two controllers: fixed-time and a verified agent', () => {
    expect(presentControllers()).toEqual(['fixed_time', 'rl_ppo'])
    expect(Object.keys(results.controllers).sort()).toEqual(['fixed_time', 'rl_ppo'])
    expect(results.controllers.rl_ppo?.status).toBe('verified')
    expect(isExperimental(results.controllers.rl_ppo?.status)).toBe(false)
    expect(new Set(results.results.runs.map((r) => r.c))).toEqual(new Set<ControllerId>(['fixed_time', 'rl_ppo']))
    expect(Object.keys(results.results.vsFixed.overall).sort()).toEqual(['fixed_time', 'rl_ppo'])
  })
  it('pairs every agent run with a fixed-time run on the same demand', () => {
    const r = results.results
    const p = pairs(r, 'fixed_time', 'rl_ppo')
    expect(p.length).toBe(r.pairs)
    expect(r.vsFixed.overall.rl_ppo?.metrics.mean_wait_s?.vs?.paired).toBe(p.length)
  })
  it('agrees with the stored win count', () => {
    const r = results.results
    const wins = pairs(r, 'fixed_time', 'rl_ppo').filter((x) => x.b.wait < x.a.wait).length
    expect(wins).toBe(r.vsFixed.overall.rl_ppo?.metrics.mean_wait_s?.vs?.better)
  })
  it('holds one scenario out', () => {
    expect(results.scenarios.filter((s) => s.heldOut).map((s) => s.id)).toEqual(['stadium'])
  })
})
