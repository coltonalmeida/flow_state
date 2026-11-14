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
    // the headline features the learned policy only while the gate has verified it; the rules are always the reference
    expect(results.flowstate).toBe(results.controllers.rl_ppo?.status === 'verified' ? 'rl_ppo' : 'queue_responsive')
    expect(results.rules).toBe('queue_responsive')
    expect(results.network.id).toBe('district')
  })
  it('lists the controllers present, baseline first, and the default is not experimental', () => {
    const ids = presentControllers()
    expect(ids[0]).toBe('fixed_time')
    expect(ids).toContain('queue_responsive')
    expect(isExperimental(results.controllers.queue_responsive?.status)).toBe(false)
  })
  it('flags every trained model that failed the gate as experimental, with its failed checks', () => {
    for (const id of ['ml_forecast', 'rl_ppo'] as ControllerId[]) {
      const c = results.controllers[id]
      if (!c) continue
      expect(c.gate).toBeDefined()
      if (c.gate!.failed.length) expect(c.status).toBe('experimental')
    }
  })
  it('pairs every controller run with a fixed-time run on the same demand', () => {
    const r = results.results
    const p = pairs(r, 'fixed_time', 'queue_responsive')
    expect(p.length).toBe(r.pairs)
    expect(r.vsFixed.overall.queue_responsive?.metrics.mean_wait_s?.vs?.paired).toBe(p.length)
  })
  it('agrees with the stored win count', () => {
    const r = results.results
    const wins = pairs(r, 'fixed_time', 'queue_responsive').filter((x) => x.b.wait < x.a.wait).length
    expect(wins).toBe(r.vsFixed.overall.queue_responsive?.metrics.mean_wait_s?.vs?.better)
  })
  it('holds one scenario out', () => {
    expect(results.scenarios.filter((s) => s.heldOut).map((s) => s.id)).toEqual(['stadium'])
  })
})
