import { describe, expect, it } from 'vitest'
import { isAmbulance, normalise, sideId, spriteCode, type DemoAsset } from '../lib/flowstateDemo'

const placed = (type: number) => ({ id: 1, x: 0, y: 0, angle: 0, speed: 0, type })

describe('schema 2 (district) assets', () => {
  it('uses main sprite codes directly and recognises the ambulance', () => {
    expect(spriteCode(placed(5), true)).toBe(5)
    expect(isAmbulance(placed(5), true)).toBe(true)
    expect(isAmbulance(placed(2), true)).toBe(false)
    expect(spriteCode(placed(2), false)).toBe(5) // lab code 2 = ambulance
  })
  it('reads per-side ambulance ids', () => {
    const u = { id: 5, left: { id: 5 }, right: { id: 50 } } as never
    expect(sideId(u, 'left')).toBe(5)
    expect(sideId(u, 'right')).toBe(50)
    expect(sideId({ id: 9, left: {}, right: {} } as never, 'right')).toBe(9)
  })
  it('maps main car metrics onto the page fields without overwriting existing ones', () => {
    const asset = {
      totals: { left: { car: { waiting: 25, travelTime: 137 } }, right: { car: { waiting: 20, travelTime: 132 }, mean_wait_civilian_s: 1 } },
      aggregate: { metrics: { car_waiting: { mean_diff: -4 }, mean_travel_s: { mean_diff: 0 } } },
    } as unknown as DemoAsset
    normalise(asset)
    expect(asset.totals.left.mean_wait_civilian_s).toBe(25)
    expect(asset.totals.right.mean_wait_civilian_s).toBe(1)
    expect(asset.aggregate.metrics.mean_wait_civilian_s.mean_diff).toBe(-4)
    expect(asset.aggregate.metrics.car_travelTime).toBeUndefined()
  })
})

describe('schema 4 (controller choice)', () => {
  it('sends controller only when it is not the default', async () => {
    const { replayBody } = await import('../lib/flowstateDemo')
    expect(replayBody('rush_hour', null)).toEqual({ scenario: 'rush_hour', seed: null })
    expect(replayBody('rush_hour', null, 'queue_responsive', 'queue_responsive')).toEqual({ scenario: 'rush_hour', seed: null })
    expect(replayBody('rush_hour', 3, 'rl_ppo', 'queue_responsive')).toEqual({ scenario: 'rush_hour', seed: 3, controller: 'rl_ppo' })
  })
  it('labels the right pane; schema 3 assets are the FlowState rules', async () => {
    const { rightPane } = await import('../lib/flowstateDemo')
    expect(rightPane({})).toEqual({ id: 'queue_responsive', name: 'FlowState', experimental: false, fallback: null })
    const p = rightPane({ rightController: 'rl_ppo', rightStatus: 'experimental', fallback: null }, { rl_ppo: 'Learned policy (PPO)' })
    expect(p).toMatchObject({ name: 'Learned policy (PPO)', experimental: true })
    expect(rightPane({ rightController: 'queue_responsive', rightStatus: 'default', fallback: 'no trained model on this installation' }).fallback).toMatch(/no trained model/)
  })
})
