# Canonical Metric Specifications (v1)

This benchmark treats metric definitions as part of the contract.

## Common conventions
- Signal: mono float32/float64 numpy arrays
- Sample rate: `fs` from metadata if available, else 48000 Hz
- Time zero: direct arrival at the maximum absolute sample (configurable, default used)

## Schroeder EDC
- Energy: `e[n] = x[n]^2`
- Cumulative tail energy: `E_tail[n] = sum_{k=n..N-1} e[k]`
- Normalize by `E_tail[0]`
- Convert to dB: `10*log10(E_tail + eps)`

## RT60
We estimate decay slope in dB using least squares over a selected range:
- Prefer T30 if EDC crosses -35 dB:
  - Fit range: [-5 dB, -35 dB]
- Else fall back to T20 if EDC crosses -25 dB:
  - Fit range: [-5 dB, -25 dB]
- Else mark RT60 as NaN

Let slope be `m` in dB per second (negative). Extrapolate to -60 dB:
- `rt60 = -60 / m`

## EDT
- Fit range: [0 dB, -10 dB] using EDC
- `edt = -60 / m_0_10`

## Direct-to-reverberant ratio (DRR)
We define direct window relative to direct index `n0`:
- Direct window: `[n0 - pre_ms, n0 + direct_ms]`
- Default: `pre_ms=1.0`, `direct_ms=2.5`
- Reverberant window: remainder after direct window to end
- DRR(dB) = 10*log10(Edirect / Ereverb)

## Clarity C50/C80
Using same direct index `n0`:
- C50: early window length 50 ms after `n0` (including `n0`)
- C80: early window length 80 ms after `n0`
- Ratio: `10*log10(Eearly / Elate)`

## Ts (center time)
- `ts = sum(t * e(t)) / sum(e(t))` with `t= n/fs`

## Edge cases
- If energy sums are zero, return NaN.
- We use eps = 1e-12 for log stability.

