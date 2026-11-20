// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

// Package align implements inter-channel delay estimation and fractional
// sample-delay alignment for stereo material that was captured with a
// different path (or a few frames' offset) per microphone.
//
// Two microphones recording the same source are rarely sample-aligned: the
// distance between the capsules, separate converters, or a different preamp
// latency shifts one channel against the other by an integer or fractional
// number of samples. Summing or downmixing such channels comb-filters the
// result: a delay of d samples cancels the shared content at frequencies
// where d*f is a half-integer cycle, and no resampler fixes that because the
// two channels drift apart in time, not in rate. The fix is to estimate d
// and shift one channel by it before any further processing.
//
// The estimator here is a windowed, mean-removed, normalized cross-correlation
// over a bounded lag search. The integer peak is refined to a fraction of a
// sample by sampling the correlation sequence with a band-limited (Lanczos)
// interpolator, which is accurate to well under 0.01 samples on
// broadband/musical material — far below the pitch period of any audible
// tone, so the comb filter is removed rather than merely moved.
package align

import "math"

// Estimate parameters shared by the estimator and the aligner. They are
// constants rather than tunables because the tests pin the estimator's error
// against synthetic and real material, and an exposed knob would let a caller
// silently trade that accuracy away.
const (
	// LanczosA is the half-width (in correlation samples) of the band-limited
	// interpolator used to refine the integer cross-correlation peak. The
	// correlation of a band-limited signal is itself band-limited to the
	// signal's bandwidth, so interpolating it at this width recovers the
	// sub-sample peak to well under 0.01 samples; a parabola through the peak
	// and its two neighbours is biased by up to 0.1 samples for hf material.
	LanczosA = 16

	// refineStep is the grid step of the sub-sample peak search, in samples.
	// 1/1000 sample is two orders of magnitude below the alignment tolerance
	// the tests assert, and the grid search over 1001 points costs nothing
	// next to the O(lags * window) correlation.
	refineStep = 0.001

	// minEstimateSamples is the shortest analysis window the estimator
	// accepts. The lag search needs room on both sides of the window, and a
	// shorter window cannot resolve a sub-sample peak on typical music.
	minEstimateSamples = 256
)

// Config configures a delay estimate.
type Config struct {
	// MaxDelay is the largest inter-channel delay, in samples, that the
	// search considers. The estimate saturates at +/-MaxDelay: a correlation
	// peak at either end of the search range sets OutOfRange, because a peak
	// on the boundary may belong to a true delay beyond it. MaxDelay must be
	// positive.
	MaxDelay int

	// SilenceRMS is the RMS level, in the same units as the samples, below
	// which both channels are treated as silence. An estimate over silence is
	// reported as Silent and the aligner leaves the channels untouched; it
	// never invents a delay from dither or room noise. A value of 0 uses the
	// default (DefaultSilenceRMS).
	SilenceRMS float64
}

// DefaultSilenceRMS is the silence threshold used when Config.SilenceRMS is
// zero: 1e-4 is about -80 dBFS, below the quietest dither and well below any
// audible content, so a genuinely silent passage is recognised while a very
// quiet signal is still aligned on its own merits.
const DefaultSilenceRMS = 1e-4

// Estimate is the result of a delay estimate between two channels.
type Estimate struct {
	// Delay is the delay of Right relative to Left, in samples: a positive
	// value means Right lags Left and must be advanced by Delay to align, a
	// negative value means Right leads and must be delayed. The value is
	// fractional unless the material happens to be integer-aligned.
	Delay float64

	// Confidence is the absolute normalized cross-correlation at the chosen
	// delay, in [0, 1]. It measures how much of the two channels' energy is
	// shared coherently at that delay: 1 is an identical (or exactly
	// inverted) signal, 0.5 is two microphones with substantially different
	// room sound, and near 0 is uncorrelated material. Callers should treat a
	// low value (below about 0.2) as "no reliable alignment" rather than
	// trusting the reported Delay.
	Confidence float64

	// Polarity is +1 for in-phase channels and -1 for inverted channels: the
	// correlation peak is negative, meaning one channel's signal is the
	// other's with opposite sign (a reversed cable or a polarity-flipped
	// capsule). Alignment shifts the channels but does not invert them; the
	// caller decides whether to correct the polarity.
	Polarity float64

	// Silent reports that both channels were below the silence threshold over
	// the analysis window. Delay, Confidence and Polarity are then zero, and
	// the aligner leaves both channels untouched.
	Silent bool

	// OutOfRange reports that the best correlation lay at the edge of the lag
	// search, or that a stronger correlation was found just past it, so the
	// true delay may be larger than MaxDelay. The reported Delay must not be
	// applied; the caller should widen MaxDelay, supply more material, or
	// leave the channels alone.
	OutOfRange bool
}

// Valid reports whether the estimate is safe to apply: not silent, not on the
// lag search boundary, and above the confidence floor.
func (e Estimate) Valid() bool {
	return !e.Silent && !e.OutOfRange && e.Confidence >= MinConfidence
}

// MinConfidence is the confidence below which an estimate is considered
// unreliable. At 0.2 only a fifth of the energy is coherently shared, which
// on real material means the channels do not contain the same source and a
// shift would be fitting noise; measured on uncorrelated dither the
// correlation peak is below 0.1 for every window the estimator accepts.
const MinConfidence = 0.2

// EstimateDelay estimates the delay of right relative to left over a window
// of at most win samples and a lag search of +/-cfg.MaxDelay. win only
// selects how much material is available: the estimator always uses full
// overlap minus the lag margin, so a caller can pass a growing ring buffer
// without changing the result for a given prefix.
//
// It returns an error only for a structurally impossible request (a
// non-positive MaxDelay, or too few samples to analyse). A silent window, an
// uncorrelated pair or a delay at the search boundary is not an error: those
// are reported in the Estimate so the streaming caller can keep its previous
// alignment state instead of failing the stream.
func EstimateDelay(left, right []float64, cfg Config) (Estimate, error) {
	if cfg.MaxDelay <= 0 {
		return Estimate{}, errBadMaxDelay
	}
	if cfg.SilenceRMS < 0 || math.IsNaN(cfg.SilenceRMS) {
		return Estimate{}, errBadSilence
	}
	n := min(len(left), len(right))
	if n < minEstimateSamples {
		return Estimate{}, ErrShortWindow
	}
	// The lag search shifts one channel by up to MaxDelay against the other,
	// so it needs that many samples of overlap margin at each end to stay
	// unbiased. A window that is mostly margin cannot support the search.
	margin := cfg.MaxDelay
	if n <= 2*margin+minEstimateSamples {
		return Estimate{}, ErrShortWindow
	}

	silence := cfg.SilenceRMS
	if silence == 0 {
		silence = DefaultSilenceRMS
	}

	lo, hi := margin, n-margin
	count := hi - lo

	var meanL, meanR, powL, powR float64
	for i := lo; i < hi; i++ {
		meanL += left[i]
		meanR += right[i]
		powL += left[i] * left[i]
		powR += right[i] * right[i]
	}
	inv := 1 / float64(count)
	meanL *= inv
	meanR *= inv
	// Variance, not mean square: a channel with a large DC offset but no
	// signal is silence for alignment purposes.
	varL := powL*inv - meanL*meanL
	varR := powR*inv - meanR*meanR
	if varL < 0 {
		varL = 0
	}
	if varR < 0 {
		varR = 0
	}
	if math.Sqrt(varL) < silence || math.Sqrt(varR) < silence {
		return Estimate{Silent: true}, nil
	}

	// Hann-window the analysis region to keep the correlation from leaking
	// across the lag range; the window is applied over the full extent so the
	// same taper multiplies every lag of the search.
	xw := make([]float64, n)
	yw := make([]float64, n)
	for i := range xw {
		w := hann(i, n)
		xw[i] = (left[i] - meanL) * w
		yw[i] = (right[i] - meanR) * w
	}
	var energyL float64
	for i := lo; i < hi; i++ {
		energyL += xw[i] * xw[i]
	}

	maxLag := cfg.MaxDelay
	corr := make([]float64, 2*maxLag+1)
	best := -1.0
	bestLag := 0
	for lag := -maxLag; lag <= maxLag; lag++ {
		var cross, energyR float64
		for i := lo; i < hi; i++ {
			b := yw[i+lag]
			cross += xw[i] * b
			energyR += b * b
		}
		if energyL <= 0 || energyR <= 0 {
			continue
		}
		// Positively expressed so a NaN from a denormal-free zero energy
		// channel cannot slip past the comparison and poison the peak search.
		r := cross / math.Sqrt(energyL*energyR)
		if !(math.Abs(r) >= 0) { // NaN guard
			continue
		}
		corr[lag+maxLag] = r
		if math.Abs(r) > best {
			best = math.Abs(r)
			bestLag = lag
		}
	}
	if best < 0 {
		// Every lag had zero energy on one side: the channels are constant.
		return Estimate{Silent: true}, nil
	}

	est := Estimate{
		Delay:      float64(bestLag),
		Confidence: best,
		Polarity:   1,
	}
	if corr[bestLag+maxLag] < 0 {
		est.Polarity = -1
	}
	if bestLag == maxLag || bestLag == -maxLag {
		// The peak sits on the search boundary. The true peak may be outside
		// the range, so the delay is not usable; report it as out of range
		// instead of applying a truncated shift.
		est.OutOfRange = true
		return est, nil
	}
	// Probe a short distance past the advertised range: if the correlation
	// there is stronger than the best in-range peak, the true delay lies
	// beyond MaxDelay and the in-range peak is a secondary lobe. This catches
	// a delay just past the bound, which otherwise reports a plausible-looking
	// but wrong delay; a delay far beyond it can alias into the range and
	// cannot be distinguished from in-range content, which is what the
	// confidence value is for.
	if beyondStronger(xw, yw, lo, hi, best, maxLag) {
		est.OutOfRange = true
		return est, nil
	}

	// Refine to a fraction of a sample. The correlation sequence is itself
	// band-limited (it is the product of the two signals' spectra), so a
	// windowed-sinc interpolant through 32 samples around the integer peak
	// locates the true maximum of the continuous correlation to far better
	// than one sample. Keeping the search within +/-0.5 of the integer peak is
	// correct because the parabolic envelope of a correlation peak of a
	// band-limited signal is (to first order) symmetric around its maximum.
	center := float64(bestLag + maxLag)
	frac := 0.0
	peak := best
	for s := -0.5; s <= 0.5+refineStep/2; s += refineStep {
		v := math.Abs(lanczosAt(corr, center+s, LanczosA))
		if v > peak {
			peak = v
			frac = s
		}
	}
	est.Delay = float64(bestLag) + frac
	est.Confidence = peak
	// A refined delay that rounds onto the boundary is equally unusable.
	if math.Abs(est.Delay) >= float64(maxLag) {
		est.OutOfRange = true
	}
	return est, nil
}

// beyondProbe is how many lags past each search bound are checked for a
// stronger correlation than the best in-range peak. 8 samples is enough to
// catch a delay just past the bound while costing 16 extra correlations.
const beyondProbe = 8

// beyondStronger reports whether the normalized correlation at any lag in
// (maxLag, maxLag+beyondProbe] (or the mirror on the negative side) exceeds
// best. Each probe lag uses its own overlap window and normalises by the
// energy in that window, so the comparison is fair despite the shrinking
// window.
func beyondStronger(xw, yw []float64, lo, hi int, best float64, maxLag int) bool {
	n := len(xw)
	for _, side := range []int{-1, 1} {
		for k := 1; k <= beyondProbe; k++ {
			lag := side * (maxLag + k)
			from := max(lo, -lag)
			to := min(hi, n-lag)
			if to-from < minEstimateSamples/4 {
				continue
			}
			var cross, energyL, energyR float64
			for i := from; i < to; i++ {
				b := yw[i+lag]
				cross += xw[i] * b
				energyL += xw[i] * xw[i]
				energyR += b * b
			}
			if energyL <= 0 || energyR <= 0 {
				continue
			}
			if math.Abs(cross/math.Sqrt(energyL*energyR)) > best {
				return true
			}
		}
	}
	return false
}

// hann returns the n-point Hann window value at index i.
func hann(i, n int) float64 {
	if n <= 1 {
		return 1
	}
	return 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1))
}

// lanczosAt evaluates the Lanczos-a interpolant of v (indexed 0..len(v)-1) at
// fractional position pos, clamping the kernel support to the array so the
// sum stays defined near the array ends. Outside the array it returns the
// end value rather than extrapolating.
func lanczosAt(v []float64, pos float64, a int) float64 {
	k0 := int(math.Floor(pos))
	if k0 < 0 || k0 >= len(v)-1 {
		// Do not extrapolate: the refinement search never leaves the interior
		// because the integer peak is strictly inside the lag range, but the
		// kernel support at the very edge would otherwise read past it.
		if k0 < 0 {
			k0 = 0
		}
		if k0 >= len(v) {
			k0 = len(v) - 1
		}
	}
	var num, den float64
	for j := 1 - a; j <= a; j++ {
		k := k0 + j
		if k < 0 || k >= len(v) {
			continue
		}
		u := pos - float64(k)
		if math.Abs(u) < 1e-12 {
			return v[k]
		}
		w := math.Sin(math.Pi*u) / (math.Pi * u)
		ww := math.Sin(math.Pi*u/float64(a)) / (math.Pi * u / float64(a))
		w *= ww
		num += v[k] * w
		den += w
	}
	if den == 0 {
		return v[k0]
	}
	return num / den
}
