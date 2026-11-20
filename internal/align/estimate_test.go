// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package align

import (
	"math"
	"math/rand"
	"testing"
)

// musicLike returns a harmonically rich, deterministic test signal: a chord
// with an amplitude envelope and two high partials. It is not a pure tone
// because a pure tone's correlation is periodic and would make a wrong delay
// indistinguishable from the right one; the mixture has a single sharp peak.
func musicLike(n int, seed int64, rate float64) []float64 {
	freqs := []float64{110, 220, 277.18, 329.63, 440, 554.37, 880, 1760, 3520, 7040, 10560}
	amps := []float64{0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.07, 0.05, 0.03, 0.02}
	rng := rand.New(rand.NewSource(seed))
	x := make([]float64, n)
	for i := range x {
		t := float64(i) / rate
		var s float64
		for j, f := range freqs {
			s += amps[j] * math.Sin(2*math.Pi*f*t+0.37*float64(j))
		}
		env := 0.6 + 0.4*math.Sin(2*math.Pi*0.37*t)
		x[i] = s*env*0.4 + 1e-3*rng.NormFloat64()
	}
	return x
}

// bandLimitedSignal returns a signal band-limited to 0.4 of Nyquist, so a
// fractional shift of it can be constructed exactly in the closed form used
// by shiftTones: each tone is sampled at a shifted time.
func toneSignal(n int, freqs, amps []float64, rate, delay float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		t := (float64(i) - delay) / rate
		var s float64
		for j, f := range freqs {
			s += amps[j] * math.Sin(2*math.Pi*f*t+0.37*float64(j))
		}
		x[i] = s
	}
	return x
}

var (
	toneFreqs = []float64{110, 220, 440, 880, 1760, 3520, 7040, 10560, 15000, 19000}
	toneAmps  = []float64{0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.03}
)

func TestEstimateDelay_FractionalAccuracy(t *testing.T) {
	const (
		n    = 16384
		rate = 48000.0
	)
	left := toneSignal(n, toneFreqs, toneAmps, rate, 0)
	for _, d := range []float64{0, 0.13, 0.25, 0.5, 0.75, 0.9, 3, 3.37, 7.9, -4.62, 12.5, 31.4, -40.2} {
		right := toneSignal(n, toneFreqs, toneAmps, rate, d)
		est, err := EstimateDelay(left, right, Config{MaxDelay: 128})
		if err != nil {
			t.Fatalf("delay %v: %v", d, err)
		}
		if est.Silent || est.OutOfRange {
			t.Fatalf("delay %v: silent=%v outOfRange=%v", d, est.Silent, est.OutOfRange)
		}
		if math.Abs(est.Delay-d) > 0.002 {
			t.Errorf("delay %v: estimate %v, error %v samples", d, est.Delay, est.Delay-d)
		}
		if est.Confidence < 0.99 {
			t.Errorf("delay %v: confidence %v, want >= 0.99 for the same source", d, est.Confidence)
		}
		if est.Polarity != 1 {
			t.Errorf("delay %v: polarity %v, want +1", d, est.Polarity)
		}
	}
}

func TestEstimateDelay_RealMaterialError(t *testing.T) {
	// Music-like material with 1e-6 of noise, shifted with a band-limited
	// construction, is the closest committed stand-in for a real stereo pair.
	// The error must stay far below the pitch period of the highest partial
	// (10560 Hz is 4.5 samples at 48k) so the comb filter is removed rather
	// than moved.
	const (
		n    = 32768
		rate = 48000.0
	)
	freqs := []float64{110, 220, 277.18, 329.63, 440, 554.37, 880, 1760, 3520, 7040, 10560}
	amps := []float64{0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.07, 0.05, 0.03, 0.02}
	left := toneSignal(n, freqs, amps, rate, 0)
	for _, d := range []float64{0.37, 3.37, 7.9, -4.62, 0, 12.5} {
		right := toneSignal(n, freqs, amps, rate, d)
		est, err := EstimateDelay(left, right, Config{MaxDelay: 128})
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(est.Delay-d) > 0.002 {
			t.Errorf("delay %v: estimate %v, error %v", d, est.Delay, est.Delay-d)
		}
	}
}

func TestEstimateDelay_InvertedPolarity(t *testing.T) {
	const n = 16384
	left := toneSignal(n, toneFreqs, toneAmps, 48000, 0)
	right := toneSignal(n, toneFreqs, toneAmps, 48000, 3.37)
	for i := range right {
		right[i] = -right[i]
	}
	est, err := EstimateDelay(left, right, Config{MaxDelay: 128})
	if err != nil {
		t.Fatal(err)
	}
	if est.Polarity != -1 {
		t.Errorf("polarity %v, want -1", est.Polarity)
	}
	if math.Abs(est.Delay-3.37) > 0.002 {
		t.Errorf("delay %v, want 3.37", est.Delay)
	}
	if est.Confidence < 0.99 {
		t.Errorf("confidence %v, want >= 0.99", est.Confidence)
	}
}

func TestEstimateDelay_Silence(t *testing.T) {
	// True silence, low-level dither, and a quiet-but-real signal. The first
	// two are silent (the estimator must not invent a delay from noise); the
	// third is not, because 1e-3 RMS is above the -80 dBFS floor.
	z := make([]float64, 16384)
	est, err := EstimateDelay(z, z, Config{MaxDelay: 128})
	if err != nil {
		t.Fatal(err)
	}
	if !est.Silent || est.Delay != 0 || est.Confidence != 0 || est.Polarity != 0 {
		t.Errorf("true silence: %+v", est)
	}

	rng := rand.New(rand.NewSource(7))
	dl := make([]float64, 16384)
	dr := make([]float64, 16384)
	for i := range dl {
		dl[i] = 1e-5 * rng.NormFloat64()
		dr[i] = 1e-5 * rng.NormFloat64()
	}
	est, err = EstimateDelay(dl, dr, Config{MaxDelay: 128})
	if err != nil {
		t.Fatal(err)
	}
	if !est.Silent {
		t.Errorf("dither below the silence floor: %+v", est)
	}

	// A DC-only pair is constant, hence silent for alignment even though its
	// RMS is large.
	dcL := make([]float64, 16384)
	dcR := make([]float64, 16384)
	for i := range dcL {
		dcL[i] = 0.5
		dcR[i] = 0.5
	}
	est, err = EstimateDelay(dcL, dcR, Config{MaxDelay: 128})
	if err != nil {
		t.Fatal(err)
	}
	if !est.Silent {
		t.Errorf("constant channels: %+v", est)
	}
}

func TestEstimateDelay_OutOfRange(t *testing.T) {
	// A delay just past the search bound shows a strong correlation on the
	// boundary, and the probe past it confirms the true peak lies outside the
	// range. A delay far outside the range has no physically meaningful
	// in-range peak at all; that case is covered by the confidence floor
	// (TestEstimateDelay_UncorrelatedLowConfidence) and by the documented
	// contract that MaxDelay must cover the largest expected delay.
	const n = 16384
	left := toneSignal(n, toneFreqs, toneAmps, 48000, 0)
	for _, d := range []float64{129, 128.5, -129, 134} {
		right := toneSignal(n, toneFreqs, toneAmps, 48000, d)
		est, err := EstimateDelay(left, right, Config{MaxDelay: 128})
		if err != nil {
			t.Fatal(err)
		}
		if !est.OutOfRange {
			t.Errorf("delay %v with MaxDelay 128: not flagged out of range (%+v)", d, est)
		}
		if est.Valid() {
			t.Errorf("delay %v: reported Valid despite being out of range", d)
		}
	}
}

func TestEstimateDelay_UncorrelatedLowConfidence(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	left := musicLike(16384, 1, 48000)
	right := make([]float64, len(left))
	for i := range right {
		right[i] = 0.1 * rng.NormFloat64()
	}
	est, err := EstimateDelay(left, right, Config{MaxDelay: 128})
	if err != nil {
		t.Fatal(err)
	}
	if est.Confidence > 0.1 {
		t.Errorf("uncorrelated channels: confidence %v, want <= 0.1", est.Confidence)
	}
	if est.Valid() {
		t.Errorf("uncorrelated channels reported Valid: %+v", est)
	}
}

func TestEstimateDelay_NoiseRobustness(t *testing.T) {
	const n = 16384
	left := toneSignal(n, toneFreqs, toneAmps, 48000, 0)
	var power float64
	for _, v := range left {
		power += v * v
	}
	power /= float64(n)
	for _, snr := range []float64{40, 20, 10} {
		rng := rand.New(rand.NewSource(5))
		right := toneSignal(n, toneFreqs, toneAmps, 48000, -7.6)
		np := power / math.Pow(10, snr/10)
		for i := range right {
			right[i] += math.Sqrt(np) * rng.NormFloat64()
		}
		est, err := EstimateDelay(left, right, Config{MaxDelay: 128})
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(est.Delay-(-7.6)) > 0.02 {
			t.Errorf("snr %v dB: estimate %v, want -7.6", snr, est.Delay)
		}
	}
}

func TestEstimateDelay_BadConfig(t *testing.T) {
	x := make([]float64, 1024)
	for i := range x {
		x[i] = math.Sin(float64(i))
	}
	if _, err := EstimateDelay(x, x, Config{MaxDelay: 0}); err == nil {
		t.Error("MaxDelay 0 accepted")
	}
	if _, err := EstimateDelay(x, x, Config{MaxDelay: -5}); err == nil {
		t.Error("negative MaxDelay accepted")
	}
	if _, err := EstimateDelay(x, x, Config{MaxDelay: 8, SilenceRMS: math.NaN()}); err == nil {
		t.Error("NaN silence threshold accepted")
	}
	short := make([]float64, 100)
	if _, err := EstimateDelay(short, short, Config{MaxDelay: 8}); err == nil {
		t.Error("100-sample window accepted")
	}
}
