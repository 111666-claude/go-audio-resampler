// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package align

import (
	"math"
	"testing"
)

// delaySignal returns src delayed by d samples (positive retards), via the
// closed-form tone construction for tone signals.
func shiftInteger(x []float64, d int) []float64 {
	y := make([]float64, len(x))
	for i := range y {
		if j := i - d; j >= 0 && j < len(x) {
			y[i] = x[j]
		}
	}
	return y
}

func newTestSession(t *testing.T, opts SessionOptions) *Session {
	t.Helper()
	s, err := NewSession(opts)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	return s
}

// runSession feeds left/right through the session in chunks of size chunk,
// returning both aligned outputs and the final estimate.
func runSession(t *testing.T, s *Session, left, right []float64, chunk int) (outL, outR []float64, est Estimate) {
	t.Helper()
	for i := 0; i < len(left); i += chunk {
		end := min(i+chunk, len(left))
		ol, or, e, err := s.Process(left[i:end], right[i:end])
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		outL = append(outL, ol...)
		outR = append(outR, or...)
		est = e
	}
	return outL, outR, est
}

func TestSession_ChunkingInvariance(t *testing.T) {
	const (
		n    = 48000
		rate = 48000.0
	)
	left := musicLike(n, 3, rate)
	right := shiftInteger(left, 3)
	for _, chunk := range []int{256, 1000, 4096, 20000} {
		s := newTestSession(t, SessionOptions{MaxDelay: 32, Window: 16384})
		ol, or, _ := runSession(t, s, left, right, chunk)
		if len(ol) != n || len(or) != n {
			t.Fatalf("chunk %d: lengths %d/%d, want %d", chunk, len(ol), len(or), n)
		}
		if chunk == 256 {
			continue
		}
		// Compare against the chunk-256 run.
		ref := newTestSession(t, SessionOptions{MaxDelay: 32, Window: 16384})
		rl, rr, _ := runSession(t, ref, left, right, 256)
		for i := range ol {
			if math.Abs(ol[i]-rl[i]) > 1e-12 {
				t.Fatalf("chunk %d: left sample %d = %v, want %v", chunk, i, ol[i], rl[i])
			}
			if math.Abs(or[i]-rr[i]) > 1e-12 {
				t.Fatalf("chunk %d: right sample %d = %v, want %v", chunk, i, or[i], rr[i])
			}
		}
	}
}

func TestSession_FindsIntegerDelay(t *testing.T) {
	const n = 48000
	left := musicLike(n, 1, 48000)
	for _, d := range []int{0, 1, 5, 17} {
		right := shiftInteger(left, d)
		s := newTestSession(t, SessionOptions{MaxDelay: 32, Window: 16384})
		_, _, est := runSession(t, s, left, right, 4096)
		if !s.HaveEst {
			t.Fatalf("delay %d: no estimate", d)
		}
		if math.Abs(est.Delay-float64(d)) > 0.01 {
			t.Errorf("delay %d: estimate %v", d, est.Delay)
		}
		if !est.Valid() {
			t.Errorf("delay %d: estimate not Valid: %+v", d, est)
		}
		// The estimate must also be the delay the aligners actually apply.
		if math.Abs(s.Delay()-float64(d)) > 1.0/(2*subPhases) {
			t.Errorf("delay %d: applied delay %v", d, s.Delay())
		}
	}
}

func TestSession_AlignsFractionalDelay(t *testing.T) {
	// The fractional shift is constructed in closed form, then the session is
	// checked by measuring the residual delay between its two outputs: an
	// aligned pair must be near sample-aligned (the estimator's own error is
	// below 0.002 samples) and the two outputs must correlate above 0.99.
	const n = 131072
	freqs := []float64{110, 220, 440, 880, 1760, 3520, 7040}
	amps := []float64{0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1}
	const d = 3.37
	left := toneSignal(n, freqs, amps, 48000, 0)
	right := toneSignal(n, freqs, amps, 48000, d)
	s := newTestSession(t, SessionOptions{MaxDelay: 32, Window: 16384})
	ol, or, est := runSession(t, s, left, right, 4096)
	if math.Abs(est.Delay-d) > 0.002 {
		t.Errorf("estimate %v, want %v", est.Delay, d)
	}
	// Measure residual delay on the middle half through the same estimator.
	mid := 32768
	res, err := EstimateDelay(ol[mid:mid+32768], or[mid:mid+32768], Config{MaxDelay: 16})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.Delay) > 0.01 {
		t.Errorf("residual delay %v, want ~0", res.Delay)
	}
	if res.Confidence < 0.99 {
		t.Errorf("aligned confidence %v, want >= 0.99", res.Confidence)
	}
}

func TestSession_SilenceHoldsAlignment(t *testing.T) {
	const n = 48000
	left := musicLike(n, 2, 48000)
	right := shiftInteger(left, 4)
	s := newTestSession(t, SessionOptions{MaxDelay: 32, Window: 16384})
	runSession(t, s, left[:30000], right[:30000], 4096)
	if !s.HaveEst || math.Abs(s.est.Delay-4) > 0.01 {
		t.Fatalf("warm-up did not find the delay: %+v", s.est)
	}
	before := s.est.Delay
	// Feed a long silent block with dither on both channels. A naive
	// estimator would lock onto the noise and move the alignment; the session
	// must keep the previous estimate and its applied delay.
	z := make([]float64, 48000)
	for i := range z {
		z[i] = 1e-6 * math.Sin(float64(i)*0.7)
	}
	_, _, est := runSession(t, s, z, z, 4096)
	if est.Delay != before {
		t.Errorf("silence moved the delay from %v to %v", before, est.Delay)
	}
	if s.Delay() != before {
		t.Errorf("applied delay changed to %v, want %v", s.Delay(), before)
	}
}

func TestSession_RejectsBadInput(t *testing.T) {
	s := newTestSession(t, SessionOptions{MaxDelay: 32, Window: 4096})
	x := make([]float64, 100)
	if _, _, _, err := s.Process(x, make([]float64, 50)); err == nil {
		t.Error("accepted mismatched chunk lengths")
	}
	bad := make([]float64, 100)
	bad[10] = math.NaN()
	if _, _, _, err := s.Process(bad, x); err == nil {
		t.Error("accepted a NaN sample")
	}
	bad[10] = math.Inf(1)
	if _, _, _, err := s.Process(x, bad); err == nil {
		t.Error("accepted an infinite sample")
	}
	dst := make([]float64, 50)
	if _, err := s.ProcessInto(x, x, dst, dst); err == nil {
		t.Error("ProcessInto accepted undersized buffers")
	}
}

func TestSession_BadOptions(t *testing.T) {
	for _, opts := range []SessionOptions{
		{MaxDelay: 0},
		{MaxDelay: -1},
		{MaxDelay: 100, Window: 300},
		{MaxDelay: 8, SilenceRMS: math.NaN()},
	} {
		if _, err := NewSession(opts); err == nil {
			t.Errorf("NewSession(%+v) accepted bad options", opts)
		}
	}
}

func TestSession_OutOfRangeReported(t *testing.T) {
	const n = 48000
	left := musicLike(n, 4, 48000)
	right := shiftInteger(left, 40)
	s := newTestSession(t, SessionOptions{MaxDelay: 8, Window: 8192})
	_, _, est := runSession(t, s, left, right, 4096)
	if !est.OutOfRange {
		t.Errorf("delay 40 with MaxDelay 8: %+v", est)
	}
	if s.HaveEst {
		t.Errorf("reported HaveEst for an out-of-range delay")
	}
}

func TestSession_Freeze(t *testing.T) {
	const n = 96000
	left := musicLike(n, 6, 48000)
	right := shiftInteger(left, 3)
	s := newTestSession(t, SessionOptions{MaxDelay: 32, Window: 16384, Freeze: true})
	runSession(t, s, left[:40000], right[:40000], 4096)
	if !s.HaveEst || math.Abs(s.est.Delay-3) > 0.01 {
		t.Fatalf("freeze warm-up: %+v", s.est)
	}
	// A distant block with a different delay must not move the frozen
	// alignment.
	right2 := shiftInteger(left[40000:], -3)
	_, _, est := runSession(t, s, left[40000:], right2, 4096)
	if math.Abs(est.Delay-3) > 0.01 {
		t.Errorf("frozen session moved to %v, want 3", est.Delay)
	}
}
