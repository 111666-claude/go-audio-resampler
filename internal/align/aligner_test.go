// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package align

import (
	"math"
	"testing"
)

// sineSignal returns a unit-amplitude sine at f Hz sampled at rate.
func sineSignal(n int, f, rate float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = math.Sin(2 * math.Pi * f * float64(i) / rate)
	}
	return x
}

// phaseDelay fits the sine of frequency f (sampled at rate) to y over its
// middle half and returns the delay in samples and the fitted amplitude.
func phaseDelay(y []float64, f, rate float64) (delay, amp float64) {
	w := 2 * math.Pi * f / rate
	lo, hi := len(y)/4, 3*len(y)/4
	var ss, cc, sc, ys, yc float64
	for m := lo; m < hi; m++ {
		s, c := math.Sin(w*float64(m)), math.Cos(w*float64(m))
		ss += s * s
		cc += c * c
		sc += s * c
		ys += y[m] * s
		yc += y[m] * c
	}
	det := ss*cc - sc*sc
	a := (ys*cc - yc*sc) / det
	b := (yc*ss - ys*sc) / det
	return -math.Atan2(b, a) / w, math.Hypot(a, b)
}

// chunked runs x through a fresh aligner in chunks of size chunk and returns
// the concatenated output.
func chunked(t *testing.T, a *Aligner, x []float64, chunk int) []float64 {
	t.Helper()
	out := make([]float64, 0, len(x))
	for i := 0; i < len(x); i += chunk {
		end := min(i+chunk, len(x))
		out = append(out, a.Process(x[i:end])...)
	}
	return out
}

func TestAligner_DelayAccuracy(t *testing.T) {
	const (
		rate = 48000.0
		n    = 8192
	)
	// 997 Hz is not commensurate with the sample rate, so the fitted phase
	// delay is unambiguous modulo its period and a wrong delay cannot alias
	// onto the right one.
	const toneHz = 997.0
	x := sineSignal(n, toneHz, rate)
	for _, d := range []float64{64, 64.25, 64.5, 65, 67.37, 76.5, 100.75} {
		a, err := NewAligner(d, 128)
		if err != nil {
			t.Fatalf("NewAligner(%v): %v", d, err)
		}
		y := chunked(t, a, x, len(x))
		got, amp := phaseDelay(y, toneHz, rate)
		// The fitted phase delay over the middle half equals the requested
		// delay; the only residual is the 256-phase quantisation, at most
		// 1/512 sample.
		// The sinusoid is periodic at 1000 Hz, so the fitted phase delay is
		// only defined modulo the 48-sample period; compare on the circle.
		period := rate / toneHz
		residual := math.Mod(got-d, period)
		if residual > period/2 {
			residual -= period
		}
		if residual < -period/2 {
			residual += period
		}
		if math.Abs(residual) > 1.0/(2*subPhases)+1e-6 {
			t.Errorf("delay %v: applied phase delay %v, residual %v samples", d, got, residual)
		}
		if math.Abs(amp-1) > 2e-3 {
			t.Errorf("delay %v: amplitude %v, want 1", d, amp)
		}
	}
}

func TestAligner_ChunkingInvariance(t *testing.T) {
	x := sineSignal(6000, 1000, 48000)
	// Delays at or above HalfTaps are causal by construction.
	for _, d := range []float64{64.13, 67.37, 72.5, 100.9} {
		a, err := NewAligner(d, 128)
		if err != nil {
			t.Fatal(err)
		}
		ref := chunked(t, a, x, len(x))
		for _, chunk := range []int{1, 7, 100, 4096} {
			b, err := NewAligner(d, 128)
			if err != nil {
				t.Fatal(err)
			}
			got := chunked(t, b, x, chunk)
			if len(got) != len(ref) {
				t.Fatalf("delay %v chunk %d: length %d != %d", d, chunk, len(got), len(ref))
			}
			for i := range got {
				if math.Abs(got[i]-ref[i]) > 1e-12 {
					t.Fatalf("delay %v chunk %d: sample %d = %v, want %v", d, chunk, i, got[i], ref[i])
				}
			}
		}
	}
}

func TestAligner_IntegerDelayIsExact(t *testing.T) {
	// An integer delay selects the frac=0 kernel, which is a windowed sinc
	// sampled exactly at its zeros: a delta at the kernel centre. The output
	// must therefore be the input delayed by exactly that many samples, with
	// the leading positions zero, to floating-point round-off.
	const d = int(HalfTaps) + 7
	x := sineSignal(1024, 997, 48000)
	a, err := NewAligner(float64(d), 256)
	if err != nil {
		t.Fatal(err)
	}
	y := chunked(t, a, x, 100)
	if len(y) != len(x) {
		t.Fatalf("length %d != %d", len(y), len(x))
	}
	for i := range y {
		want := 0.0
		if i >= d {
			want = x[i-d]
		}
		if math.Abs(y[i]-want) > 1e-9 {
			t.Fatalf("sample %d = %v, want %v", i, y[i], want)
		}
	}
}

func TestAligner_RejectsBadDelay(t *testing.T) {
	for _, d := range []float64{math.NaN(), math.Inf(1), math.Inf(-1), 200, -200, 0, 63.9} {
		if _, err := NewAligner(d, 128); err == nil {
			t.Errorf("NewAligner(%v) accepted a bad delay", d)
		}
	}
}
