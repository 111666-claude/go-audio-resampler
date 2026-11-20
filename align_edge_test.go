// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"errors"
	"math"
	"testing"
)

// Edge cases required of the alignment entry points: pure silence, inverted
// signal, invalid rates, non-finite samples, delay out of range, and short
// input tails. Each must give a defined, documented result rather than a
// panic, a hang or a silently wrong alignment.

func TestAlignEdge_PureSilence(t *testing.T) {
	const n = 16384
	z := make([]float64, n)
	cfg := AlignConfig{MaxDelay: 32}

	// Estimate: silent, sentinel error, zero delay.
	res, err := EstimateChannelDelay(z, z, cfg)
	if !errors.Is(err, ErrAlignSilent) {
		t.Errorf("EstimateChannelDelay: %v, want ErrAlignSilent", err)
	}
	if !res.Silent || res.Delay != 0 || res.Confidence != 0 {
		t.Errorf("EstimateChannelDelay result: %+v", res)
	}
	if res.Valid() {
		t.Errorf("silent estimate reported Valid")
	}

	// One-shot: sentinel error, nil outputs, no panic.
	if _, _, _, err := AlignStereo(z, z, cfg); !errors.Is(err, ErrAlignSilent) {
		t.Errorf("AlignStereo: %v, want ErrAlignSilent", err)
	}

	// Streaming: silence from the very first block is not an error; the
	// outputs are the input delayed by the common lateness, and no alignment
	// is claimed.
	s, err := NewAlignedStereo(48000, 48000, QualityHigh, AlignConfig{MaxDelay: 32, Window: 4096})
	if err != nil {
		t.Fatal(err)
	}
	// One window to produce the first analysis, which is silent.
	outL, outR, res, err := s.Process(z, z)
	if err != nil {
		t.Fatalf("streaming silence returned error: %v", err)
	}
	if !res.Silent {
		t.Errorf("streaming silence not reported: %+v", res)
	}
	if _, ok := s.Alignment(); ok {
		t.Errorf("Alignment() reported a usable estimate for pure silence")
	}
	if len(outL) != n || len(outR) != n {
		t.Errorf("silent output lengths %d/%d, want %d", len(outL), len(outR), n)
	}
	for _, v := range outL {
		if v != 0 {
			t.Fatalf("silent input produced non-zero aligned output %v", v)
		}
	}
}

func TestAlignEdge_InvertedSignal(t *testing.T) {
	const n = 32768
	left := syntheticMusic(n, 48000, 0)
	right := syntheticMusic(n, 48000, 2.5)
	for i := range right {
		right[i] = -right[i]
	}
	res, err := EstimateChannelDelay(left, right, AlignConfig{MaxDelay: 32})
	if err != nil {
		t.Fatalf("inverted pair: %v", err)
	}
	if res.Polarity != -1 {
		t.Errorf("polarity %v, want -1", res.Polarity)
	}
	if math.Abs(res.Delay-2.5) > 0.01 {
		t.Errorf("delay %v, want 2.5", res.Delay)
	}
	// AlignStereo keeps the inversion; the aligned pair still correlates at
	// -1 at lag zero.
	al, ar, res, err := AlignStereo(left, right, AlignConfig{MaxDelay: 32})
	if err != nil {
		t.Fatal(err)
	}
	if res.Polarity != -1 {
		t.Errorf("AlignStereo polarity %v, want -1", res.Polarity)
	}
	var sxy, sxx, syy float64
	for i := n / 4; i < 3*n/4; i++ {
		sxy += al[i] * ar[i]
		sxx += al[i] * al[i]
		syy += ar[i] * ar[i]
	}
	if c := sxy / math.Sqrt(sxx*syy); c > -0.99 {
		t.Errorf("aligned inverted pair correlation %v, want near -1", c)
	}
	// An exactly inverted silence-free pair at zero delay is still -1.
	right0 := syntheticMusic(n, 48000, 0)
	for i := range right0 {
		right0[i] = -left[i]
	}
	if res, err := EstimateChannelDelay(left, right0, AlignConfig{MaxDelay: 32}); err != nil || res.Polarity != -1 || math.Abs(res.Delay) > 0.01 {
		t.Errorf("zero-delay inverted pair: %+v %v", res, err)
	}
}

func TestAlignEdge_InvalidRates(t *testing.T) {
	const n = 4096
	x := syntheticMusic(n, 48000, 0)
	cfg := AlignConfig{MaxDelay: 16}
	for _, c := range []struct{ in, out float64 }{
		{0, 48000}, {48000, 0}, {-1, 48000}, {48000, -1},
		{math.NaN(), 48000}, {48000, math.NaN()},
		{math.Inf(1), 48000}, {48000, math.Inf(-1)},
		{1, 1e9}, // ratio far outside the supported range
		{1e9, 1},
	} {
		if _, err := NewAlignedStereo(c.in, c.out, QualityHigh, cfg); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("NewAlignedStereo(%v, %v): %v, want ErrInvalidConfig", c.in, c.out, err)
		}
	}
	_ = x
}

func TestAlignEdge_NonFiniteSamples(t *testing.T) {
	const n = 4096
	left := syntheticMusic(n, 48000, 0)
	right := syntheticMusic(n, 48000, 3)
	cfg := AlignConfig{MaxDelay: 16, Window: 4096}

	// Estimates over non-finite input are refused rather than producing a
	// NaN delay: the caller sees ErrInvalidConfig and can sanitise the input.
	for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		r := append([]float64(nil), right...)
		r[100] = bad
		if _, err := EstimateChannelDelay(left, r, cfg); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("sample %v: EstimateChannelDelay err %v, want ErrInvalidConfig", bad, err)
		}
		if _, _, _, err := AlignStereo(left, r, cfg); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("sample %v: AlignStereo err %v, want ErrInvalidConfig", bad, err)
		}
		s, err := NewAlignedStereo(48000, 48000, QualityHigh, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, _, err := s.Process(left, r); err == nil {
			t.Errorf("sample %v: streaming Process accepted non-finite input", bad)
		}
	}
}

func TestAlignEdge_DelayOutOfRange(t *testing.T) {
	const n = 16384
	left := syntheticMusic(n, 48000, 0)
	// A delay at and just past the bound.
	for _, d := range []float64{16, 16.4, 24} {
		right := syntheticMusic(n, 48000, d)
		res, err := EstimateChannelDelay(left, right, AlignConfig{MaxDelay: 16})
		if !errors.Is(err, ErrAlignOutOfRange) {
			t.Errorf("delay %v, MaxDelay 16: %v, want ErrAlignOutOfRange", d, err)
		}
		if !res.OutOfRange || res.Valid() {
			t.Errorf("delay %v: result %+v not flagged out of range", d, res)
		}
		if _, _, _, err := AlignStereo(left, right, AlignConfig{MaxDelay: 16}); !errors.Is(err, ErrAlignOutOfRange) {
			t.Errorf("delay %v: AlignStereo err %v, want ErrAlignOutOfRange", d, err)
		}
		// The streaming stage reports it and keeps its previous (zero)
		// alignment instead of applying a truncated shift.
		s, err := NewAlignedStereo(48000, 48000, QualityHigh, AlignConfig{MaxDelay: 16, Window: 16000})
		if err != nil {
			t.Fatal(err)
		}
		_, _, sres, err := s.Process(left, right)
		if err != nil {
			t.Fatalf("delay %v: streaming returned error %v", d, err)
		}
		if !sres.OutOfRange {
			t.Errorf("delay %v: streaming result %+v not out of range", d, sres)
		}
		if _, ok := s.Alignment(); ok {
			t.Errorf("delay %v: streaming claimed a usable alignment", d)
		}
	}
}

func TestAlignEdge_ShortTails(t *testing.T) {
	// Input shorter than the analysis window, input shorter than the lag
	// margin, and an empty block must all be defined: no panic, an explicit
	// error from the estimator, and clean streaming output.
	left := syntheticMusic(20000, 48000, 0)
	right := syntheticMusic(20000, 48000, 3)

	// The one-shot helpers accept short input by using what is there, as long
	// as it can resolve a delay.
	short := 1000
	if _, err := EstimateChannelDelay(left[:short], right[:short], AlignConfig{MaxDelay: 8}); err != nil {
		t.Errorf("1000-sample estimate: %v", err)
	}
	// Very short input: explicit error, not a panic or a bogus delay.
	tiny := 100
	if _, err := EstimateChannelDelay(left[:tiny], right[:tiny], AlignConfig{MaxDelay: 8}); !errors.Is(err, ErrAlignShortWindow) {
		t.Errorf("100-sample estimate: %v, want ErrAlignShortWindow", err)
	}
	// Streaming: a block shorter than the window is buffered in the analysis
	// history and produces output of exactly the input length.
	s, err := NewAlignedStereo(48000, 48000, QualityHigh, AlignConfig{MaxDelay: 8, Window: 4096})
	if err != nil {
		t.Fatal(err)
	}
	for _, size := range []int{1, 2, 7, 100, 4095} {
		ol, oR, _, err := s.Process(left[:size], right[:size])
		if err != nil {
			t.Fatalf("block %d: %v", size, err)
		}
		if len(ol) != size || len(oR) != size {
			t.Fatalf("block %d: output lengths %d/%d", size, len(ol), len(oR))
		}
	}
	// An empty block is a no-op returning empty output.
	ol, oR, res, err := s.Process(nil, nil)
	if err != nil {
		t.Fatalf("empty block: %v", err)
	}
	if len(ol) != 0 || len(oR) != 0 {
		t.Errorf("empty block returned %d/%d samples", len(ol), len(oR))
	}
	_ = res
	// Flush is legal at any point and, after normal processing, gives a short
	// tail whose length is the resampler's usual one; a second Flush is empty.
	if _, _, _, err := s.Flush(); err != nil {
		t.Fatal(err)
	}
	if tl, tr, _, err := s.Flush(); err != nil || len(tl) != 0 || len(tr) != 0 {
		t.Errorf("second Flush: %d/%d samples, %v", len(tl), len(tr), err)
	}
}
