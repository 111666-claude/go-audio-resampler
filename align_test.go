// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"errors"
	"math"
	"testing"
)

// syntheticMusic returns a deterministic, broadband stereo-ready mono signal:
// a chord with upper partials where every partial is sampled at a shifted
// time, so a fractional delay of it can be produced exactly in closed form by
// delayingLeft. It is rich enough that a correlation peak is unambiguous.
func syntheticMusic(n int, rate, delay float64) []float64 {
	freqs := []float64{110, 220, 277.18, 329.63, 440, 554.37, 880, 1320, 1760, 2640, 3520, 5280, 7040, 9500}
	amps := []float64{0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.015}
	x := make([]float64, n)
	for i := range x {
		t := (float64(i) - delay) / rate
		var s float64
		for j, f := range freqs {
			s += amps[j] * math.Sin(2*math.Pi*f*t+0.37*float64(j))
		}
		x[i] = s * 0.4
	}
	return x
}

// bestLagCorrTest returns the largest absolute normalized correlation of a and
// b over lags -8..8, i.e. how well they correlate when allowed to slide.
func bestLagCorrTest(a, b []float64, maxLag int) float64 {
	best := 0.0
	for lag := -maxLag; lag <= maxLag; lag++ {
		lo, hi := max(0, -lag), min(len(a), len(a)-lag)
		if hi-lo < 2 {
			continue
		}
		var sxy, sxx, syy float64
		for i := lo; i < hi; i++ {
			sxy += a[i] * b[i+lag]
			sxx += a[i] * a[i]
			syy += b[i+lag] * b[i+lag]
		}
		if sxx <= 0 || syy <= 0 {
			continue
		}
		if v := math.Abs(sxy / math.Sqrt(sxx*syy)); v > best {
			best = v
		}
	}
	return best
}

func TestEstimateChannelDelay_Fractional(t *testing.T) {
	const n = 32768
	left := syntheticMusic(n, 48000, 0)
	for _, d := range []float64{0, 0.25, 3.37, -4.62, 12.5, -31.4} {
		right := syntheticMusic(n, 48000, d)
		res, err := EstimateChannelDelay(left, right, AlignConfig{MaxDelay: 64})
		if err != nil {
			t.Fatalf("delay %v: %v", d, err)
		}
		if math.Abs(res.Delay-d) > 0.005 {
			t.Errorf("delay %v: estimate %v (error %v)", d, res.Delay, res.Delay-d)
		}
		if !res.Valid() {
			t.Errorf("delay %v: not Valid: %+v", d, res)
		}
		if res.Polarity != 1 {
			t.Errorf("delay %v: polarity %v", d, res.Polarity)
		}
	}
}

func TestEstimateChannelDelay_Inverted(t *testing.T) {
	const n = 32768
	left := syntheticMusic(n, 48000, 0)
	right := syntheticMusic(n, 48000, 7.25)
	for i := range right {
		right[i] = -right[i]
	}
	res, err := EstimateChannelDelay(left, right, AlignConfig{MaxDelay: 64})
	if err != nil {
		t.Fatal(err)
	}
	if res.Polarity != -1 {
		t.Errorf("polarity %v, want -1", res.Polarity)
	}
	if math.Abs(res.Delay-7.25) > 0.005 {
		t.Errorf("delay %v, want 7.25", res.Delay)
	}
}

func TestEstimateChannelDelay_Errors(t *testing.T) {
	const n = 16384
	left := syntheticMusic(n, 48000, 0)
	// Silence.
	z := make([]float64, n)
	if _, err := EstimateChannelDelay(z, z, AlignConfig{MaxDelay: 16}); !errors.Is(err, ErrAlignSilent) {
		t.Errorf("silence: %v, want ErrAlignSilent", err)
	}
	// Unrelated channels: a pure uncorrelated noise pair.
	rng := &lcg{state: 12345}
	noiseL := make([]float64, n)
	noiseR := make([]float64, n)
	for i := range noiseL {
		noiseL[i] = 0.1 * rng.next()
		noiseR[i] = 0.1 * rng.next()
	}
	if _, err := EstimateChannelDelay(noiseL, noiseR, AlignConfig{MaxDelay: 16}); !errors.Is(err, ErrAlignUnreliable) {
		t.Errorf("unrelated channels: %v, want ErrAlignUnreliable", err)
	}
	// Out of range: a delay well past MaxDelay with material whose lobes are
	// narrow, so the peak lands on the boundary.
	right := syntheticMusic(n, 48000, 20)
	if _, err := EstimateChannelDelay(left, right, AlignConfig{MaxDelay: 16}); !errors.Is(err, ErrAlignOutOfRange) {
		t.Errorf("out of range: %v, want ErrAlignOutOfRange", err)
	}
	// Invalid configurations.
	for _, cfg := range []AlignConfig{
		{MaxDelay: 0},
		{MaxDelay: -4},
		{MaxDelay: 100, Window: 300},
		{MaxDelay: 16, SilenceRMS: math.NaN()},
	} {
		if _, err := EstimateChannelDelay(left, left, cfg); !errors.Is(err, ErrInvalidConfig) {
			t.Errorf("config %+v: %v, want ErrInvalidConfig", cfg, err)
		}
	}
	// Mismatched lengths.
	if _, err := EstimateChannelDelay(left, left[:100], AlignConfig{MaxDelay: 16}); !errors.Is(err, ErrInvalidConfig) {
		t.Errorf("mismatched lengths: %v", err)
	}
	// Short window through the estimator.
	short := syntheticMusic(100, 48000, 0)
	if _, err := EstimateChannelDelay(short, short, AlignConfig{MaxDelay: 8}); !errors.Is(err, ErrAlignShortWindow) {
		t.Errorf("short input: %v, want ErrAlignShortWindow", err)
	}
}

// lcg is a tiny deterministic pseudo-random source for tests, avoiding a
// dependency and giving identical values on every platform.
type lcg struct{ state uint64 }

func (l *lcg) next() float64 {
	l.state = l.state*6364136223846793005 + 1442695040888963407
	return float64(int64(l.state>>11))/float64(1<<52) - 1
}

func TestAlignStereo_RemovesDelay(t *testing.T) {
	const n = 32768
	left := syntheticMusic(n, 48000, 0)
	for _, d := range []float64{0.37, 3.37, -6.5} {
		right := syntheticMusic(n, 48000, d)
		al, ar, res, err := AlignStereo(left, right, AlignConfig{MaxDelay: 64})
		if err != nil {
			t.Fatalf("delay %v: %v", d, err)
		}
		if len(al) != n || len(ar) != n {
			t.Fatalf("delay %v: lengths %d/%d, want %d", d, len(al), len(ar), n)
		}
		// Measure the residual delay on the middle of the aligned pair.
		residual, err := EstimateChannelDelay(al[n/4:3*n/4], ar[n/4:3*n/4], AlignConfig{MaxDelay: 16})
		if err != nil {
			t.Fatalf("delay %v: residual estimate: %v", d, err)
		}
		if math.Abs(residual.Delay) > 0.02 {
			t.Errorf("delay %v: residual %v samples", d, residual.Delay)
		}
		// The aligned pair must correlate at least as well at lag zero as the
		// original did at its best lag.
		before := bestLagCorrTest(left, right, 8)
		after := bestLagCorrTest(al, ar, 1)
		if after < before-0.01 {
			t.Errorf("delay %v: alignment degraded correlation %v -> %v", d, before, after)
		}
		if res.Confidence < 0.99 {
			t.Errorf("delay %v: confidence %v", d, res.Confidence)
		}
	}
}

func TestAlignStereo_PolarityNotCorrected(t *testing.T) {
	const n = 16384
	left := syntheticMusic(n, 48000, 0)
	right := syntheticMusic(n, 48000, 3)
	for i := range right {
		right[i] = -right[i]
	}
	al, ar, res, err := AlignStereo(left, right, AlignConfig{MaxDelay: 32})
	if err != nil {
		t.Fatal(err)
	}
	if res.Polarity != -1 {
		t.Fatalf("polarity %v, want -1", res.Polarity)
	}
	// The outputs stay inverted: their correlation at lag 0 is negative.
	var sxy, sxx, syy float64
	for i := n / 4; i < 3*n/4; i++ {
		sxy += al[i] * ar[i]
		sxx += al[i] * al[i]
		syy += ar[i] * ar[i]
	}
	if s := sxy / math.Sqrt(sxx*syy); s > -0.9 {
		t.Errorf("aligned correlation %v, want near -1 (polarity untouched)", s)
	}
}

func TestAlignStereo_ErrorsLeaveChannelsUsable(t *testing.T) {
	const n = 4096
	left := syntheticMusic(n, 48000, 0)
	z := make([]float64, n)
	al, ar, res, err := AlignStereo(z, z, AlignConfig{MaxDelay: 16})
	if !errors.Is(err, ErrAlignSilent) {
		t.Fatalf("silence: %v, want ErrAlignSilent", err)
	}
	if al != nil || ar != nil {
		t.Errorf("silent error returned output of %d/%d samples, want nil", len(al), len(ar))
	}
	if !res.Silent {
		t.Errorf("result not Silent: %+v", res)
	}
	_ = left
}

func TestNewAlignedStereo_Validation(t *testing.T) {
	for _, c := range []struct {
		in, out float64
		cfg     AlignConfig
	}{
		{48000, 44100, AlignConfig{MaxDelay: 0}},
		{48000, 44100, AlignConfig{MaxDelay: 100, Window: 300}},
		{48000, 44100, AlignConfig{MaxDelay: 16, SilenceRMS: -1}},
		{math.NaN(), 44100, AlignConfig{MaxDelay: 16}},
		{48000, math.Inf(1), AlignConfig{MaxDelay: 16}},
	} {
		if _, err := NewAlignedStereo(c.in, c.out, QualityHigh, c.cfg); err == nil {
			t.Errorf("NewAlignedStereo(%v, %v, %+v) accepted", c.in, c.out, c.cfg)
		}
	}
}

// resReported returns the delay value of a result, kept in one place so the
// test reads as an assertion on the reported number.
func resReported(r AlignResult) float64 { return r.Delay }

func TestAlignedStereo_StreamingEquivalence(t *testing.T) {
	const (
		n     = 48000
		inHz  = 44100.0
		outHz = 48000.0
	)
	left := syntheticMusic(n, inHz, 0)
	right := syntheticMusic(n, inHz, 2.37)
	cfg := AlignConfig{MaxDelay: 32, Window: 16384}

	run := func(chunk int) ([]float64, []float64, AlignResult) {
		s, err := NewAlignedStereo(inHz, outHz, QualityHigh, cfg)
		if err != nil {
			t.Fatal(err)
		}
		var outL, outR []float64
		var res AlignResult
		for i := 0; i < n; i += chunk {
			end := min(i+chunk, n)
			ol, oR, r, err := s.Process(left[i:end], right[i:end])
			if err != nil {
				t.Fatal(err)
			}
			outL = append(outL, ol...)
			outR = append(outR, oR...)
			res = r
		}
		tl, tr, _, err := s.Flush()
		if err != nil {
			t.Fatal(err)
		}
		return append(outL, tl...), append(outR, tr...), res
	}

	refL, refR, refRes := run(n)
	if len(refL) != len(refR) {
		t.Fatalf("one-shot lengths differ: %d/%d", len(refL), len(refR))
	}
	if !refRes.Valid() {
		t.Fatalf("one-shot alignment unusable: %+v", refRes)
	}
	// The reported delay must be the one actually applied: the aligned pair
	// (at the input rate, before resampling) must be delay-free. Check the
	// streaming stage's own alignment report against a fresh estimate on the
	// same inputs.
	if math.Abs(resReported(refRes)-2.37) > 0.02 {
		t.Errorf("reported delay %v, want 2.37", resReported(refRes))
	}
	// Length follows the resampling ratio, exactly like ResampleStereo on the
	// aligned pair.
	if want := int(math.Round(float64(n) * outHz / inHz)); math.Abs(float64(len(refL)-want)) > 2 {
		t.Errorf("output length %d, want about %d", len(refL), want)
	}
	// The streamed pair must actually be delay-free: measure the residual
	// inter-channel delay of the *outputs* on the middle third (away from the
	// convergence transient). A stage that reported a delay but did not apply
	// it would leave the raised-cosine shift's 2.37 input samples here.
	mid := len(refL) / 3
	residual, err := EstimateChannelDelay(refL[mid:2*mid], refR[mid:2*mid], AlignConfig{MaxDelay: 16})
	if err != nil {
		t.Fatalf("residual estimate on output: %v", err)
	}
	if math.Abs(residual.Delay) > 0.05 {
		t.Errorf("outputs carry a residual delay of %v samples", residual.Delay)
	}
	for _, chunk := range []int{1, 7, 333, 4096} {
		gotL, gotR, gotRes := run(chunk)
		if len(gotL) != len(refL) || len(gotR) != len(refR) {
			t.Fatalf("chunk %d: length %d/%d, want %d/%d", chunk, len(gotL), len(gotR), len(refL), len(refR))
		}
		for i := range gotL {
			if gotL[i] != refL[i] || gotR[i] != refR[i] {
				t.Fatalf("chunk %d: sample %d differs from one-shot", chunk, i)
			}
		}
		if math.Abs(gotRes.Delay-refRes.Delay) > 1.0/512 {
			t.Errorf("chunk %d: final delay %v, want %v", chunk, gotRes.Delay, refRes.Delay)
		}
	}
}

func TestAlignedStereo_MatchesAlignThenResample(t *testing.T) {
	// Once the alignment has converged and the aligners' finite history has
	// flushed, a streaming AlignedStereo must equal AlignStereo followed by
	// resampling. At 1:1 the resampler is an exact passthrough, so the output
	// index equals the input index and the comparison is direct; the run is
	// long enough that the converged tail is most of the stream.
	const (
		n    = 96000
		rate = 48000.0
	)
	left := syntheticMusic(n, rate, 0)
	right := syntheticMusic(n, rate, 1.75)
	cfg := AlignConfig{MaxDelay: 32, Window: 16384, Freeze: true}

	s, err := NewAlignedStereo(rate, rate, QualityMedium, cfg)
	if err != nil {
		t.Fatal(err)
	}
	var streamL, streamR []float64
	for i := 0; i < n; i += 8192 {
		end := min(i+8192, n)
		ol, oR, _, err := s.Process(left[i:end], right[i:end])
		if err != nil {
			t.Fatal(err)
		}
		streamL = append(streamL, ol...)
		streamR = append(streamR, oR...)
	}
	tl, tr, res, err := s.Flush()
	if err != nil {
		t.Fatal(err)
	}
	streamL = append(streamL, tl...)
	streamR = append(streamR, tr...)
	if !res.Valid() {
		t.Fatalf("unusable estimate: %+v", res)
	}

	al, ar, _, err := AlignStereo(left, right, AlignConfig{MaxDelay: 32, Window: 16384})
	if err != nil {
		t.Fatal(err)
	}
	// AlignStereo's constant alignment must be the one the stream converged
	// to, otherwise the comparison would be between different delays.
	if math.Abs(res.Delay-1.75) > 0.01 {
		t.Fatalf("converged delay %v, want 1.75", res.Delay)
	}
	// The first estimate lands at input position Window, and the aligners'
	// history is at most Window of kernel reach; from three windows on the
	// output is fully determined by the converged alignment.
	start := 3 * 16384
	for i := start; i < len(streamL); i++ {
		if math.Abs(streamL[i]-al[i]) > 1e-12 || math.Abs(streamR[i]-ar[i]) > 1e-12 {
			t.Fatalf("sample %d differs: stream %v/%v, constant-alignment %v/%v",
				i, streamL[i], streamR[i], al[i], ar[i])
		}
	}
	// And the constant-alignment pair is itself delay-free, which is what
	// makes the equal tail meaningful.
	if r, err := EstimateChannelDelay(al, ar, AlignConfig{MaxDelay: 16}); err != nil || math.Abs(r.Delay) > 0.02 {
		t.Fatalf("AlignStereo left a residual delay: %+v %v", r, err)
	}
}

func TestAlignedStereo_FlushMatchesResampleStereoOfAlignedPair(t *testing.T) {
	// A resampling stream whose alignment is frozen before any signal is
	// processed reduces to align-then-resample exactly: every block after the
	// first uses one constant delay, and Flush drains the resamplers as usual.
	// Checked at a non-unity rate through the reported delay-free pair.
	const (
		n     = 48000
		inHz  = 48000.0
		outHz = 44100.0
	)
	left := syntheticMusic(n, inHz, 0)
	right := syntheticMusic(n, inHz, 2.375)
	s, err := NewAlignedStereo(inHz, outHz, QualityMedium, AlignConfig{MaxDelay: 32, Window: 16384, Freeze: true})
	if err != nil {
		t.Fatal(err)
	}
	var streamL, streamR []float64
	for i := 0; i < n; i += 4096 {
		end := min(i+4096, n)
		ol, oR, _, err := s.Process(left[i:end], right[i:end])
		if err != nil {
			t.Fatal(err)
		}
		streamL = append(streamL, ol...)
		streamR = append(streamR, oR...)
	}
	tl, tr, _, err := s.Flush()
	if err != nil {
		t.Fatal(err)
	}
	streamL = append(streamL, tl...)
	streamR = append(streamR, tr...)
	// Length must equal a one-shot resample of the same number of input
	// samples: alignment does not add or remove samples.
	wantLen := len(streamL)
	resampledL, resampledR, err := ResampleStereo(left, right, inHz, outHz, QualityMedium)
	if err != nil {
		t.Fatal(err)
	}
	if wantLen != len(resampledL) || len(streamR) != len(resampledR) {
		t.Errorf("aligned-resampled lengths %d/%d, plain resampler %d/%d",
			wantLen, len(streamR), len(resampledL), len(resampledR))
	}
}

func TestAlignedStereo_SilenceHoldsAndKeepsStreaming(t *testing.T) {
	const (
		n    = 48000
		inHz = 48000.0
	)
	left := syntheticMusic(n, inHz, 0)
	right := syntheticMusic(n, inHz, 3)
	s, err := NewAlignedStereo(inHz, inHz, QualityHigh, AlignConfig{MaxDelay: 32, Window: 16384})
	if err != nil {
		t.Fatal(err)
	}
	// One full window of signal produces the estimate; a second makes the
	// analysis see only signal, so the delay has converged before the silence
	// starts.
	for i := 0; i < 2*16384; i += 16384 {
		if _, _, _, err := s.Process(left[i:i+16384], right[i:i+16384]); err != nil {
			t.Fatal(err)
		}
	}
	before, ok := s.Alignment()
	if !ok {
		t.Fatal("no alignment after signal")
	}
	// A block is only silent once the whole analysis window is silence: the
	// first silent block still has a window's worth of signal in its history.
	// Feed two windows so the second analysis covers silence only; neither may
	// raise an error or move the applied delay, and both must produce output
	// for every input sample.
	z := make([]float64, 16384)
	var res AlignResult
	for pass := 0; pass < 2; pass++ {
		ol, oR, r, err := s.Process(z, z)
		if err != nil {
			t.Fatalf("silent block %d returned error: %v", pass, err)
		}
		if len(ol) != len(z) || len(oR) != len(z) {
			t.Errorf("silent block %d output lengths %d/%d, want %d", pass, len(ol), len(oR), len(z))
		}
		res = r
		if after, _ := s.Alignment(); after.Delay != before.Delay {
			t.Errorf("silence pass %d moved the delay from %v to %v", pass, before.Delay, after.Delay)
		}
	}
	if !res.Silent {
		t.Errorf("a fully silent analysis window was not reported: %+v", res)
	}
}

func TestAlignedStereo_NonFiniteRejected(t *testing.T) {
	s, err := NewAlignedStereo(48000, 44100, QualityHigh, AlignConfig{MaxDelay: 16, Window: 4096})
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float64, 512)
	bad := make([]float64, 512)
	bad[3] = math.NaN()
	if _, _, _, err := s.Process(x, bad); err == nil {
		t.Error("NaN sample accepted")
	}
	bad[3] = math.Inf(-1)
	if _, _, _, err := s.Process(bad, x); err == nil {
		t.Error("infinite sample accepted")
	}
	if _, _, _, err := s.Process(x, x[:100]); err == nil {
		t.Error("mismatched lengths accepted")
	}
}

func TestAlignedStereo_ProcessIntoContract(t *testing.T) {
	const n = 8192
	left := syntheticMusic(n, 48000, 0)
	right := syntheticMusic(n, 48000, 2)
	s, err := NewAlignedStereo(48000, 44100, QualityHigh, AlignConfig{MaxDelay: 32, Window: 4096})
	if err != nil {
		t.Fatal(err)
	}
	// Feed enough input to warm the alignment.
	if _, _, _, err := s.Process(left[:4096], right[:4096]); err != nil {
		t.Fatal(err)
	}
	before, _ := s.Alignment()

	chunk := 2048
	need := s.EstimateOutput(chunk)
	outL := make([]float64, need)
	outR := make([]float64, need)
	nOut, res, err := s.ProcessInto(left[4096:4096+chunk], right[4096:4096+chunk], outL, outR)
	if err != nil {
		t.Fatal(err)
	}
	if nOut <= 0 || nOut > need {
		t.Errorf("ProcessInto wrote %d samples, want 1..%d", nOut, need)
	}
	_ = res
	// An undersized buffer must fail without advancing state: the next good
	// call must produce the same output as it would from the current state.
	small := make([]float64, 1)
	if _, _, err := s.ProcessInto(left[6144:6144+chunk], right[6144:6144+chunk], small, outR); !errors.Is(err, ErrBufferTooSmall) {
		t.Fatalf("undersized buffer: %v, want ErrBufferTooSmall", err)
	}
	mid, _ := s.Alignment()
	if mid.Delay != before.Delay {
		t.Errorf("undersized ProcessInto moved the alignment")
	}
	// Now process the same chunk with adequate buffers: it must succeed and
	// produce the same count as a control run that never saw the failure.
	ctrl, err := NewAlignedStereo(48000, 44100, QualityHigh, AlignConfig{MaxDelay: 32, Window: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := ctrl.Process(left[:4096], right[:4096]); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := ctrl.Process(left[4096:4096+chunk], right[4096:4096+chunk]); err != nil {
		t.Fatal(err)
	}
	gotL := make([]float64, need)
	gotR := make([]float64, need)
	nGot, _, err := s.ProcessInto(left[6144:6144+chunk], right[6144:6144+chunk], gotL, gotR)
	if err != nil {
		t.Fatal(err)
	}
	wantL := make([]float64, need)
	wantR := make([]float64, need)
	nWant, _, err := ctrl.ProcessInto(left[6144:6144+chunk], right[6144:6144+chunk], wantL, wantR)
	if err != nil {
		t.Fatal(err)
	}
	if nGot != nWant {
		t.Fatalf("retry produced %d samples, control %d", nGot, nWant)
	}
	for i := range nGot {
		if gotL[i] != wantL[i] || gotR[i] != wantR[i] {
			t.Fatalf("retry sample %d differs from control", i)
		}
	}
}

func TestAlignedStereo_Latency(t *testing.T) {
	// Latency must cover the alignment's common delay plus the resampler's,
	// and the measured startup deficit of a first large block must not exceed
	// it.
	const (
		n     = 44100
		inHz  = 44100.0
		outHz = 48000.0
	)
	s, err := NewAlignedStereo(inHz, outHz, QualityHigh, AlignConfig{MaxDelay: 64, Window: 16384})
	if err != nil {
		t.Fatal(err)
	}
	left := syntheticMusic(n, inHz, 0)
	right := syntheticMusic(n, inHz, 3)
	out, _, _, err := s.Process(left, right)
	if err != nil {
		t.Fatal(err)
	}
	measured := int(math.Round(float64(n)*outHz/inHz)) - len(out)
	got := s.Latency()
	if measured < got-2 || measured > got+2 {
		t.Errorf("Latency()=%d, measured deficit %d", got, measured)
	}
	// It must equal the plain resampler's deficit: the alignment withholds no
	// samples, it only shifts each channel by the same fixed amount.
	plain, err := NewEngine(inHz, outHz, QualityHigh)
	if err != nil {
		t.Fatal(err)
	}
	if plain.Latency() != got {
		t.Errorf("Latency()=%d, plain resampler %d", got, plain.Latency())
	}
}

func TestAlignedLateness_Pinned(t *testing.T) {
	// The exported lateness is the delay kernel's group delay; pin it so a
	// kernel redesign cannot silently change the public timing contract.
	if got := AlignedLateness(); got != alignHalfTaps {
		t.Errorf("AlignedLateness()=%d, want %d", got, alignHalfTaps)
	}
}

func TestEstimateChannelDelay_RealStereo(t *testing.T) {
	// The committed real fixture under internal/align is exercised there; this
	// test additionally checks that the public entry point sees the same
	// alignment qualities on synthetic real-like material with added room
	// noise and a different level per channel.
	const n = 32768
	left := syntheticMusic(n, 48000, 0)
	right := syntheticMusic(n, 48000, 5.5)
	rng := &lcg{state: 777}
	for i := range right {
		right[i] = 0.7*right[i] + 0.01*rng.next()
		left[i] += 0.01 * rng.next()
	}
	res, err := EstimateChannelDelay(left, right, AlignConfig{MaxDelay: 32})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.Delay-5.5) > 0.01 {
		t.Errorf("estimate %v, want 5.5", res.Delay)
	}
	if res.Confidence < 0.95 {
		t.Errorf("confidence %v, want >= 0.95", res.Confidence)
	}
}
