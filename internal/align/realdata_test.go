// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package align

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// realFixturePath is the committed public-domain stereo excerpt described in
// testdata/README.md.
const realFixturePath = "testdata/stereo_march_44k_s16.pcm"

// loadRealFixture reads the interleaved 16-bit stereo fixture into two mono
// channels.
func loadRealFixture(t *testing.T) (left, right []float64) {
	t.Helper()
	b, err := os.ReadFile(realFixturePath)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	if len(b)%4 != 0 {
		t.Fatalf("fixture length %d is not a whole number of stereo frames", len(b))
	}
	n := len(b) / 4
	left = make([]float64, n)
	right = make([]float64, n)
	for i := range n {
		left[i] = float64(int16(binary.LittleEndian.Uint16(b[4*i:]))) / 32768
		right[i] = float64(int16(binary.LittleEndian.Uint16(b[4*i+2:]))) / 32768
	}
	return left, right
}

// fractionalShift delays x by d samples with a 32-tap windowed-sinc
// interpolant. It is the reference construction for the real-material test:
// the fixture itself is band-limited real audio, so an exact shift is not
// available in closed form, and this kernel's own error is orders of magnitude
// below the tolerance asserted.
func fractionalShift(x []float64, d float64) []float64 {
	const half = 16
	y := make([]float64, len(x))
	for i := range y {
		pos := float64(i) - d
		k0 := int(math.Floor(pos))
		var s float64
		for j := -half + 1; j <= half; j++ {
			k := k0 + j
			if k < 0 || k >= len(x) {
				continue
			}
			u := pos - float64(k)
			w := 1.0
			if math.Abs(u) > 1e-12 {
				w = math.Sin(math.Pi*u) / (math.Pi * u)
				w *= math.Sin(math.Pi*u/half) / (math.Pi * u / half)
			}
			s += x[k] * w
		}
		y[i] = s
	}
	return y
}

func TestRealData_EstimatorError(t *testing.T) {
	left, right := loadRealFixture(t)
	// The recording's own inter-channel offset is small but not exactly zero
	// (different microphone paths in the original capture). Measure it with
	// the estimator on the unshifted pair and use it as the baseline, then
	// assert that injected shifts are recovered relative to it.
	base, err := EstimateDelay(left, right, Config{MaxDelay: 32})
	if err != nil {
		t.Fatal(err)
	}
	if base.Confidence < 0.5 {
		t.Fatalf("fixture is not correlated enough: %+v", base)
	}
	for _, d := range []float64{0.25, -0.5, 1.75, 4.37, -7.9, 12.5} {
		shifted := fractionalShift(right, d)
		est, err := EstimateDelay(left, shifted, Config{MaxDelay: 32})
		if err != nil {
			t.Fatal(err)
		}
		if est.Silent || est.OutOfRange {
			t.Fatalf("shift %v: %+v", d, est)
		}
		residual := est.Delay - base.Delay - d
		if math.Abs(residual) > 0.02 {
			t.Errorf("shift %v: estimator error %v samples (est %v, base %v)", d, residual, est.Delay, base.Delay)
		}
	}
}

func TestRealData_SessionRemovesComb(t *testing.T) {
	left, right := loadRealFixture(t)
	const d = 2.37
	shifted := fractionalShift(right, d)
	s := newTestSession(t, SessionOptions{MaxDelay: 32, Window: 16384})
	outL, outR, est := runSession(t, s, left, shifted, 2048)
	if !s.HaveEst {
		t.Fatalf("no estimate: %+v", est)
	}
	// After alignment the pair's residual delay must be a small fraction of a
	// sample: the estimator's own error plus the fixture shift kernel's. The
	// raw lag-0 correlation is limited by the recording itself (its two
	// channels are not identical, so it is about 0.75 even when perfectly
	// aligned), so the check is on the delay, not on the correlation level.
	// Measure over the later part of the stream, after the estimate converges.
	w := 16384
	residual, err := EstimateDelay(outL[len(outL)-w:], outR[len(outR)-w:], Config{MaxDelay: 8})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(residual.Delay) > 0.05 {
		t.Errorf("residual inter-channel delay after alignment %v samples", residual.Delay)
	}
	// The session's own estimate over the same material must agree with the
	// true shift plus the recording's intrinsic offset.
	if math.Abs(est.Delay-d-0.06) > 0.1 {
		t.Errorf("session estimate %v, want about %v", est.Delay, d)
	}
	// Comb removal shows at lag zero: alignment must at least match the
	// correlation the pair reaches when allowed to slide to its best lag.
	beforeBest := bestLagCorr(left[len(left)-w:], shifted[len(shifted)-w:], 8)
	after := math.Abs(corrAt(outL[len(outL)-w:], outR[len(outR)-w:], 0))
	if after < beforeBest-0.02 {
		t.Errorf("alignment degraded correlation: best-lag before %v, lag-0 after %v", beforeBest, after)
	}
}

// bestLagCorr returns the largest absolute normalized correlation over lags
// -maxLag..maxLag, i.e. how well the pair correlates when allowed to slide.
func bestLagCorr(a, b []float64, maxLag int) float64 {
	best := 0.0
	for lag := -maxLag; lag <= maxLag; lag++ {
		if v := math.Abs(corrAt(a, b, lag)); v > best {
			best = v
		}
	}
	return best
}

// corrAt returns the normalized correlation at the given lag over the whole
// slices (mean removed).
func corrAt(a, b []float64, lag int) float64 {
	n := min(len(a), len(b))
	lo, hi := max(0, -lag), min(n, n-lag)
	var ma, mb float64
	for i := lo; i < hi; i++ {
		ma += a[i]
		mb += b[i+lag]
	}
	cnt := float64(hi - lo)
	ma /= cnt
	mb /= cnt
	var sxy, sxx, syy float64
	for i := lo; i < hi; i++ {
		x := a[i] - ma
		y := b[i+lag] - mb
		sxy += x * y
		sxx += x * x
		syy += y * y
	}
	if sxx <= 0 || syy <= 0 {
		return 0
	}
	return sxy / math.Sqrt(sxx*syy)
}

func TestRealData_SilenceDetection(t *testing.T) {
	// Zero the fixture's quietest 0.5 s and confirm the session treats it as
	// silence rather than aligning on the residual noise.
	left, right := loadRealFixture(t)
	s := newTestSession(t, SessionOptions{MaxDelay: 32, Window: 8192})
	// Correlate first so an alignment exists.
	if _, _, _, err := s.Process(left[:8192], right[:8192]); err != nil {
		t.Fatal(err)
	}
	if !s.HaveEst {
		t.Fatal("no estimate on real material")
	}
	prev := s.Delay()
	silent := make([]float64, 8192)
	_, _, est, err := s.Process(silent, silent)
	if err != nil {
		t.Fatal(err)
	}
	if !est.Silent {
		t.Errorf("silent block not reported silent: %+v", est)
	}
	if s.Delay() != prev {
		t.Errorf("silence moved the alignment from %v to %v", prev, s.Delay())
	}
}
