// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package align

import (
	"math"

	"github.com/tphakala/go-audio-resampler/internal/mathutil"
)

// Aligner applies a fixed fractional sample delay to a streamed channel.
//
// The delay is realised with a bank of Kaiser-windowed sinc kernels, one
// kernel per sub-sample position. Each kernel has 2*HalfTaps+1 taps and its
// own DC-gain normalisation, so no position introduces a level step: the bank
// is flat to better than 0.001 dB and its phase delay matches the requested
// delay to a fraction of a sample across the passband (aligner_test.go
// measures both).
//
// The Aligner is causal: it accepts delays of at least HalfTaps samples, so
// every output sample depends only on input already seen. A stream of N input
// samples yields exactly N output samples with no added latency beyond the
// requested delay itself, and the result does not depend on how the input was
// chunked. The caller that needs a smaller delay adds a fixed whole-sample
// base delay; the stereo Session does exactly that.
type Aligner struct {
	// maxDelay bounds every delay SetDelay may install, and therefore sizes
	// the retained history.
	maxDelay int

	// whole and fracIndex encode the applied delay: delay = whole +
	// fracIndex/subPhases, with 0 <= fracIndex < subPhases.
	whole     int
	fracIndex int

	bank [][]float64

	// history retains the raw input tail the next chunk needs to evaluate its
	// first outputs without a seam. It is independent of the delay (it is
	// sized for maxDelay), so SetDelay never invalidates it.
	history []float64
	// scratch is the combined history+input+pad buffer for one ProcessInto
	// call; the warm path allocates nothing.
	scratch []float64
}

// Delay kernel design constants.
const (
	// subPhases is the number of fractional positions resolved by the delay
	// bank, 256. The worst-case quantisation is half a sub-phase, 1/512
	// sample, two orders of magnitude below the pitch period of the highest
	// audible tone.
	subPhases = 256

	// HalfTaps is the half-length of each polyphase kernel, so each kernel has
	// 2*HalfTaps+1 = 129 taps, and it is also the smallest delay the causal
	// aligner can apply. It is exported because the root package reports it as
	// the alignment lateness; the kernel itself is internal.
	HalfTaps = 64

	// kernelBeta is the Kaiser window shape parameter for the delay kernels.
	// 9.0 is the value that, for a 129-tap fractionally shifted sinc, holds
	// the passband flat to better than 0.001 dB and keeps the transition
	// inside the top 7% of the band (aligner_test.go measures both).
	kernelBeta = 9.0
)

// NewAligner creates an aligner for a fixed delay of delay samples: output
// sample n is the input at n-delay. delay must be finite, at least HalfTaps
// (the kernel's own group delay; a smaller delay would need input not yet
// received) and at most maxDelay. Later SetDelay calls are held to the same
// bound, which is what lets the retained history survive a delay change.
//
// The delay is quantised to the subPhases bank, so the applied delay differs
// from the requested one by at most 1/(2*subPhases) samples.
func NewAligner(delay float64, maxDelay int) (*Aligner, error) {
	if maxDelay < HalfTaps {
		maxDelay = HalfTaps
	}
	a := &Aligner{
		maxDelay: maxDelay,
		bank:     designBank(),
	}
	if err := a.SetDelay(delay); err != nil {
		return nil, err
	}
	a.history = make([]float64, 0, a.historyKeep())
	return a, nil
}

// SetDelay changes the applied delay, keeping the retained history and
// therefore the stream's continuity. It is used by the Session when a new
// segment's estimate differs from the applied one; the delay must be within
// the [HalfTaps, maxDelay] bounds given at construction.
func (a *Aligner) SetDelay(delay float64) error {
	if math.IsNaN(delay) || math.IsInf(delay, 0) || delay < HalfTaps || delay > float64(a.maxDelay) {
		return errBadDelay
	}
	// Quantise the delay into a whole part and a non-negative phase index,
	// keeping Delay() consistent with the kernel actually selected.
	q := int64(math.Round(delay * subPhases))
	whole := int(q / subPhases)
	idx := int(q % subPhases)
	if idx < 0 {
		idx += subPhases
		whole--
	}
	a.whole = whole
	a.fracIndex = idx
	return nil
}

// Delay returns the exact delay the aligner applies, in samples.
func (a *Aligner) Delay() float64 {
	return float64(a.whole) + float64(a.fracIndex)/subPhases
}

// Reset discards the retained history, returning the aligner to its
// just-constructed state.
func (a *Aligner) Reset() {
	a.history = a.history[:0]
}

// historyKeep returns how many samples of the retained tail the next chunk
// needs: its first output's kernel window reaches HalfTaps+whole samples before
// the chunk start. Sizing for maxDelay rather than the current delay keeps the
// history valid across SetDelay.
func (a *Aligner) historyKeep() int {
	return a.maxDelay + HalfTaps
}

// Process applies the delay to one chunk and returns freshly owned output of
// the same length. It is the convenient form of ProcessInto.
func (a *Aligner) Process(input []float64) []float64 {
	out := make([]float64, len(input))
	_, _ = a.ProcessInto(input, out)
	return out
}

// ProcessInto applies the delay to input, writing len(input) samples into dst,
// and returns the count. It reuses grow-only internal buffers, so once warm it
// does not allocate. dst must be at least len(input); otherwise ProcessInto
// returns an error and leaves the aligner's state untouched.
func (a *Aligner) ProcessInto(input, dst []float64) (int, error) {
	if len(dst) < len(input) {
		return 0, errBufferTooSmall
	}
	// Output sample n is x[n-whole-frac], evaluated through the kernel: its
	// read window starts whole+half samples before n and ends half-whole after
	// it. leftPad and rightPad extend the scratch with the zeros a stream
	// begins and ends with, so the kernel is never truncated at a chunk edge
	// (which would make the output depend on the chunking).
	leftPad := maxInt(0, HalfTaps+a.whole)
	rightPad := maxInt(HalfTaps-a.whole+1, 0)
	histLen := len(a.history)
	total := leftPad + histLen + len(input) + rightPad
	if cap(a.scratch) < total {
		a.scratch = make([]float64, total)
	} else {
		a.scratch = a.scratch[:total]
		clear(a.scratch)
	}
	scratch := a.scratch
	copy(scratch[leftPad:], a.history)
	copy(scratch[leftPad+histLen:], input)
	// The left and right pads stay zero.

	h := a.bank[a.fracIndex]
	// h[j] = sinc(j-half-frac): y[m] = sum_j h[j] x[m+half-j] approximates
	// x[m-frac]. With m = n-whole, output sample n reads x[n-whole+half-j], so
	// the read index walks down as j walks up. base is that index for j = 0 in
	// scratch coordinates.
	base := leftPad + histLen + HalfTaps - a.whole
	for i := range input {
		p := base + i
		var sum float64
		for j, hv := range h {
			sum += hv * scratch[p-j]
		}
		dst[i] = sum
	}

	// Retain the tail for the next chunk: its first output reads back
	// historyKeep samples, so the last that many merged samples suffice.
	keep := a.historyKeep()
	mergedEnd := leftPad + histLen + len(input)
	start := maxInt(mergedEnd-keep, 0)
	a.history = append(a.history[:0], scratch[start:mergedEnd]...)
	return len(input), nil
}

// designBank builds the sub-phase kernel bank once for the process. The bank
// depends only on the kernel shape constants, not on the delay, so every
// aligner and every aligner rebuilt on an alignment update shares it.
func designBank() [][]float64 {
	bankOnce.Do(func() {
		bank = buildBank()
	})
	return bank
}

// buildBank constructs the Kaiser-windowed sinc bank. Each phase p delays by
// HalfTaps + p/subPhases samples in the causal indexing; the whole-sample
// HalfTaps is the kernel's group delay and is compensated by the base index
// in ProcessInto. Every kernel is normalised to unit DC gain, which is what
// keeps a constant input constant for every fractional position.
func buildBank() [][]float64 {
	i0Beta := mathutil.BesselI0(kernelBeta)
	length := 2*HalfTaps + 1
	bank := make([][]float64, subPhases)
	for p := range subPhases {
		frac := float64(p) / subPhases
		h := make([]float64, length)
		var sum float64
		for j := range h {
			u := float64(j) - HalfTaps - frac
			var sinc float64
			if math.Abs(u) < 1e-15 {
				sinc = 1
			} else {
				sinc = math.Sin(math.Pi*u) / (math.Pi * u)
			}
			// Window over the distance from the kernel centre, normalised to
			// the kernel's half-extent so both ends taper to the window zero.
			arg := kernelBeta * math.Sqrt(math.Max(0, 1-math.Pow(u/(HalfTaps+0.5), 2)))
			h[j] = sinc * mathutil.BesselI0(arg) / i0Beta
			sum += h[j]
		}
		if sum != 0 {
			for j := range h {
				h[j] /= sum
			}
		}
		bank[p] = h
	}
	return bank
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
