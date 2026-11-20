// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package align

import (
	"errors"
	"fmt"
	"math"
)

// SessionOptions configures a stereo alignment session.
type SessionOptions struct {
	// MaxDelay is the largest inter-channel delay the estimator searches for,
	// in input samples, in either direction. It must be positive and no more
	// than maxSessionDelay.
	MaxDelay int

	// Window is the analysis window, in input samples, used for each delay
	// estimate. A longer window resolves a fractional delay more accurately
	// and survives more noise; the estimator needs at least
	// 2*MaxDelay+minEstimateSamples samples. A value of 0 uses DefaultWindow.
	Window int

	// SilenceRMS is the RMS level below which a block is treated as silence
	// and the previous alignment is retained rather than re-estimated from
	// noise. A value of 0 uses DefaultSilenceRMS.
	SilenceRMS float64

	// Freeze stops re-estimation after the first usable estimate and keeps
	// that alignment for the whole stream. It is for material whose delay is
	// known constant, where a quiet or noisy passage could otherwise perturb
	// the alignment.
	Freeze bool
}

// DefaultWindow is the analysis window used when SessionOptions.Window is
// zero, in input samples: 16384 samples is 0.34 s at 48 kHz, long enough for
// the estimator to reach sub-millisample accuracy on music while short enough
// to fit a typical streaming block.
const DefaultWindow = 16384

// maxSessionDelay bounds SessionOptions.MaxDelay. The estimator's lag search
// is O(MaxDelay * window), so an unbounded value would let a caller request an
// arbitrarily expensive estimate; 16384 samples (0.34 s at 48 kHz) is far
// beyond any plausible microphone offset.
const maxSessionDelay = 16384

// alignmentStep is the smallest change in the estimated delay that triggers a
// delay update. 1/256 sample is the aligner's own quantisation, so a smaller
// change could not be applied anyway; the hysteresis avoids re-tuning on every
// estimator jitter of a thousandth of a sample.
const alignmentStep = 1.0 / subPhases

// Session carries a stereo alignment across streaming blocks: the delay
// estimate, both aligners and the analysis history are held between calls.
//
// Re-estimation is driven by absolute stream position, not by call
// boundaries: the estimate is refreshed exactly when the stream crosses a
// multiple of Window samples, using the Window samples that end there. A fixed
// input therefore yields the same aligned output for every chunking, because
// neither the estimate schedule nor the aligners' internal state depends on
// where the caller splits the input.
//
// A silent or unreliable block never moves a previously found alignment: the
// estimator reports Silent / OutOfRange / low Confidence, and the session
// keeps its current delay and continues aligning with it.
type Session struct {
	opts SessionOptions

	// base is the common delay, in samples, applied to both channels so the
	// inter-channel offset can be cancelled by retarding one of them. It is
	// HalfTaps, the smallest delay the causal kernels support, and never
	// changes.
	base float64

	alignL *Aligner
	alignR *Aligner

	// delay is the inter-channel delay currently applied: positive retards
	// the right channel. est is the last estimate reported to the caller.
	delay   float64
	est     Estimate
	HaveEst bool

	// pos is the number of input samples consumed so far; nextEst is the next
	// stream position at which to re-estimate.
	pos     int64
	nextEst int64

	// histL and histR hold the last Window input samples (before alignment)
	// ending at pos, so an estimate always sees the same material for a given
	// stream position regardless of chunking.
	histL []float64
	histR []float64

	frozen bool
}

// NewSession creates a stereo alignment session.
func NewSession(opts SessionOptions) (*Session, error) {
	if opts.MaxDelay <= 0 {
		return nil, errBadMaxDelay
	}
	if opts.MaxDelay > maxSessionDelay {
		return nil, fmt.Errorf("%w: MaxDelay %d exceeds the %d-sample limit", errBadMaxDelay, opts.MaxDelay, maxSessionDelay)
	}
	if opts.Window == 0 {
		opts.Window = DefaultWindow
	}
	if opts.Window < 2*opts.MaxDelay+minEstimateSamples {
		return nil, fmt.Errorf("%w: window %d is shorter than 2*MaxDelay+%d", ErrShortWindow, opts.Window, minEstimateSamples)
	}
	if opts.SilenceRMS < 0 || math.IsNaN(opts.SilenceRMS) {
		return nil, errBadSilence
	}
	// Both channels get the same base delay, and the one that is early gets
	// the extra offset, so each aligner's delay is base+[0,MaxDelay] and stays
	// within the kernels' causal range without a large common latency.
	base := float64(HalfTaps)
	s := &Session{
		opts:    opts,
		base:    base,
		nextEst: int64(opts.Window),
	}
	bound := int(base) + opts.MaxDelay + HalfTaps
	l, err := NewAligner(base, bound)
	if err != nil {
		return nil, err
	}
	r, err := NewAligner(base, bound)
	if err != nil {
		return nil, err
	}
	s.alignL, s.alignR = l, r
	s.histL = make([]float64, 0, opts.Window)
	s.histR = make([]float64, 0, opts.Window)
	return s, nil
}

// Reset discards the alignment, the aligners and the analysis history,
// returning the session to its just-constructed state.
func (s *Session) Reset() {
	s.delay = 0
	s.est = Estimate{}
	s.HaveEst = false
	s.frozen = false
	s.pos = 0
	s.nextEst = int64(s.opts.Window)
	s.histL = s.histL[:0]
	s.histR = s.histR[:0]
	s.alignL.Reset()
	s.alignR.Reset()
	_ = s.alignL.SetDelay(s.base)
	_ = s.alignR.SetDelay(s.base)
}

// Estimate returns the alignment currently applied, and whether a usable
// estimate has been found. Before the first usable estimate, Delay is 0 and
// the channels pass with only the common base delay. The returned Estimate's
// Delay is always the applied delay; Silent and OutOfRange describe the most
// recent analysis window.
func (s *Session) Estimate() (Estimate, bool) {
	return s.est, s.HaveEst
}

// Delay returns the inter-channel delay currently cancelled, in samples:
// positive means the right channel lagged the left and was advanced, negative
// means it led and was retarded. It is the estimate in force.
func (s *Session) Delay() float64 {
	return s.delay
}

// Latency returns the minimum delay, in samples, that alignment adds to both
// channels: HalfTaps, the causal kernels' group delay. The channel that is
// early is retarded by Delay() more than the other, so the aligned pair's
// absolute timeline lags the input by Latency()+max(Delay(), 0) for the late
// channel and Latency()+max(-Delay(), 0) for the early one; the difference
// between the two is exactly the offset that was cancelled.
func (s *Session) Latency() int {
	return int(s.base)
}

// Process aligns one block of both channels, returning aligned left and right
// output of len(left) samples each. The two chunks must be the same length:
// they are corresponding frames of one stream.
//
// The returned Estimate is the alignment state during this block. Process
// returns an error only for malformed input (length mismatch, NaN or infinite
// samples); silence, uncorrelated material and an out-of-range delay are
// reported through the Estimate and leave the previous alignment in place.
func (s *Session) Process(left, right []float64) (leftOut, rightOut []float64, est Estimate, err error) {
	if len(left) != len(right) {
		return nil, nil, Estimate{}, fmt.Errorf("%w: left has %d samples, right %d", errLengthMismatch, len(left), len(right))
	}
	if hasNonFinite(left) || hasNonFinite(right) {
		return nil, nil, Estimate{}, errNonFinite
	}
	if len(left) == 0 {
		return []float64{}, []float64{}, s.est, nil
	}
	leftOut = make([]float64, len(left))
	rightOut = make([]float64, len(right))
	if err := s.processInto(left, right, leftOut, rightOut); err != nil {
		return nil, nil, Estimate{}, err
	}
	return leftOut, rightOut, s.est, nil
}

// ProcessInto is Process with caller-owned output buffers. It writes
// len(left) samples into each buffer. Both buffers must be at least as long as
// their channel's input; otherwise it returns ErrBufferTooSmall before any
// state changes, so the caller can retry with larger buffers.
func (s *Session) ProcessInto(left, right, leftOut, rightOut []float64) (Estimate, error) {
	if len(left) != len(right) {
		return Estimate{}, fmt.Errorf("%w: left has %d samples, right %d", errLengthMismatch, len(left), len(right))
	}
	if len(leftOut) < len(left) || len(rightOut) < len(right) {
		return Estimate{}, errBufferTooSmall
	}
	if hasNonFinite(left) || hasNonFinite(right) {
		return Estimate{}, errNonFinite
	}
	if len(left) == 0 {
		return s.est, nil
	}
	if err := s.processInto(left, right, leftOut, rightOut); err != nil {
		return Estimate{}, err
	}
	return s.est, nil
}

// processInto aligns the block, splitting it at absolute estimate boundaries.
func (s *Session) processInto(left, right, leftOut, rightOut []float64) error {
	start := 0
	// appended tracks how much of this block is already in the analysis
	// history, so a block split at several boundaries appends each sample
	// exactly once.
	appended := 0
	for start < len(left) {
		toBoundary := s.nextEst - s.pos
		if toBoundary > int64(len(left)-start) {
			// No estimate boundary inside the rest of the block.
			if err := s.alignSegment(left, right, leftOut, rightOut, start, len(left)); err != nil {
				return err
			}
			s.appendHistory(left[appended:], right[appended:])
			s.pos += int64(len(left) - start)
			return nil
		}
		end := start + int(toBoundary)
		if err := s.alignSegment(left, right, leftOut, rightOut, start, end); err != nil {
			return err
		}
		s.appendHistory(left[appended:end], right[appended:end])
		appended = end
		s.pos = s.nextEst
		s.estimate()
		s.nextEst += int64(s.opts.Window)
		start = end
	}
	return nil
}

func (s *Session) alignSegment(left, right, leftOut, rightOut []float64, start, end int) error {
	if _, err := s.alignL.ProcessInto(left[start:end], leftOut[start:end]); err != nil {
		return err
	}
	if _, err := s.alignR.ProcessInto(right[start:end], rightOut[start:end]); err != nil {
		return err
	}
	return nil
}

// appendHistory appends the given slices to the analysis history, keeping the
// most recent Window samples of each channel.
func (s *Session) appendHistory(left, right []float64) {
	s.histL = appendBounded(s.histL, left, s.opts.Window)
	s.histR = appendBounded(s.histR, right, s.opts.Window)
}

// appendBounded appends src to dst and keeps the most recent keep samples.
func appendBounded(dst, src []float64, keep int) []float64 {
	if len(src) >= keep {
		dst = append(dst[:0], src[len(src)-keep:]...)
		return dst
	}
	combined := len(dst) + len(src)
	if combined > keep {
		drop := combined - keep
		copy(dst, dst[drop:])
		dst = dst[:len(dst)-drop]
	}
	return append(dst, src...)
}

// estimate refreshes the applied delay from the accumulated history.
//
// The Estimate reported to the caller always describes the alignment actually
// applied (Delay is the delay in force, Confidence and Polarity belong to the
// last usable observation), while Silent and OutOfRange describe the newest
// observation. That way a caller reading Delay at any point gets the value the
// output was aligned with, and a silent or out-of-range passage is still
// visible as a flag rather than as a spurious change of alignment.
func (s *Session) estimate() {
	if s.frozen {
		return
	}
	if len(s.histL) < s.opts.Window {
		return
	}
	e, err := EstimateDelay(s.histL, s.histR, Config{
		MaxDelay:   s.opts.MaxDelay,
		SilenceRMS: s.opts.SilenceRMS,
	})
	if err != nil {
		// The history is by construction the configured window, so this is
		// unreachable; retaining the previous alignment is the safe response.
		return
	}
	if !e.Valid() {
		// Silence, an unreliable estimate or a delay beyond the search range:
		// keep the alignment in place rather than moving on noise, but report
		// the observation's flags.
		s.est.Silent = e.Silent
		s.est.OutOfRange = e.OutOfRange
		return
	}
	s.est = Estimate{
		Delay:      e.Delay,
		Confidence: e.Confidence,
		Polarity:   e.Polarity,
	}
	s.HaveEst = true
	if math.Abs(e.Delay-s.delay) >= alignmentStep {
		// The early channel is retarded by the offset: a delayed right
		// channel (positive Delay) has the left retarded, and vice versa.
		// Both delays are base+[0,MaxDelay], inside the bound passed to
		// NewAligner, so SetDelay cannot fail for a Valid estimate; if it
		// somehow did, keep the previous alignment rather than reporting a
		// delay the filters are not applying.
		if err := s.alignL.SetDelay(s.base + math.Max(e.Delay, 0)); err != nil {
			return
		}
		if err := s.alignR.SetDelay(s.base + math.Max(-e.Delay, 0)); err != nil {
			return
		}
		s.delay = e.Delay
	}
	if s.opts.Freeze {
		s.frozen = true
	}
}

// hasNonFinite reports whether any sample is NaN or infinite.
func hasNonFinite(x []float64) bool {
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

// Errors reported by the session for malformed input.
var (
	errLengthMismatch = errors.New("align: left and right chunks differ in length")
	errNonFinite      = errors.New("align: input contains NaN or infinite samples")
)
