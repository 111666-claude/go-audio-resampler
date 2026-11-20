// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package resampler

import (
	"errors"
	"fmt"
	"math"

	"github.com/tphakala/go-audio-resampler/internal/align"
)

// Stereo channel alignment.
//
// Two microphones recording the same performance are rarely sample-aligned:
// capsule spacing, separate converters and different preamp latencies each
// shift one channel against the other by an integer or fractional number of
// samples. Summing or downmixing such channels comb-filters the result — a
// delay of d samples cancels the shared content where d*f is a half-integer
// cycle — and resampling the pair does not fix it, because the channels drift
// apart in time, not in rate. AlignStereo estimates that delay and removes it
// before resampling; AlignedStereo does the same as a streaming entry point
// that resamples the aligned pair.

// ErrAlignSilent reports that a block contained no signal above the silence
// threshold, so no delay estimate was possible. The aligned output is still
// valid: the aligner keeps the alignment it already had.
var ErrAlignSilent = errors.New("align: no signal in block, alignment unchanged")

// ErrAlignUnreliable reports that the block's two channels did not correlate
// strongly enough to estimate a delay (below MinAlignConfidence), so the
// alignment was left unchanged. It is typical of unrelated channels, heavy
// noise, or material with no shared source.
var ErrAlignUnreliable = errors.New("align: channels do not correlate, alignment unchanged")

// ErrAlignOutOfRange reports that the cross-correlation peak lay at the edge
// of AlignConfig.MaxDelay (or a stronger peak was found just past it), so the
// true delay may be larger than the search bound. The alignment was left
// unchanged; widen MaxDelay or leave the channels as they are.
var ErrAlignOutOfRange = errors.New("align: delay beyond MaxDelay, alignment unchanged")

// ErrAlignShortWindow reports that the supplied material is too short for the
// lag search: the window must exceed 2*MaxDelay plus a small minimum. A
// configuration whose Window is too small for its MaxDelay is a configuration
// error and is reported as ErrInvalidConfig instead.
var ErrAlignShortWindow = errors.New("align: too few samples for the delay search")

// MinAlignConfidence is the confidence below which an alignment estimate is
// treated as unreliable and not applied. It is the absolute normalized
// cross-correlation at the chosen delay; 0.2 means only a fifth of the energy
// is coherently shared, at which point shifting would be fitting noise. The
// value matches the internal estimator's floor and is exported so callers can
// apply the same rule to the Confidence they read from [AlignResult].
const MinAlignConfidence = align.MinConfidence

// DefaultAlignSilenceRMS is the block RMS level (about -80 dBFS) below which
// a block is treated as silence, so an alignment is never re-estimated from
// dither or room noise.
const DefaultAlignSilenceRMS = align.DefaultSilenceRMS

// DefaultAlignWindow is the default analysis window, in input samples, when
// AlignConfig.Window is zero: 16384 samples, 0.34 s at 48 kHz.
const DefaultAlignWindow = align.DefaultWindow

// AlignConfig configures a stereo alignment.
//
// The zero value is not valid: MaxDelay must be set. The other fields default
// to DefaultAlignWindow and DefaultAlignSilenceRMS, with Freeze false.
type AlignConfig struct {
	// MaxDelay is the largest inter-channel delay, in input samples, that the
	// search considers in either direction. It bounds the estimate and must be
	// positive; a delay at or beyond it is reported as out of range rather
	// than applied. Typical microphone offsets are a few samples to a few
	// hundred (0.1 ms to 10 ms); 1024 covers both at 48 kHz.
	MaxDelay int

	// Window is the analysis window, in input samples, used for each delay
	// estimate. It must be longer than 2*MaxDelay plus a small minimum;
	// longer windows resolve the fractional delay more accurately and survive
	// more noise. Zero uses DefaultAlignWindow.
	Window int

	// SilenceRMS is the RMS level below which a block is treated as silence
	// and the previous alignment retained, so dither or room noise can never
	// move the alignment. Zero uses DefaultAlignSilenceRMS.
	SilenceRMS float64

	// Freeze stops re-estimation after the first usable estimate and keeps
	// that alignment for the rest of the stream. Use it when the rig is fixed
	// and a quiet or noisy passage must not perturb the alignment.
	Freeze bool
}

// Validate checks the alignment configuration.
func (c *AlignConfig) Validate() error {
	if c.MaxDelay <= 0 {
		return fmt.Errorf("%w: MaxDelay must be positive", ErrInvalidConfig)
	}
	if c.MaxDelay > maxAlignDelay {
		return fmt.Errorf("%w: MaxDelay %d exceeds the %d-sample limit", ErrInvalidConfig, c.MaxDelay, maxAlignDelay)
	}
	window := c.Window
	if window == 0 {
		window = DefaultAlignWindow
	}
	if window < 2*c.MaxDelay+minAlignWindowSlack {
		return fmt.Errorf("%w: window %d is shorter than 2*MaxDelay+%d", ErrInvalidConfig, window, minAlignWindowSlack)
	}
	if c.SilenceRMS < 0 || math.IsNaN(c.SilenceRMS) {
		return fmt.Errorf("%w: SilenceRMS must be finite and non-negative", ErrInvalidConfig)
	}
	return nil
}

// Alignment limits shared with the internal package.
const (
	// maxAlignDelay bounds AlignConfig.MaxDelay. The lag search is
	// O(MaxDelay * window), so an unbounded value would let a caller request an
	// arbitrarily expensive estimate; 16384 samples is 0.34 s at 48 kHz.
	maxAlignDelay = 16384

	// minAlignWindowSlack is the minimum window length beyond the lag margin
	// the estimator needs to resolve a delay at all.
	minAlignWindowSlack = 256
)

// AlignResult is the outcome of a stereo delay estimate.
type AlignResult struct {
	// Delay is the delay of the right channel relative to the left, in
	// samples: a positive value means the right channel lags and is advanced
	// by Delay to align, a negative value means it leads and is retarded. The
	// value is fractional.
	Delay float64

	// Confidence is the absolute normalized cross-correlation at Delay, in
	// [0, 1]: 1 is an identical (or exactly inverted) signal, 0.5 is two
	// microphones with substantially different room sound, and near 0 is
	// unrelated material. Values below MinAlignConfidence are not applied.
	Confidence float64

	// Polarity is +1 for in-phase channels and -1 when one channel is the
	// other's inverse (a reversed cable or polarity-flipped capsule).
	// Alignment shifts the channels but does not invert them.
	Polarity float64

	// Silent reports that both channels were below the silence threshold, so
	// no estimate was made and any previous alignment is kept.
	Silent bool

	// OutOfRange reports that the correlation peak lay at or beyond
	// AlignConfig.MaxDelay, so the reported Delay must not be applied.
	OutOfRange bool
}

// Valid reports whether the estimate may be applied: not silent, not out of
// range, and at least MinAlignConfidence.
func (r AlignResult) Valid() bool {
	return !r.Silent && !r.OutOfRange && r.Confidence >= MinAlignConfidence
}

// errForResult maps an unusable AlignResult to its sentinel error.
func errForResult(r AlignResult) error {
	switch {
	case r.Silent:
		return ErrAlignSilent
	case r.OutOfRange:
		return ErrAlignOutOfRange
	case r.Confidence < MinAlignConfidence:
		return ErrAlignUnreliable
	default:
		return nil
	}
}

// errPairMismatch is the single message for mismatched channel lengths, kept
// as one literal so every entry point reports the mismatch identically.
const errPairMismatch = "channel chunks must have the same length: left %d, right %d"

// checkPair rejects channel chunks of different lengths, which cannot be
// corresponding frames of one stream.
func checkPair(left, right []float64) error {
	if len(left) != len(right) {
		return fmt.Errorf("%w: "+errPairMismatch, ErrInvalidConfig, len(left), len(right))
	}
	return nil
}

// checkFinite rejects a pair containing NaN or infinite samples. A single
// non-finite sample poisons the mean and variance, turning every correlation
// into NaN, so the estimate would be meaningless; the caller is told the input
// is invalid rather than being handed a silent or unreliable result that hides
// the real cause.
func checkFinite(channels ...[]float64) error {
	for _, ch := range channels {
		for _, v := range ch {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("%w: non-finite sample in input", ErrInvalidConfig)
			}
		}
	}
	return nil
}

// toAlignResult converts an internal estimate.
func toAlignResult(e align.Estimate) AlignResult {
	return AlignResult{
		Delay:      e.Delay,
		Confidence: e.Confidence,
		Polarity:   e.Polarity,
		Silent:     e.Silent,
		OutOfRange: e.OutOfRange,
	}
}

// EstimateChannelDelay estimates the delay between the left and right channels
// of a stereo pair over their full length (or the configured window, taken
// from the end). It is the stateless form of the estimate that
// [AlignedStereo] performs; use it to inspect a pair before committing to an
// alignment.
//
// An estimate that cannot be used (silence, unrelated channels, a delay beyond
// MaxDelay) is returned together with the corresponding sentinel error
// ([ErrAlignSilent], [ErrAlignUnreliable], [ErrAlignOutOfRange]) so a caller
// can report why. The two channels must have the same length.
func EstimateChannelDelay(left, right []float64, cfg AlignConfig) (AlignResult, error) {
	if err := cfg.Validate(); err != nil {
		return AlignResult{}, err
	}
	if err := checkPair(left, right); err != nil {
		return AlignResult{}, err
	}
	if err := checkFinite(left, right); err != nil {
		return AlignResult{}, err
	}
	window := cfg.Window
	if window == 0 {
		window = DefaultAlignWindow
	}
	if len(left) > window {
		left = left[len(left)-window:]
		right = right[len(right)-window:]
	}
	e, err := align.EstimateDelay(left, right, align.Config{
		MaxDelay:   cfg.MaxDelay,
		SilenceRMS: cfg.SilenceRMS,
	})
	if err != nil {
		if errors.Is(err, align.ErrShortWindow) {
			return AlignResult{}, fmt.Errorf("%w: %v", ErrAlignShortWindow, err)
		}
		return AlignResult{}, fmt.Errorf("%w: %v", ErrInvalidConfig, err)
	}
	res := toAlignResult(e)
	return res, errForResult(res)
}

// AlignStereo removes the inter-channel delay of a stereo pair: it estimates
// the delay of right relative to left and shifts the early channel by that
// fractional number of samples, returning the aligned pair. Both channels are
// delayed by a common [AlignedLateness] samples so the (possibly negative)
// offset can be applied causally; that common delay shifts the pair's
// timeline but cancels out of the inter-channel relationship.
//
// The polarity is not corrected: an inverted channel is reported through
// Polarity so the caller can decide.
//
// An unusable estimate (silence, unrelated channels, a delay beyond
// MaxDelay) is returned as an error and no output is produced
// (alignedLeft and alignedRight are nil). Use EstimateChannelDelay to
// inspect the estimate before calling this.
func AlignStereo(left, right []float64, cfg AlignConfig) (alignedLeft, alignedRight []float64, res AlignResult, err error) {
	if err := cfg.Validate(); err != nil {
		return nil, nil, AlignResult{}, err
	}
	if err := checkPair(left, right); err != nil {
		return nil, nil, AlignResult{}, err
	}
	res, err = EstimateChannelDelay(left, right, cfg)
	if err != nil {
		return nil, nil, res, err
	}
	if !res.Valid() {
		// EstimateChannelDelay already returned the matching sentinel error
		// for every unusable result; this is a guard against the two rules
		// drifting apart, and keeps the channels untouched rather than
		// applying a delay the estimator rejected.
		return nil, nil, res, ErrAlignUnreliable
	}
	// Both channels get the common base delay; the early one gets the extra
	// offset. A positive estimate means the right lags, so the left is the one
	// retarded by the extra.
	base := alignBaseDelay
	l, lerr := align.NewAligner(base+math.Max(res.Delay, 0), maxAlignLateness(cfg.MaxDelay))
	if lerr != nil {
		return nil, nil, res, fmt.Errorf("%w: %v", ErrInvalidConfig, lerr)
	}
	r, rerr := align.NewAligner(base+math.Max(-res.Delay, 0), maxAlignLateness(cfg.MaxDelay))
	if rerr != nil {
		return nil, nil, res, fmt.Errorf("%w: %v", ErrInvalidConfig, rerr)
	}
	alignedLeft = l.Process(left)
	alignedRight = r.Process(right)
	return alignedLeft, alignedRight, res, nil
}

// AlignedLateness returns the common delay, in samples, that AlignStereo and
// AlignedStereo add to both channels on top of the inter-channel correction.
// It is the delay kernel's group delay, independent of the configuration: the
// inter-channel offset is carried by the early channel's own aligner, so it
// never changes the common delay.
func AlignedLateness() int {
	return alignHalfTaps
}

// alignBaseDelay is the common delay both channels receive, as a float for the
// aligner constructors.
const alignBaseDelay = float64(alignHalfTaps)

// maxAlignLateness is the upper delay bound passed to the aligners: the base
// plus the largest possible offset plus room for the kernel's own history.
func maxAlignLateness(maxDelay int) int {
	return alignHalfTaps + maxDelay + alignHalfTaps
}

// AlignedStereo is a streaming stereo aligner: it estimates and applies the
// inter-channel delay across blocks and resamples the aligned pair, so a
// stereo stream that was captured with a few frames' offset per microphone is
// converted as one coherent image instead of a comb-filtered sum.
//
// It combines an alignment session (delay estimation across blocks, held
// state, chunk-size invariance, silence handling) with two resamplers (one per
// channel, both float64). The streams are independent: align first, then
// resample, so the delay is corrected at the input rate where it was measured.
//
// The zero value is not usable; construct with [NewAlignedStereo]. A stream
// is: NewAlignedStereo, Process per chunk, Flush once at the end. The
// alignment and the resamplers keep their state between calls, so the output
// for a fixed input does not depend on the block size.
type AlignedStereo struct {
	session *align.Session
	left    *SimpleResampler
	right   *SimpleResampler
	ratio   float64
}

// NewAlignedStereo creates a streaming stereo align-and-resample stage.
//
// The resampling quality applies to both channels. The alignment configuration
// is validated; MaxDelay must cover the largest expected offset.
func NewAlignedStereo(inputRate, outputRate float64, quality QualityPreset, cfg AlignConfig) (*AlignedStereo, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	// Validate the rates and ratio before building anything, so a bad
	// conversion is rejected as ErrInvalidConfig rather than as a downstream
	// engine error.
	probe := Config{
		InputRate:  inputRate,
		OutputRate: outputRate,
		Channels:   stereoChannels,
		Quality:    QualitySpec{Preset: quality},
	}
	if err := probe.Validate(); err != nil {
		return nil, err
	}
	session, err := align.NewSession(align.SessionOptions{
		MaxDelay:   cfg.MaxDelay,
		Window:     cfg.Window,
		SilenceRMS: cfg.SilenceRMS,
		Freeze:     cfg.Freeze,
	})
	if err != nil {
		if errors.Is(err, align.ErrShortWindow) {
			return nil, fmt.Errorf("%w: %v", ErrAlignShortWindow, err)
		}
		return nil, fmt.Errorf("%w: %v", ErrInvalidConfig, err)
	}
	l, err := NewEngine(inputRate, outputRate, quality)
	if err != nil {
		return nil, err
	}
	r, err := NewEngine(inputRate, outputRate, quality)
	if err != nil {
		return nil, err
	}
	return &AlignedStereo{
		session: session,
		left:    l,
		right:   r,
		ratio:   l.GetRatio(),
	}, nil
}

// Process aligns and resamples one block of both channels, returning the
// resampled left and right output. The two input chunks must be the same
// length.
//
// The returned AlignResult describes the alignment state for the block. A
// silent, unreliable or out-of-range estimate is reported in the result
// (Silent, OutOfRange, low Confidence) and the previous alignment is kept; it
// is not an error, because a stream may legitimately pass through silence.
// Process itself returns an error only for malformed input or a resampler
// failure.
func (a *AlignedStereo) Process(left, right []float64) (leftOut, rightOut []float64, res AlignResult, err error) {
	if err := checkPair(left, right); err != nil {
		return nil, nil, AlignResult{}, err
	}
	al, ar, est, err := a.session.Process(left, right)
	if err != nil {
		return nil, nil, AlignResult{}, fmt.Errorf("%w: %v", ErrInvalidConfig, err)
	}
	leftOut, err = a.left.Process(al)
	if err != nil {
		return nil, nil, toAlignResult(est), err
	}
	rightOut, err = a.right.Process(ar)
	if err != nil {
		return nil, nil, toAlignResult(est), err
	}
	return leftOut, rightOut, toAlignResult(est), nil
}

// ProcessInto aligns and resamples one block into caller-owned output
// buffers, returning the number of output samples written. The output buffers
// must each hold [AlignedStereo.EstimateOutput] samples for the block.
//
// If either buffer is too small it returns [ErrBufferTooSmall] before any
// state is advanced, so the caller can retry with larger buffers.
func (a *AlignedStereo) ProcessInto(left, right, leftOut, rightOut []float64) (int, AlignResult, error) {
	if err := checkPair(left, right); err != nil {
		return 0, AlignResult{}, err
	}
	required := a.EstimateOutput(len(left))
	if len(leftOut) < required || len(rightOut) < required {
		return 0, AlignResult{}, ErrBufferTooSmall
	}
	// The alignment session must not advance if the resampler path fails, and
	// the resampler's ProcessInto must not advance on an undersized buffer.
	// Both are already true, and the buffers were checked above, so the state
	// advances only once both channels are written.
	al, ar, est, err := a.session.Process(left, right)
	if err != nil {
		return 0, AlignResult{}, fmt.Errorf("%w: %v", ErrInvalidConfig, err)
	}
	nl, err := a.left.ProcessInto(al, leftOut)
	if err != nil {
		return 0, toAlignResult(est), err
	}
	nr, err := a.right.ProcessInto(ar, rightOut)
	if err != nil {
		return 0, toAlignResult(est), err
	}
	return min(nl, nr), toAlignResult(est), nil
}

// EstimateOutput returns the maximum number of output samples one channel of
// a ProcessInto call for an input of inputLen samples may produce. It is the
// resampler's conservative bound.
func (a *AlignedStereo) EstimateOutput(inputLen int) int {
	return a.left.EstimateOutput(inputLen)
}

// Flush drains both resamplers at end of stream and returns the remaining
// samples. It is end-of-stream only, exactly like the underlying resamplers:
// call it once, after the last Process; a second call returns empty slices.
func (a *AlignedStereo) Flush() (leftOut, rightOut []float64, res AlignResult, err error) {
	est, _ := a.session.Estimate()
	leftOut, err = a.left.Flush()
	if err != nil {
		return nil, nil, toAlignResult(est), err
	}
	rightOut, err = a.right.Flush()
	if err != nil {
		return nil, nil, toAlignResult(est), err
	}
	return leftOut, rightOut, toAlignResult(est), nil
}

// Latency returns the startup deficit, in output samples: how many samples the
// first Process calls withhold while the resampler filters prime. It is the
// same figure as the underlying resampler's Latency(), because the alignment
// stage delays each channel by a fixed number of samples but withholds none:
// every input sample yields an output sample in the same call.
//
// The fixed delay itself is separate and does not vary: the aligned pair is
// shifted by AlignedLateness() input samples relative to the original input
// (unchanged between the two channels, so the corrected image is coherent).
// Output sample n therefore corresponds to input time n/ratio minus
// AlignedLateness()/inputRate, plus the usual startup deficit. A caller that
// needs the absolute timeline back adds AlignedLateness()*ratio output samples
// of silence to its FIFO on top of Latency().
func (a *AlignedStereo) Latency() int {
	return a.left.Latency()
}

// Alignment returns the alignment currently applied and whether a usable
// estimate has been found.
func (a *AlignedStereo) Alignment() (AlignResult, bool) {
	est, ok := a.session.Estimate()
	return toAlignResult(est), ok
}

// Reset discards the alignment and both resamplers' state, returning the stage
// to its just-constructed condition. It is for starting an unrelated stream.
func (a *AlignedStereo) Reset() {
	a.session.Reset()
	a.left.Reset()
	a.right.Reset()
}

// GetRatio returns the resampling ratio (outputRate/inputRate).
func (a *AlignedStereo) GetRatio() float64 {
	return a.ratio
}

// alignHalfTaps is the delay kernels' half-length and group delay, taken from
// the internal package so the public lateness and the implementation cannot
// drift apart.
const alignHalfTaps = align.HalfTaps
