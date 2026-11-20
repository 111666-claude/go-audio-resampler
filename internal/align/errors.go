// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package align

import "errors"

// Errors returned by the estimator and the aligner. They are wrapped by the
// root package's ErrInvalidConfig / ErrAlignShortWindow so callers can test
// with errors.Is against either layer.
var (
	// errBadMaxDelay reports a non-positive Config.MaxDelay.
	errBadMaxDelay = errors.New("align: MaxDelay must be positive")
	// errBadSilence reports a negative or NaN silence threshold.
	errBadSilence = errors.New("align: SilenceRMS must be finite and non-negative")
	// ErrShortWindow reports a window too short for the lag search. It is
	// exported so the root package can map it to ErrAlignShortWindow.
	ErrShortWindow = errors.New("align: too few samples for the delay search")
	// errBadDelay reports a delay that is NaN, infinite, or past the search
	// bound the aligner was constructed with.
	errBadDelay = errors.New("align: delay is not finite or exceeds MaxDelay")
)

// errBufferTooSmall reports a destination shorter than the input chunk.
var errBufferTooSmall = errors.New("align: destination buffer too small")
