// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

package align

import "sync"

// The delay kernel bank depends only on the design constants, so it is built
// once per process and shared read-only by every Aligner. Building it costs a
// few hundred thousand transcendental calls; a stereo stream that recreates
// its aligner on every alignment update must not repay that.
var (
	bankOnce sync.Once
	bank     [][]float64
)
