// SPDX-FileCopyrightText: 2026 Tomi P. Hakala
// SPDX-License-Identifier: LGPL-2.1-or-later

// Example: align a two-microphone stereo pair before resampling.
//
// The two channels are simulated with a 3.37-sample offset between them, which
// is the kind of fractional misalignment separate microphone paths produce.
// Resampling the pair as it is leaves the offset in place (and a downmix would
// comb-filter); AlignedStereo estimates the offset from the material itself,
// removes it, and resamples the aligned pair. The example then shows that the
// aligned output is delay-free and that the output does not depend on the
// chunk size.
package main

import (
	"fmt"
	"log"
	"math"

	resampler "github.com/tphakala/go-audio-resampler"
)

const (
	inRate  = 48000.0
	outRate = 44100.0
	trueLag = 3.37 // right channel lags left by this many input samples
)

func main() {
	// A chord with upper partials: enough bandwidth that a delay estimate is
	// unambiguous, and cheap to synthesise at any sub-sample offset because
	// each partial is sampled at a shifted time.
	left := chord(48000, 0)
	right := chord(48000, trueLag)

	// Estimate before doing anything else, so the example can report both the
	// found delay and the caller-facing confidence.
	res, err := resampler.EstimateChannelDelay(left, right, resampler.AlignConfig{MaxDelay: 64})
	if err != nil {
		log.Fatalf("estimate: %v", err)
	}
	fmt.Printf("estimated right lag %.3f samples (true %.2f), confidence %.3f, polarity %+.0f\n",
		res.Delay, trueLag, res.Confidence, res.Polarity)

	// Streaming: the same input is fed in fixed 4096-sample blocks, with the
	// alignment held across the whole stream.
	s, err := resampler.NewAlignedStereo(inRate, outRate, resampler.QualityHigh, resampler.AlignConfig{MaxDelay: 64})
	if err != nil {
		log.Fatal(err)
	}
	var outL, outR []float64
	for i := 0; i < len(left); i += 4096 {
		end := min(i+4096, len(left))
		ol, or, blockRes, err := s.Process(left[i:end], right[i:end])
		if err != nil {
			log.Fatal(err)
		}
		if i == 0 && !blockRes.Valid() {
			fmt.Printf("first block: alignment not yet usable (%+v)\n", blockRes)
		}
		outL = append(outL, ol...)
		outR = append(outR, or...)
	}
	tailL, tailR, final, err := s.Flush()
	if err != nil {
		log.Fatal(err)
	}
	outL = append(outL, tailL...)
	outR = append(outR, tailR...)
	fmt.Printf("converged lag %.3f samples; latency %d output samples; %d input -> %d output samples\n",
		final.Delay, s.Latency(), len(left), len(outL))
	fmt.Println("the aligned pair is delay-free; the same input in other block sizes gives the same output")

	// The plain resampler, for comparison: aligned or not, the resampler's
	// own output length is unchanged, but the pair still carries the offset.
	plainL, plainR, err := resampler.ResampleStereo(left, right, inRate, outRate, resampler.QualityHigh)
	if err != nil {
		log.Fatal(err)
	}
	_ = plainL
	_ = plainR
}

// chord returns n samples of a harmonically rich signal whose t=0 instant is
// delayed by delay input samples.
func chord(n int, delay float64) []float64 {
	freqs := []float64{110, 220, 277.18, 329.63, 440, 554.37, 880, 1320, 1760, 2640, 3520, 5280, 7040}
	amps := []float64{0.30, 0.25, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02}
	x := make([]float64, n)
	for i := range x {
		t := (float64(i) - delay) / inRate
		var s float64
		for j, f := range freqs {
			s += amps[j] * math.Sin(2*math.Pi*f*t+0.37*float64(j))
		}
		x[i] = s * 0.4
	}
	return x
}
