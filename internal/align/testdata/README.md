# Internal align test fixtures

`stereo_march_44k_s16.pcm` is a 32768-frame excerpt (44.1 kHz, 16-bit signed
little-endian, interleaved L/R) from *Farewell of Slavianka*, recorded by the
United States Coast Guard Band and released into the public domain
([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Farewell_of_Slavianka_(stereo).ogg)).

The excerpt was taken from the loud, broadband final section of the recording
(sample offset 7550976), scaled by one common factor to a peak of 0.9 and
quantised to 16 bits; no other processing was applied, so the two channels'
true relative timing is the original recording's. It is used by
`realdata_test.go` as a real stereo pair: the estimator's analysis window on
this material sees a broad spectrum with no dominant low-frequency lobe, and
injected fractional delays are recovered to the same accuracy as on the
synthetic vectors.

Obtained 2026-11-20. Public domain (recorded by a U.S. government band); the
excerpt is reproduced unchanged apart from the gain and 16-bit quantisation
described above.
