from __future__ import annotations

import unittest

import numpy as np

from voice_features import SAMPLE_RATE, feature_vector, prepare_audio


class FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        samples = np.arange(SAMPLE_RATE * 4, dtype=np.float32) / SAMPLE_RATE
        self.tone = 0.1 * np.sin(2 * np.pi * 440 * samples)

    def test_librosa_channel_layout_becomes_mono(self) -> None:
        stereo_channels_first = np.vstack((self.tone, self.tone))
        mono = prepare_audio(stereo_channels_first, SAMPLE_RATE)
        self.assertEqual(mono.shape, (SAMPLE_RATE * 4,))
        np.testing.assert_allclose(mono, self.tone, rtol=0, atol=1e-7)

    def test_live_channel_layout_becomes_mono(self) -> None:
        stereo_frames_first = np.column_stack((self.tone, self.tone))
        mono = prepare_audio(stereo_frames_first, SAMPLE_RATE)
        self.assertEqual(mono.shape, (SAMPLE_RATE * 4,))
        np.testing.assert_allclose(mono, self.tone, rtol=0, atol=1e-7)

    def test_feature_vector_is_finite_and_stable_size(self) -> None:
        vector = feature_vector(self.tone, SAMPLE_RATE)
        self.assertEqual(vector.shape, (86,))
        self.assertTrue(np.isfinite(vector).all())


if __name__ == "__main__":
    unittest.main()
