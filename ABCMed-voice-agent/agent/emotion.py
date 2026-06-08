"""Emotion analysis from audio features using librosa."""

import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class VoiceEmotionAnalyzer:
    """Analyzes voice features to estimate emotional state.
    
    Uses pitch, energy, speech rate, and spectral features
    as indicators of emotional arousal and valence.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.baseline_pitch = None
        self.baseline_energy = None

    def analyze_audio(self, audio: np.ndarray) -> dict:
        """Analyze audio segment for emotional indicators."""
        if not LIBROSA_AVAILABLE:
            return self._default_result()

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        if len(audio) < self.sample_rate * 0.5:
            return self._default_result()

        features = {}

        # Pitch (F0) - higher pitch = more arousal/stress
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=60, fmax=500, sr=self.sample_rate
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(f0_valid) > 0:
            features["pitch_mean"] = float(np.mean(f0_valid))
            features["pitch_std"] = float(np.std(f0_valid))
            features["pitch_range"] = float(np.ptp(f0_valid))
        else:
            features["pitch_mean"] = 0
            features["pitch_std"] = 0
            features["pitch_range"] = 0

        # Energy / loudness - higher = more intense emotion
        rms = librosa.feature.rms(y=audio)[0]
        features["energy_mean"] = float(np.mean(rms))
        features["energy_std"] = float(np.std(rms))

        # Speech rate estimate (via onset detection)
        onsets = librosa.onset.onset_detect(y=audio, sr=self.sample_rate)
        duration = len(audio) / self.sample_rate
        features["speech_rate"] = len(onsets) / duration if duration > 0 else 0

        # Spectral centroid - brighter voice = more arousal
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features["spectral_centroid_mean"] = float(np.mean(centroid))

        # MFCC statistics for voice quality
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features["mfcc_mean"] = [float(x) for x in np.mean(mfccs, axis=1)]

        # Derive emotional indicators
        arousal = self._estimate_arousal(features)
        voice_profile = self._estimate_voice_profile(features)

        return {
            "arousal": arousal,  # 0-1, higher = more agitated
            "voice_stress_indicators": {
                "high_pitch": features["pitch_mean"] > 200,
                "pitch_variability": features["pitch_std"] > 40,
                "high_energy": features["energy_mean"] > 0.05,
                "fast_speech": features["speech_rate"] > 4.0,
            },
            "voice_profile": voice_profile,
            "raw_features": features,
        }

    def _estimate_arousal(self, features: dict) -> float:
        """Estimate arousal level (0=calm, 1=highly agitated)."""
        scores = []

        # Pitch contribution
        pitch = features["pitch_mean"]
        if pitch > 0:
            pitch_score = min(1.0, max(0.0, (pitch - 100) / 200))
            scores.append(pitch_score)

        # Pitch variability
        pitch_var = features["pitch_std"]
        var_score = min(1.0, pitch_var / 60)
        scores.append(var_score)

        # Energy
        energy = features["energy_mean"]
        energy_score = min(1.0, energy / 0.08)
        scores.append(energy_score)

        # Speech rate
        rate = features["speech_rate"]
        rate_score = min(1.0, max(0.0, (rate - 2) / 4))
        scores.append(rate_score)

        return float(np.mean(scores)) if scores else 0.5

    def _estimate_voice_profile(self, features: dict) -> dict:
        """Rough voice profile estimation from acoustic features."""
        pitch = features["pitch_mean"]

        # Very rough gender estimation from F0
        if pitch > 165:
            estimated_gender = "kobieta"
        elif pitch < 130:
            estimated_gender = "mężczyzna"
        else:
            estimated_gender = "nieokreślone"

        # Very rough age group from voice characteristics
        spectral_cent = features["spectral_centroid_mean"]
        if pitch > 250:
            age_group = "dziecko"
        elif spectral_cent < 1500 and pitch < 140:
            age_group = "senior"
        else:
            age_group = "dorosły"

        return {
            "estimated_gender": estimated_gender,
            "estimated_age_group": age_group,
        }

    def _default_result(self) -> dict:
        return {
            "arousal": 0.5,
            "voice_stress_indicators": {
                "high_pitch": False,
                "pitch_variability": False,
                "high_energy": False,
                "fast_speech": False,
            },
            "voice_profile": {
                "estimated_gender": "nieokreślone",
                "estimated_age_group": "dorosły",
            },
            "raw_features": {},
        }
