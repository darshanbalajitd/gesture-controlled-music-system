import argparse
import json
from collections import deque
from pathlib import Path
import os
import time

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import vlc  # pip install python-vlc

try:
    import mediapipe as mp
except Exception as e:
    print('mediapipe import failed. Ensure compatible versions, e.g.:')
    print('  pip install mediapipe==0.10.14 protobuf==3.20.*')
    raise

try:
    from google.protobuf import __version__ as _pb_ver
    if int(_pb_ver.split('.')[0]) >= 4:
        print('protobuf>=4 detected; install mediapipe==0.10.14 protobuf==3.20.*')
except Exception:
    pass

# ===================== MLP MODEL =====================

class MLP(nn.Module):
    def __init__(self, in_dim, h1, h2, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ===================== FEATURE EXTRACTION =====================

def extract_features(image_bgr, hands):
    """
    Returns a flat np.array of shape (63,) = 21 landmarks * (x,y,z)
    or None if no hand is detected.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(image_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    vals = []
    for i in range(21):
        p = lm.landmark[i]
        vals.extend([p.x, p.y, p.z])
    return np.array(vals, dtype=np.float32)


# ===================== AUDIO HELPERS =====================

AUDIO_EXTS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.MP3', '.WAV'}


def list_audio_files(root: Path):
    if not root.exists():
        return []
    return [
        p for p in root.rglob('*')
        if p.is_file() and p.suffix in AUDIO_EXTS
    ]


# ===================== MUSIC PLAYER =====================

class MusicPlayer:
    """
    Handles:
      - loading playlists by genre
      - next / previous track
      - volume up/down
      - continuous playback via VLC
    """

    def __init__(self, music_dir: Path, genre_map: dict):
        self.music_dir = music_dir
        self.genre_map = genre_map  # gesture label -> folder name
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.current_playlist = []  # list[Path]
        self.current_index = 0
        self.volume = 70
        self.player.audio_set_volume(self.volume)

    def _build_playlist(self, folder_name: str):
        genre_path = self.music_dir / folder_name
        if not genre_path.exists():
            print(f"[MusicPlayer] Genre folder does not exist: {genre_path}")
            self.current_playlist = []
            return

        files = [
            p for p in genre_path.rglob('*')
            if p.is_file() and p.suffix in AUDIO_EXTS
        ]
        files.sort()
        self.current_playlist = files
        if not self.current_playlist:
            print(f"[MusicPlayer] No audio files in {genre_path}")

    def _play_current(self):
        if not self.current_playlist:
            print("[MusicPlayer] No playlist loaded.")
            return
        track = self.current_playlist[self.current_index]
        print(f"[MusicPlayer] Playing: {track}")
        media = self.instance.media_new(str(track))
        self.player.set_media(media)
        self.player.play()

    def load_genre(self, gesture_label: str):
        """
        gesture_label: 'genre-1' .. 'genre-5'
        """
        folder_name = self.genre_map.get(gesture_label)
        if folder_name is None:
            print(f"[MusicPlayer] Unknown genre gesture: {gesture_label}")
            return
        print(f"[MusicPlayer] Loading genre '{gesture_label}' -> folder '{folder_name}'")
        self._build_playlist(folder_name)
        if self.current_playlist:
            self.current_index = 0
            self._play_current()

    def next_track(self):
        if not self.current_playlist:
            print("[MusicPlayer] next_track: no playlist")
            return
        self.current_index = (self.current_index + 1) % len(self.current_playlist)
        self._play_current()

    def previous_track(self):
        if not self.current_playlist:
            print("[MusicPlayer] previous_track: no playlist")
            return
        self.current_index = (self.current_index - 1) % len(self.current_playlist)
        self._play_current()

    def volume_up(self, step=2):
        self.volume = min(100, self.volume + step)
        self.player.audio_set_volume(self.volume)
        print(f"[MusicPlayer] Volume up -> {self.volume}")

    def volume_down(self, step=2):
        self.volume = max(0, self.volume - step)
        self.player.audio_set_volume(self.volume)
        print(f"[MusicPlayer] Volume down -> {self.volume}")

    def pause(self):
        self.player.pause()

    def resume(self):
        self._play_current()

    def toggle_play_pause(self):
        state = self.player.get_state()
        if str(state) == 'State.Playing':
            self.pause()
        else:
            self.resume()

    def stop(self):
        self.player.stop()


# ===================== PROMPT SOUND PLAYER =====================

class PromptPlayer:
    """
    Plays short indicator sounds (one-shot) for gestures,
    using a separate VLC player so it doesn't interfere with music playback.
    """

    def __init__(self, sounds_dir: Path):
        self.sounds_dir = sounds_dir
        self.instance = vlc.Instance()
        # Map gesture labels to base filenames (without extension)
        self.sound_map = {
            'genre-1': 'hindi_audio',
            'genre-2': 'indie_audio',
            'genre-3': 'kpop_audio',
            'genre-4': 'phonk_audio',     # you will add this later
            'genre-5': 'rap_audio',
            'next': 'next_audio',      # generic sound for next
            'previous': 'previous_audio',
            'volume-up': 'increase_audio',
            'volume-down': 'decrease_audio',
            'play-pause': None,
            'stop': 'stop_audio',
        }
        self.generic_name = 'playing_audio'  # fallback if specific not found
        self.loop_player = self.instance.media_player_new()
        self.play_pause_first_names = ['pausing_audio']
        self.play_pause_next_names = ['playing_audio']

    def _find_sound_file(self, base_name: str) -> Path | None:
        """
        Try multiple extensions for a given base name.
        """
        for ext in AUDIO_EXTS:
            p = self.sounds_dir / f"{base_name}{ext}"
            if p.exists():
                return p
        return None

    def play_prompt(self, gesture_label: str):
        base_name = self.sound_map.get(gesture_label)
        sound_path = None
        if base_name:
            sound_path = self._find_sound_file(base_name)
        if sound_path is None:
            for p in list_audio_files(self.sounds_dir):
                if gesture_label in p.stem:
                    sound_path = p
                    break
        if sound_path is None and self.generic_name:
            sound_path = self._find_sound_file(self.generic_name)
        if sound_path is None:
            print(f"[PromptPlayer] No prompt sound found for '{gesture_label}'")
            return
        player = self.instance.media_player_new()
        media = self.instance.media_new(str(sound_path))
        player.set_media(media)
        player.play()

    def play_prompt_loop(self, gesture_label: str):
        base_name = self.sound_map.get(gesture_label)
        sound_path = None
        if base_name:
            sound_path = self._find_sound_file(base_name)
        if sound_path is None:
            for p in list_audio_files(self.sounds_dir):
                if gesture_label in p.stem:
                    sound_path = p
                    break
        if sound_path is None:
            return
        media = self.instance.media_new(str(sound_path))
        media.add_option('input-repeat=-1')
        self.loop_player.set_media(media)
        self.loop_player.play()

    def stop_loop(self):
        self.loop_player.stop()

    def _play_base_names(self, names):
        for name in names:
            p = self._find_sound_file(name)
            if p is not None:
                player = self.instance.media_player_new()
                media = self.instance.media_new(str(p))
                player.set_media(media)
                player.play()
                return True
        return False

    def play_play_pause_first(self):
        if not self._play_base_names(self.play_pause_first_names):
            print("[PromptPlayer] pausing_audio not found")

    def play_play_pause_next(self):
        if not self._play_base_names(self.play_pause_next_names):
            print("[PromptPlayer] playing_audio not found")


# ===================== MAIN REALTIME LOOP =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=Path.cwd() / 'models' / 'mlp_state.pth')
    ap.add_argument('--scaler', type=Path, default=Path.cwd() / 'models' / 'scaler.pkl')
    ap.add_argument('--labels', type=Path, default=Path.cwd() / 'models' / 'label_map.json')
    ap.add_argument('--camera-index', type=int, default=0)
    ap.add_argument('--min-conf', type=float, default=0.5)   # mediapipe detection conf
    ap.add_argument('--threshold', type=float, default=0.5)  # softmax confidence
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--stable-frames', type=int, default=8)
    ap.add_argument('--cooldown-secs', type=float, default=5.0)
    ap.add_argument('--sounds-dir', type=Path, default=Path.cwd() / 'sounds')
    ap.add_argument('--music-dir', type=Path, default=Path.cwd() / 'music')
    args = ap.parse_args()

    # ----- Load MLP model -----
    payload = torch.load(args.model, map_location='cpu')
    in_dim = int(payload['in_dim'])
    h1 = int(payload['h1'])
    h2 = int(payload['h2'])
    out_dim = int(payload['out_dim'])
    dropout = float(payload['dropout'])

    model = MLP(in_dim, h1, h2, out_dim, dropout)
    model.load_state_dict(payload['state_dict'])

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # ----- Load scaler and labels -----
    scaler = joblib.load(args.scaler)
    with open(args.labels, 'r', encoding='utf-8') as f:
        labels = json.load(f)['classes']

    # ----- Init MediaPipe Hands -----
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_conf
    )

    # ----- Camera -----
    cap = cv2.VideoCapture(args.camera_index)

    # ----- Stats & smoothing -----
    conf_hist = deque(maxlen=30)
    total_frames = 0
    detected_frames = 0
    above_thresh_frames = 0
    recent = deque(maxlen=args.stable_frames)
    last_trigger = None
    cooldown_until = 0.0
    first_play_pause = False
    actions_locked = False
    play_pause_count = 0

    # ----- Gesture → genre folder map -----
    genre_map = {
        'genre-1': 'hindi',
        'genre-2': 'indie',
        'genre-3': 'k-pop',
        'genre-4': 'phonk',
        'genre-5': 'rap',
    }

    # ----- Initialize Music & Prompt players -----
    music_player = MusicPlayer(args.music_dir, genre_map)
    prompt_player = PromptPlayer(args.sounds_dir)

    print("[INFO] Starting realtime gesture recognition. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to read from camera.")
            break

        total_frames += 1
        text = 'no hand'
        conf = 0.0
        infer_time = 0.0

        # ---------- Feature extraction ----------
        feat = extract_features(frame, hands)

        if feat is not None and feat.shape[0] == in_dim:
            detected_frames += 1
            x = scaler.transform(feat.reshape(1, -1)).astype(np.float32)
            xb = torch.from_numpy(x).to(device)

            start_inf = time.time()  # <--- START LATENCY TIMER

            with torch.no_grad():
                logits = model(xb)

                infer_time = (time.time() - start_inf) * 1000  # <--- LATENCY (ms)

                p = torch.softmax(logits, dim=1)[0]

                idx = int(torch.argmax(p).item())
                conf = float(p[idx].item())

            if conf >= args.threshold:
                above_thresh_frames += 1

            label = labels[idx] if idx < len(labels) else str(idx)
            text = f'{label} p={conf:.2f}'

            # Add to recent history for stability check
            recent.append((label, conf))

            # ---------- Trigger logic ----------
            now = time.time()
            if now >= cooldown_until and len(recent) == recent.maxlen:
                # Count how many times each label appears with conf >= threshold
                lab_count = {}
                for l, c in recent:
                    if c >= args.threshold:
                        lab_count[l] = lab_count.get(l, 0) + 1

                if lab_count:
                    best_label, best_count = max(lab_count.items(), key=lambda k: k[1])
                    required = int(0.7 * recent.maxlen)
                    if best_count >= required:
                        print(f"[TRIGGER] Gesture: {best_label}, count={best_count}/{recent.maxlen}")

                        if best_label == 'stop':
                            prompt_player.play_prompt('stop')
                            prompt_player.stop_loop()
                            music_player.stop()
                            break

                        if best_label == 'play-pause':
                            play_pause_count += 1
                            if play_pause_count % 2 == 1:
                                prompt_player.play_play_pause_first()
                            else:
                                prompt_player.play_play_pause_next()
                            music_player.toggle_play_pause()
                            actions_locked = not actions_locked
                        else:
                            if actions_locked:
                                print("[LOCK] Actions locked. Perform play-pause to resume.")
                            else:
                                if best_label in genre_map:
                                    prompt_player.play_prompt(best_label)
                                    music_player.load_genre(best_label)
                                elif best_label == 'next':
                                    prompt_player.play_prompt('next')
                                    music_player.next_track()
                                elif best_label == 'previous':
                                    prompt_player.play_prompt('previous')
                                    music_player.previous_track()
                                elif best_label == 'volume-up':
                                    prompt_player.play_prompt('volume-up')
                                    music_player.volume_up(step=2)
                                elif best_label == 'volume-down':
                                    prompt_player.play_prompt('volume-down')
                                    music_player.volume_down(step=2)
                                else:
                                    print(f"[WARN] Unmapped gesture label: {best_label}")

                        cooldown_until = now + args.cooldown_secs

        # ---------- UI overlay ----------
        conf_hist.append(conf)
        mean_conf = float(np.mean(conf_hist)) if conf_hist else 0.0
        cv2.putText(frame, f'latency={infer_time:.2f}ms', (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'mean_p={mean_conf:.2f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f'frames={total_frames} det={detected_frames} above={above_thresh_frames}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imshow('gesture-mlp', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------- Cleanup ----------
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exiting.")


if __name__ == '__main__':
    main()
