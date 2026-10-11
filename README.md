# Rave

A self-hosted take on [rave.dj](https://rave.dj) that makes **Pitch Perfect-style mashups**. Search YouTube or Spotify (or upload your own files), line up two or more songs, and it:

1. **Fetches** the audio: YouTube through [yt-dlp](https://github.com/yt-dlp/yt-dlp). Spotify streams can't be downloaded, so a Spotify track is matched to the same recording on YouTube. The matcher looks at length, title, the artist's channel or "- Topic" audio, and skips live/remix/cover versions.
2. **Separates** each song into vocals and backing with [Demucs](https://github.com/facebookresearch/demucs) (htdemucs).
3. **Analyzes** tempo, beat grid, downbeats and key (Krumhansl key profiles, shown with Camelot codes).
4. **Matches** every song to a *base* song. It time-stretches to the base tempo, folding half-time and double-time songs instead of stretching them 2x. It transposes into the base key, treating a minor key and its relative major as the same key. This uses Rubber Band with formant preservation, so pitched voices don't go chipmunk.
5. **Arranges** it like a riff-off:
   - an intro on the base groove
   - every song takes a full vocal turn
   - the turns get shorter and start to overlap, with the next voice answering underneath
   - a riser and a one-bar **a cappella hit**
   - **every chorus stacked at once**, spread across the stereo field, and out

   Each backing passage is picked by how well its chords fit the voice singing over it (bar-by-bar chroma match).
6. **Mixes** the result. It level-matches the stems, high-passes the vocals, ducks the backing under the vocals and crossfades on the bar grid. Then it masters to a loud, limited file (MP3/WAV/FLAC/M4A) with a cue sheet.

Everything runs in one Docker container with a web UI.

## Run it

```sh
cd rave
docker compose up -d --build
```

Open http://localhost:8000, search for songs (or paste YouTube/Spotify links, or switch to **Upload**), hit **+** on each, and press **Mash it up**. Drag the lineup to reorder it, star the **base** song whose beat and key everyone follows, and override any song's key if you want. Finished mashups, downloaded songs and the stem cache are kept in the `rave-data` volume.

### Song sources

- **YouTube** search and links work out of the box. The image includes yt-dlp and the deno JavaScript runtime it needs for YouTube. YouTube changes often and breaks old yt-dlp versions, so the container upgrades yt-dlp each time it starts. That can add up to a minute to startup; set `RAVE_UPDATE_YTDLP=0` to skip it.
- **Spotify links** (`https://open.spotify.com/track/...`) work without any setup. **Spotify search by name** needs a free app from the [Spotify developer dashboard](https://developer.spotify.com/dashboard). Put its keys in a `.env` file next to `docker-compose.yml`:
  ```
  SPOTIFY_CLIENT_ID=...
  SPOTIFY_CLIENT_SECRET=...
  ```
- **"Sign in to confirm you're not a bot."** YouTube sometimes says this to servers, especially on cloud/VPS IPs; a home connection is rarely affected. If it happens, export your YouTube cookies in Netscape format with a "cookies.txt" browser extension while logged in to youtube.com, then copy the file into the volume:
  ```sh
  docker compose cp cookies.txt rave:/data/cookies.txt
  ```
  The `YTDLP_COOKIES` variable points at that path. Use a throwaway Google account if you can.
- **Upload** takes mp3, wav, flac, m4a, ogg and opus files.

Downloads are capped at 15 minutes per song and cached by video id, so re-using a song costs nothing.

Without Compose:

```sh
docker build -t rave .
docker run -d -p 8000:8000 -v rave-data:/data --name rave rave
```

### Build options

| Build arg | Default | |
|---|---|---|
| `WITH_DEMUCS` | `1` | `0` leaves out PyTorch and Demucs (an image of about 600 MB instead of about 2 GB). It then uses a rough built-in separator instead. |
| `TORCH_INDEX` | CPU wheels | `https://download.pytorch.org/whl/cu124` for an NVIDIA GPU. Run the container with `--gpus all`, or uncomment the `deploy` block in `docker-compose.yml`. |

The Demucs model is downloaded at build time and baked into the image, so the container needs no internet at runtime.

### Environment

| Variable | Default | |
|---|---|---|
| `RAVE_DATA` | `/data` | Jobs and the stem cache. |
| `RAVE_CACHE` | `$RAVE_DATA/cache` | Separated stems, keyed by a hash of the file, so re-using a song skips separation. Safe to delete. |
| `RAVE_WORKERS` | `1` | Mashups made in parallel. Demucs is memory hungry; keep this at 1 unless you have the RAM. |
| `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET` | unset | Turns on Spotify search by name. |
| `YTDLP_COOKIES` | `/data/cookies.txt` | YouTube cookies, used if the file exists. |
| `RAVE_UPDATE_YTDLP` | `1` | Upgrades yt-dlp when the container starts. |

### Speed

On CPU, Demucs takes about as long as the song itself to separate it. Each new 3-4 minute song therefore costs a few minutes, and a song already in the cache costs nothing. A GPU makes separation near-instant.

## Command line

The same engine runs headless inside the container:

```sh
docker run --rm -v "$PWD:/work" -w /work rave \
  python -m rave "song1.mp3" "song2.mp3" "song3.mp3" -o mashup.mp3
```

| Flag | |
|---|---|
| `--base N` | Song number (1-based) whose beat and key everyone follows. The default picks the song that needs the least stretching. |
| `--length SECONDS` | Target length. The default is 150. |
| `--phrase 4\|8\|16` | Bars per vocal turn. |
| `--bed base\|rotate` | Keep one backing throughout, or rotate the backings too. |
| `--transpose N:SEMITONES` | Override the key matching for a song, e.g. `--transpose 2:-3`. |
| `--no-key-match`, `--no-drop`, `--no-riser` | Turn those features off. |
| `--separation auto\|demucs\|fast` | Pick the separator. |

## API

| | |
|---|---|
| `GET /api/search?q=...&source=youtube\|spotify` | Search results, or the one track a pasted YouTube/Spotify link points to. |
| `POST /api/jobs` | Multipart. `files` takes any uploads. `options` takes a JSON object whose `tracks` is the lineup in order: each entry is a search result, or `{"upload": n}` for the n-th uploaded file. It also takes `title`, `base`, `length`, `phrase_bars`, `bed`, `separation`, `format`, `match_key`, `transpose` (`{"index": semitones}`), `acappella_drop` and `riser`. |
| `GET /api/jobs`, `GET /api/jobs/{id}` | Status, progress and the cue sheet. |
| `GET /api/jobs/{id}/audio[?download=true]` | The mashup. |
| `DELETE /api/jobs/{id}` | Deletes a finished job. |

## Tips for good mashups

- Songs within about 15% of each other's tempo, or about half or double it, sound best. The cue sheet shows how far each song was stretched.
- Transposing vocals by 5-6 semitones is audible. If the cue sheet shows a big shift, make a different song the base, or force a smaller shift with the per-song **key** selector. For example, -5 can usually become +7 or +2, depending on the singer's range.
- Pop songs with clear 8-bar phrases and big choruses work best. The finale is built from whatever each song's chorus detector picked, and the cue sheet says which bar that was.

## Development

```sh
pip install -r requirements.txt pytest            # plus: pip install torch demucs
python tests/make_test_songs.py test_songs        # synthetic songs at different tempos/keys
python -m rave test_songs/*.wav -o out.wav
uvicorn rave.web:app --reload             # web UI on :8000
pytest tests
```

The code is laid out like this:

| File | What it does |
|---|---|
| `rave/analysis.py` | Beats, downbeats, key, tempo folding and key-shift maths. |
| `rave/separate.py` | Demucs and the fallback separator, with the stem cache. |
| `rave/transform.py` | Rubber Band, or the librosa fallback. |
| `rave/engine.py` | Conforming, phrase and chorus detection, the arranger and the renderer. |
| `rave/mix.py` | DSP helpers. |
| `rave/sources.py` | YouTube and Spotify search, link parsing, downloads and Spotify-to-YouTube matching. |
| `rave/web.py` and `static/index.html` | The web app. |

Only use music you have the rights to, and note that downloading from YouTube may be against its terms of service. This is a personal, self-hosted tool, so how you use it is up to you.
