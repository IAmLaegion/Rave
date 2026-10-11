#!/bin/sh
# Build two test stem packs from cached Demucs stems: Fire as a 4-file multitrack (zip), Runaway as an
# a cappella + instrumental pair (7z), so the instrumental has to be split by Rave.
set -e
cd /app
K1=$(python -c "from rave import separate;print(separate.file_key('/data/cache/youtube/a-UyKC9ApI4.webm'))")
K2=$(python -c "from rave import separate;print(separate.file_key('/data/cache/youtube/-ehVlh25p4A.webm'))")
rm -rf /tmp/p && mkdir -p /tmp/p/fire /tmp/p/run
f() { ffmpeg -loglevel error -y -i "$1" -c:a flac "$2"; }
f /data/cache/$K1.demucs.vocals.flac "/tmp/p/fire/Lead Vox.flac"
f /data/cache/$K1.demucs.drums.flac "/tmp/p/fire/Drum Kit.flac"
f /data/cache/$K1.demucs.bass.flac "/tmp/p/fire/Sub Bass.flac"
f /data/cache/$K1.demucs.other.flac "/tmp/p/fire/Synths.flac"
f /data/cache/$K2.demucs.vocals.flac "/tmp/p/run/Runaway Acapella.flac"
ffmpeg -loglevel error -y -i /data/cache/$K2.demucs.drums.flac -i /data/cache/$K2.demucs.bass.flac \
  -i /data/cache/$K2.demucs.other.flac -filter_complex amix=inputs=3:normalize=0 "/tmp/p/run/Runaway Instrumental.flac"
cd /tmp/p/fire && 7z a -bd -tzip ../fire-stems.zip . >/dev/null
cd /tmp/p/run && 7z a -bd -t7z ../runaway-stems.7z . >/dev/null
ls -la /tmp/p/*.zip /tmp/p/*.7z
