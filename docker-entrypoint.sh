#!/bin/sh
# YouTube changes often and old yt-dlp versions stop working, so fetch the latest one at start
# (into the user site, which wins over the baked-in copy). Set RAVE_UPDATE_YTDLP=0 to skip.
if [ "${RAVE_UPDATE_YTDLP:-1}" = "1" ]; then
  timeout 90 pip install --user --quiet --disable-pip-version-check --upgrade "yt-dlp[default]" \
    || echo "Couldn't update yt-dlp, using the bundled version." >&2
fi
exec "$@"
