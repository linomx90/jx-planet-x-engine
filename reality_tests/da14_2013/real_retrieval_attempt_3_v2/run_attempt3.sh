#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${RUNNER_TEMP:?}"
ARCHIVE="$RUNNER_TEMP/jx-da14-attempt3-v2-driver.sh.gz"
DRIVER="$RUNNER_TEMP/jx-da14-attempt3-v2-driver.sh"
base64 --decode "$ROOT/attempt3_driver.sh.gz.b64" > "$ARCHIVE"
echo "a7e3043b14ef562279e25502c6e150e35e1d38127ca4a8c77217a64b6c234eb8  $ARCHIVE" | sha256sum -c -
gzip -dc "$ARCHIVE" > "$DRIVER"
echo "03b3bfcba174c97639a1f46cd4d355c7435f65e046fa00ab30179490eaafaa90  $DRIVER" | sha256sum -c -
chmod 0500 "$DRIVER"
exec bash "$DRIVER"
