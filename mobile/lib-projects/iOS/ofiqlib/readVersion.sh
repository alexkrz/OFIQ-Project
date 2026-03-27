#!/bin/bash

# Read the version from the version.txt
VERSION_FILE=../../../../Version.txt
OUTPUT_HEADER="Version.h"
while IFS=' ' read -r key value; do
  case "$key" in
    VERSION_MAJOR) VERSION_MAJOR=$value ;;
    VERSION_MINOR) VERSION_MINOR=$value ;;
    VERSION_PATCH) VERSION_PATCH=$value ;;
  esac
done < "$VERSION_FILE"

cat > "$OUTPUT_HEADER" <<EOL
#ifndef Version_h
#define Version_h

#define OFIQ_VERSION_MAJOR $VERSION_MAJOR
#define OFIQ_VERSION_MINOR $VERSION_MINOR
#define OFIQ_VERSION_PATCH $VERSION_PATCH

#endif // Version_h
EOL
