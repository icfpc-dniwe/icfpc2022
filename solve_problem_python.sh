#!/usr/bin/env bash

set -eE

usage() {
  echo "Usage: $0 problem_id" >&2
  exit 1
}

problem_id="$1" && shift || usage
shift && usage || true

api_token=$(cat api_token)
img_path="problems/$problem_id.png"

if [ ! -e "$img_path" ]; then
  curl -O "$img_path" -L "https://cdn.robovinci.xyz/imageframes/$problem_id.png"
fi

tmpdir=$(mktemp -d)
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup INT TERM EXIT

out_path="$tmpdir/solution.txt"

cd src && python -m python -p "$problem_id" -o "$out_path" -r main

curl -v -H "Authorization: Bearer $api_token" -F "file=@$out_path" "https://robovinci.xyz/api/submissions/$problem_id/create"
