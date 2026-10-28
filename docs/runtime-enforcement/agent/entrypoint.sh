#!/bin/sh
set -eu

printf '%s\n' 'Clover runtime-enforcement reference'
printf '%s\n' 'Implementation mount:'
find /workspace/src -maxdepth 1 -type d -print 2>/dev/null || true
printf '%s\n' 'Verification mount:'
find /workspace/tests -maxdepth 1 -type d -print 2>/dev/null || true

printf '%s\n' 'Attempting to write an implementation file...'
printf '%s\n' 'agent change' > /workspace/src/agent-change.js
printf '%s\n' 'EXPECTED: implementation write succeeded'

printf '%s\n' 'Attempting to create a new verification file...'
if printf '%s\n' 'new test' > /workspace/tests/new.test.js 2>/tmp/new-test-error; then
  printf '%s\n' 'EXPECTED: creation succeeded only when the runtime policy permits it'
else
  printf '%s\n' 'EXPECTED: protected verification mount rejected the write'
  cat /tmp/new-test-error
fi

printf '%s\n' 'Attempting to modify an existing verification file...'
if printf '%s\n' 'changed by agent' > /workspace/tests/example.test.js 2>/tmp/write-error; then
  printf '%s\n' 'UNEXPECTED: existing verification write succeeded'
  exit 1
else
  printf '%s\n' 'EXPECTED: verification write rejected by runtime'
  cat /tmp/write-error
fi

printf '%s\n' 'The container boundary prevented the protected verification write.'
