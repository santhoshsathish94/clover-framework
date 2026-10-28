# Clover Runtime Enforcement Reference

This reference demonstrates how a Clover verification boundary can be enforced outside the model.

**Two layers, and only one of them is a boundary.** The Python policy is a gate an agent's tooling
calls; it protects nothing from code that does not call it, because a plain `open()` goes straight
past it. The container mount is the boundary: it refuses the write whether or not anything asked. The
policy exists to give a clear denial and an audit record at the point of decision, and the mount
exists because the policy can be bypassed. Do not deploy the first without the second.

## What it demonstrates

- Existing trusted verification files are treated as immutable by the write policy.
- New verification files require explicit creation permission.
- Parent traversal, symbolic links and hard links are all resolved or identity-matched before authorization.
- Denied write attempts are emitted as structured JSON security events without recording file contents.
- Docker provides the physical filesystem boundary by mounting verification artifacts read-only, on a
  read-only root filesystem with all capabilities dropped and no route to the network.

## Run the policy tests

From the repository root:

```bash
python -m pytest reference/runtime-enforcement/tests/test_write_guard.py
```

## Run the Docker boundary example

Requires Docker and Docker Compose.

**Not observed running.** The compose file and the container settings below were written without a
Docker daemon available, so `read_only`, `cap_drop`, `no-new-privileges`, the resource limits and the
internal network have never been exercised. The Python policy tests have. Treat the container half of
this reference as a pattern to check on your own machine rather than a verified result, and tell us
if it does not behave as described.

```bash
cd reference/runtime-enforcement
./verify.sh
```

The container can write under `workspace/src` but cannot modify the existing verification artifact under `workspace/tests` because that mount is read-only.

## New verification files

The Python policy supports explicit creation mode:

```bash
CLOVER_ALLOW_NEW_TESTS=1
```

Creation permission does not grant permission to modify an existing trusted verification artifact. That distinction is intentional.

## Production use

This is a reference pattern, not a security certification or a complete agent sandbox. Real systems should combine the enforcement point appropriate to their environment — for example filesystem permissions, container isolation, CI identities, protected branches, or a tool/MCP gateway — with repository-specific definitions of trusted verification state.

The key Clover rule is:

> **When a boundary matters to Outcome, enforce it outside the model wherever the environment allows.**
