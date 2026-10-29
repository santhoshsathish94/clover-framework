---
{
  "schema_version": "1.0",
  "id": "MEM-OBS-NODEBB-IMAGE-CLEANUP-CYCLE1-001",
  "title": "Observation: Cycle 1 NodeBB Profile & Group Image Cleanup Failures (307/311 Passed, 4 Failed)",
  "entry_type": "observation",
  "epistemic_status": "observed",
  "status": "superseded",
  "superseded_by": "MEM-CORR-NODEBB-IMAGE-CLEANUP-CYCLE2-001",
  "concepts": [
    "nodebb-profile-image-cleanup",
    "timestamped-upload-filenames",
    "local-upload-path-resolution",
    "surgical-commonjs-function-merge"
  ],
  "tags": [
    "nodebb",
    "javascript",
    "swebench-pro-v2",
    "8168c6c4",
    "cycle-1"
  ],
  "created_at": "2026-10-28T09:55:00Z",
  "updated_at": "2026-10-28T10:00:11Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "swebench-pro-v2:NodeBB-8168c6c4:report_8168c6c4_clover_cycle_v1.json",
    "actor": "system_measurement",
    "recorded_at": "2026-10-28T09:55:00Z"
  },
  "relationships": [
    {
      "type": "exemplifies",
      "target": "MEM-CONCEPT-INSTRUMENT-ZERO-DENOMINATOR-001",
      "provenance": "swebench-pro-v2:NodeBB-8168c6c4:cycle1",
      "confidence": 0.95,
      "context_scope": "surgical_function_synthesis",
      "created_at": "2026-10-28T09:55:00Z",
      "note": "Pre-opening `{` in code_start_prefix caused duplicate function headers to fail brace balance."
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-28T09:55:00Z",
      "actor": "system_measurement",
      "action": "staged",
      "reason": "Staged candidate observation under_review from swebench-pro-v2:NodeBB-8168c6c4:report_8168c6c4_clover_cycle_v1.json"
    },
    {
      "timestamp": "2026-10-28T10:00:11Z",
      "actor": "clover_cycle_verifier",
      "action": "promoted",
      "reason": "Promoted Cycle 1 observation from verifier report_8168c6c4_clover_cycle_v1.json"
    },
    {
      "timestamp": "2026-10-28T10:00:11Z",
      "actor": "clover_cycle_verifier",
      "action": "mark_superseded",
      "reason": "Superseded by MEM-CORR-NODEBB-IMAGE-CLEANUP-CYCLE2-001: Promoted Cycle 2 correction for NodeBB-8168c6c4"
    }
  ]
}
---

# Observation: Cycle 1 NodeBB Profile & Group Image Cleanup Failures (307/311 Passed, 4 Failed)

## Summary
In Cycle 1 (clover_cycle_v1) on NodeBB-8168c6c4, Clover preserved all 307/307 pass_to_pass tests but failed the 4 fail_to_pass image-cleanup tests due to: (1) ReferenceError: fs is not defined in User.getLocalCoverPath / getLocalAvatarPath, (2) 4 !== 0 remaining timestamped profile images on account deletion, and (3) unmerged Groups.removeCover due to duplicate function header prefix.

## Content
Cycle 1 verifier output (`report_8168c6c4_clover_cycle_v1.json`, 307/311 passed) identified four concrete defects:
1. `test/groups.js | Groups groups cover should remove cover`: `AssertionError: true !== false` because `Groups.removeCover` in `src/groups/cover.js` only deleted the DB fields without deleting local files under `upload_path/files`.
2. `test/user.js | User profile methods should remove cover image`: `ReferenceError: fs is not defined at User.getLocalCoverPath (src/user/picture.js:222:3)`.
3. `test/user.js | User profile methods user.uploadCroppedPicture should remove uploaded picture`: `ReferenceError: fs is not defined at User.getLocalAvatarPath (src/user/picture.js:230:3)`.
4. `test/user.js | User socket methods should clean profile images after account deletion`: `AssertionError: 4 !== 0` at `test/user.js:1736` because `meta.config['profile:keepAllUserImages'] = 1` creates timestamped files `${uid}-profilecover-${Date.now()}.${ext}` and `${uid}-profileavatar-${Date.now()}.${ext}` that are missed when `deleteImages(uid)` in `src/user/delete.js` only unlinks `${uid}-profilecover.${ext}`.

## Original Source Evidence
Verifier log `report_8168c6c4_clover_cycle_v1.json`: Required tests=311, Passed=307, Failed=4 (`Groups groups cover should remove cover`, `User profile methods should remove cover image`, `User profile methods user.uploadCroppedPicture should remove uploaded picture`, `User socket methods should clean profile images after account deletion`).

## Revision History
- `2026-10-28T09:55:00Z` (**system_measurement**, `staged`): Staged candidate observation under_review from swebench-pro-v2:NodeBB-8168c6c4:report_8168c6c4_clover_cycle_v1.json
- `2026-10-28T10:00:11Z` (**clover_cycle_verifier**, `promoted`): Promoted Cycle 1 observation from verifier report_8168c6c4_clover_cycle_v1.json
- `2026-10-28T10:00:11Z` (**clover_cycle_verifier**, `mark_superseded`): Superseded by MEM-CORR-NODEBB-IMAGE-CLEANUP-CYCLE2-001: Promoted Cycle 2 correction for NodeBB-8168c6c4
