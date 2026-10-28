---
{
  "schema_version": "1.0",
  "id": "MEM-CORR-NODEBB-IMAGE-CLEANUP-CYCLE2-001",
  "title": "Correction: Centralized Local Cover & Avatar Path Resolution and Glob Profile Cleanup in NodeBB",
  "entry_type": "correction",
  "epistemic_status": "externally_verified",
  "status": "active",
  "concepts": [
    "nodebb-profile-image-cleanup",
    "timestamped-upload-filenames",
    "local-upload-path-resolution",
    "statement-boundary-soundness"
  ],
  "tags": [
    "nodebb",
    "javascript",
    "swebench-pro-v2",
    "8168c6c4",
    "cycle-2",
    "correction"
  ],
  "created_at": "2026-10-28T10:00:00Z",
  "updated_at": "2026-10-28T10:00:11Z",
  "provenance": {
    "source_type": "experiment",
    "source_ref": "swebench-pro-v2:NodeBB-8168c6c4:clover_cycle_v2",
    "actor": "human_and_assistant",
    "recorded_at": "2026-10-28T10:00:00Z"
  },
  "relationships": [
    {
      "type": "corrects",
      "target": "MEM-OBS-NODEBB-IMAGE-CLEANUP-CYCLE1-001",
      "provenance": "swebench-pro-v2:NodeBB-8168c6c4:clover_cycle_v2",
      "confidence": 1.0,
      "context_scope": "nodebb_image_cleanup",
      "created_at": "2026-10-28T10:00:00Z",
      "note": "Fixes all 4 Mocha test failures observed in Cycle 1."
    },
    {
      "type": "derived_from",
      "target": "MEM-OBS-NODEBB-IMAGE-CLEANUP-CYCLE1-001",
      "provenance": "swebench-pro-v2:NodeBB-8168c6c4:clover_cycle_v1",
      "confidence": 1.0,
      "context_scope": "nodebb_image_cleanup",
      "created_at": "2026-10-28T10:00:00Z",
      "note": "Derived directly from Cycle 1 Outcome stack traces."
    },
    {
      "type": "exemplifies",
      "target": "MEM-CONCEPT-PREFIX-SHADOWING-DISPATCH-001",
      "provenance": "swebench-pro-v2:NodeBB-8168c6c4:clover_cycle_v2",
      "confidence": 0.9,
      "context_scope": "auto_linked_concept_bridge",
      "created_at": "2026-10-28T10:00:00Z",
      "note": "Shared concept bridge (statement-boundary-soundness)"
    }
  ],
  "revisions": [
    {
      "timestamp": "2026-10-28T10:00:00Z",
      "actor": "human_and_assistant",
      "action": "staged",
      "reason": "Staged candidate correction under_review from swebench-pro-v2:NodeBB-8168c6c4:clover_cycle_v2"
    },
    {
      "timestamp": "2026-10-28T10:00:11Z",
      "actor": "clover_cycle_verifier",
      "action": "promoted",
      "reason": "Promoted Cycle 2 correction for NodeBB-8168c6c4"
    }
  ]
}
---

# Correction: Centralized Local Cover & Avatar Path Resolution and Glob Profile Cleanup in NodeBB

## Summary
Resolves all 4 Cycle 1 failures on NodeBB-8168c6c4: (1) Groups.removeCover deletes local files under upload_path/files, (2) User.getLocalCoverPath and User.getLocalAvatarPath resolve paths from User.getUserField via nconf.get('relative_path') and nconf.get('upload_path'), (3) User.removeCoverPicture and User.removeProfileImage centralize disk + DB cleanup, and (4) deleteImages(uid) uses rimrafAsync on `${uid}-profile*` to remove all timestamped profile images on account deletion.

## Content
Corrects `MEM-OBS-NODEBB-IMAGE-CLEANUP-CYCLE1-001` across the NodeBB image cleanup modules (`src/groups/cover.js`, `src/user/picture.js`, `src/socket.io/user/picture.js`, `src/socket.io/user/profile.js`, `src/user/delete.js`):

```javascript
// File: src/groups/cover.js
const nconf = require('nconf');

	Groups.removeCover = async function (data) {
		const fields = ['cover:url', 'cover:thumb:url'];
		const values = await Groups.getGroupFields(data.groupName, fields);
		await Promise.all(fields.map((field) => {
			if (!values[field] || !values[field].startsWith(`${nconf.get('relative_path')}/assets/uploads/files/`)) {
				return;
			}
			const filename = values[field].split('/').pop();
			const filePath = path.join(nconf.get('upload_path'), 'files', filename);
			return file.delete(filePath);
		}));
		await db.deleteObjectFields(`group:${data.groupName}`, ['cover:url', 'cover:thumb:url', 'cover:position']);
	};
```

```javascript
// File: src/socket.io/user/picture.js
	SocketUser.removeUploadedPicture = async function (socket, data) {
		if (!socket.uid || !data || !data.uid) {
			throw new Error('[[error:invalid-data]]');
		}
		await user.isAdminOrSelf(socket.uid, data.uid);
		const userData = await user.removeProfileImage(data.uid);
		plugins.hooks.fire('action:user.removeUploadedPicture', {
			callerUid: socket.uid,
			uid: data.uid,
			user: userData,
		});
	};
```

```javascript
// File: src/socket.io/user/profile.js
	SocketUser.removeCover = async function (socket, data) {
		if (!socket.uid) {
			throw new Error('[[error:no-privileges]]');
		}
		await user.isAdminOrGlobalModOrSelf(socket.uid, data.uid);
		const userData = await user.getUserFields(data.uid, ['cover:url']);
		await user.removeCoverPicture(data);
		plugins.hooks.fire('action:user.removeCoverPicture', {
			callerUid: socket.uid,
			uid: data.uid,
			user: userData,
		});
	};
```

```javascript
// File: src/user/delete.js
const util = require('util');
const rimrafAsync = util.promisify(require('rimraf'));

	async function deleteImages(uid) {
		const folder = path.join(nconf.get('upload_path'), 'profile');
		await rimrafAsync(path.join(folder, `${uid}-profile*`));
	}
```

```javascript
// File: src/user/picture.js
	async function deleteCurrentPicture(uid, field) {
		if (meta.config['profile:keepAllUserImages']) {
			return;
		}
		await deletePicture(uid, field);
	}

	async function deletePicture(uid, field) {
		const uploadPath = await getPicturePath(uid, field);
		if (uploadPath) {
			await file.delete(uploadPath);
		}
	}

	async function getPicturePath(uid, field) {
		const value = await User.getUserField(uid, field);
		if (!value || !value.startsWith(`${nconf.get('relative_path')}/assets/uploads/profile/`)) {
			return false;
		}
		const filename = value.split('/').pop();
		return path.join(nconf.get('upload_path'), 'profile', filename);
	}

	User.getLocalCoverPath = async function (uid) {
		return getPicturePath(uid, 'cover:url');
	};

	User.getLocalAvatarPath = async function (uid) {
		return getPicturePath(uid, 'uploadedpicture');
	};

	User.removeCoverPicture = async function (data) {
		await deletePicture(data.uid, 'cover:url');
		await db.deleteObjectFields(`user:${data.uid}`, ['cover:url', 'cover:position']);
	};

	User.removeProfileImage = async function (uid) {
		const userData = await User.getUserFields(uid, ['uploadedpicture', 'picture']);
		await deletePicture(uid, 'uploadedpicture');
		await User.setUserFields(uid, {
			uploadedpicture: '',
			picture: userData.uploadedpicture === userData.picture ? '' : userData.picture,
		});
		return userData;
	};
```

## Original Source Evidence
Diagnosed from Cycle 1 verifier stack traces (`test/groups.js:1543`, `test/user.js:1044`, `test/user.js:1247`, `test/user.js:1736`) and verified against NodeBB `src/file.js`, `src/groups/cover.js`, `src/user/picture.js`, and `src/user/delete.js`.

## Revision History
- `2026-10-28T10:00:00Z` (**human_and_assistant**, `staged`): Staged candidate correction under_review from swebench-pro-v2:NodeBB-8168c6c4:clover_cycle_v2
- `2026-10-28T10:00:11Z` (**clover_cycle_verifier**, `promoted`): Promoted Cycle 2 correction for NodeBB-8168c6c4
