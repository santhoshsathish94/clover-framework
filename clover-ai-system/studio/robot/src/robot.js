import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { mergeVertices } from 'three/addons/utils/BufferGeometryUtils.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';

const SEED = 20261106;
function mulberry32(a) {
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(SEED);
const TAU = Math.PI * 2;
const DURATION = 12;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const motion = reducedMotion ? 0 : 1;
const COARSE = matchMedia('(pointer: coarse)').matches;  // phones: lighter pixel ratio and shadow map

// Transparent canvas: the host page shows through around the character (Clover AI Studio's voice mode).
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(devicePixelRatio, COARSE ? 1.5 : 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.domElement.setAttribute('aria-hidden', 'true');
renderer.domElement.style.display = 'block';

const scene = new THREE.Scene();

const view = { width: 1, height: 1 };
const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 140);
const cameraTarget = new THREE.Vector3(0, 2.3, 0);
// Fit feet to antenna (and the arms, cloud and props beside the body) into any container shape.
const FIT_HEIGHT = 5.5;
const FIT_WIDTH = 3.6;
const FIT_CENTER_Y = 2.5;
function frameCamera() {
  const aspect = view.width / view.height;
  camera.aspect = aspect;
  const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const distance = Math.max(FIT_HEIGHT / 2 / tanHalf, FIT_WIDTH / 2 / (tanHalf * aspect)) * 1.06;
  cameraTarget.set(0, FIT_CENTER_Y, 0);
  camera.position.set(-0.1 * distance, FIT_CENTER_Y + 0.7, distance);
  camera.lookAt(cameraTarget);
  camera.updateProjectionMatrix();
}
frameCamera();

const pmrem = new THREE.PMREMGenerator(renderer);
const room = new RoomEnvironment();
scene.environment = pmrem.fromScene(room, 0.04).texture;
room.dispose();
pmrem.dispose();

scene.add(new THREE.HemisphereLight(0xf4fbff, 0x71877a, 1.05));
const key = new THREE.DirectionalLight(0xfff3df, 2.35);
key.position.set(-5, 9, 7);
key.castShadow = true;
key.shadow.mapSize.set(COARSE ? 1024 : 2048, COARSE ? 1024 : 2048);
key.shadow.camera.left = -6;
key.shadow.camera.right = 6;
key.shadow.camera.top = 7;
key.shadow.camera.bottom = -4;
key.shadow.camera.near = 0.5;
key.shadow.camera.far = 28;
key.shadow.bias = -0.00025;
key.shadow.normalBias = 0.025;
key.shadow.radius = 5;
scene.add(key);

const rim = new THREE.DirectionalLight(0xc8f3df, 1.55);
rim.position.set(5, 5, -4);
scene.add(rim);
const faceLight = new THREE.DirectionalLight(0xffffff, 0.72);
faceLight.position.set(1, 3, 8);
scene.add(faceLight);

// Shadow-only floor: nothing visible is drawn, so the robot stands on whatever the host page shows.
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(6, 64).rotateX(-Math.PI / 2),
  new THREE.ShadowMaterial({ opacity: 0.16 })
);
ground.receiveShadow = true;
scene.add(ground);

function makeContactShadow() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 256;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(128, 128, 8, 128, 128, 126);
  gradient.addColorStop(0, 'rgba(27, 60, 55, 0.4)');
  gradient.addColorStop(0.45, 'rgba(27, 60, 55, 0.17)');
  gradient.addColorStop(1, 'rgba(27, 60, 55, 0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 256, 256);
  const material = new THREE.MeshBasicMaterial({
    map: new THREE.CanvasTexture(canvas), transparent: true,
    depthWrite: false, opacity: 0.72, toneMapped: false
  });
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(5.2, 4.2), material);
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.set(0, 0.018, 0.3);
  return shadow;
}
const contactShadow = makeContactShadow();
scene.add(contactShadow);

const cream = new THREE.MeshPhysicalMaterial({
  color: 0xd8bd9e, roughness: 0.3, metalness: 0.03,
  clearcoat: 0.62, clearcoatRoughness: 0.24,
  envMapIntensity: 0.68, specularIntensity: 0.52
});
const creamLight = cream.clone();
creamLight.color.set(0xe4cdae);
const darkJoint = new THREE.MeshStandardMaterial({ color: 0x202722, roughness: 0.38, metalness: 0.2 });
const handMaterial = new THREE.MeshPhysicalMaterial({
  color: 0x3b403a, roughness: 0.34, metalness: 0.2, clearcoat: 0.3
});
const fingertipMaterial = new THREE.MeshStandardMaterial({ color: 0x454a42, roughness: 0.4, metalness: 0.14 });
const oliveMaterial = new THREE.MeshPhysicalMaterial({
  color: 0x4b641c, roughness: 0.3, metalness: 0.12, clearcoat: 0.55
});
const faceMaterial = new THREE.MeshPhysicalMaterial({
  color: 0x050b08, roughness: 0.065, metalness: 0.18,
  clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 1.15
});
const greenGlow = new THREE.MeshStandardMaterial({
  color: 0xf4ffd4, emissive: 0xaaff22, emissiveIntensity: 3.4,
  roughness: 0.24, toneMapped: false
});
const eyeGlowMaterial = new THREE.MeshBasicMaterial({
  color: 0x9cff28, transparent: true, opacity: 0.22,
  blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
});

const character = new THREE.Group();
scene.add(character);

function shadowed(mesh, receive = true) {
  mesh.castShadow = true;
  mesh.receiveShadow = receive;
  return mesh;
}
function ellipsoid(scale, material, position = [0, 0, 0], segments = 48) {
  const mesh = shadowed(new THREE.Mesh(new THREE.SphereGeometry(1, segments, Math.round(segments * 0.65)), material));
  mesh.scale.set(...scale);
  mesh.position.set(...position);
  return mesh;
}
function capsuleBetween(start, end, radius, material, radialSegments = 24) {
  const a = new THREE.Vector3(...start);
  const b = new THREE.Vector3(...end);
  const direction = b.clone().sub(a);
  const length = direction.length();
  const geometry = new THREE.CapsuleGeometry(radius, Math.max(0.01, length - 2 * radius), 8, radialSegments);
  const mesh = shadowed(new THREE.Mesh(geometry, material));
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  return mesh;
}
function tubeFrom(points, radius, material, segments = 32) {
  const vectors = points.map((p) => new THREE.Vector3(...p));
  const curve = vectors.length === 4
    ? new THREE.CubicBezierCurve3(...vectors)
    : new THREE.CatmullRomCurve3(vectors);
  return shadowed(new THREE.Mesh(new THREE.TubeGeometry(curve, segments, radius, 10, false), material), false);
}

function leafBend(y) {
  const normalized = THREE.MathUtils.clamp(y / 1.1, 0, 1);
  return Math.sin(normalized * Math.PI) * 0.07;
}
function makeHeartLeafGeometry() {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.bezierCurveTo(-0.1, 0.24, -0.61, 0.29, -0.61, 0.67);
  shape.bezierCurveTo(-0.61, 1.04, -0.13, 1.12, 0, 0.79);
  shape.bezierCurveTo(0.13, 1.12, 0.61, 1.04, 0.61, 0.67);
  shape.bezierCurveTo(0.61, 0.29, 0.1, 0.24, 0, 0);
  const geometry = new THREE.ExtrudeGeometry(shape, {
    steps: 1, depth: 0.08, bevelEnabled: true,
    bevelThickness: 0.035, bevelSize: 0.035, bevelSegments: 4, curveSegments: 28
  });
  const positions = geometry.attributes.position;
  for (let i = 0; i < positions.count; i++) {
    positions.setZ(i, positions.getZ(i) + leafBend(positions.getY(i)));
  }
  positions.needsUpdate = true;
  geometry.deleteAttribute('normal');
  geometry.deleteAttribute('uv');
  const smooth = mergeVertices(geometry, 0.0001);
  smooth.computeVertexNormals();
  geometry.dispose();
  return smooth;
}
const cloverLeafGeometry = makeHeartLeafGeometry();
const cloverMaterials = [0x527a13, 0x426b0d, 0x5f8c1b, 0x4a7410].map((color) => new THREE.MeshPhysicalMaterial({
  color, roughness: 0.4, clearcoat: 0.2, clearcoatRoughness: 0.36,
  envMapIntensity: 0.38, specularIntensity: 0.4
}));
const cloverCenterMaterial = new THREE.MeshStandardMaterial({ color: 0x5d8618, roughness: 0.42 });
function makeClover(scale = 1) {
  const clover = new THREE.Group();
  for (let i = 0; i < 5; i++) {
    const leaf = shadowed(new THREE.Mesh(cloverLeafGeometry, cloverMaterials[i % cloverMaterials.length]), false);
    leaf.rotation.z = i * TAU / 5;
    clover.add(leaf);
  }
  const stem = tubeFrom([[0, 0.04, -0.01], [0.02, -0.3, -0.01], [0.08, -0.64, -0.01]], 0.045, cloverCenterMaterial, 24);
  clover.add(stem);
  const center = shadowed(new THREE.Mesh(new THREE.SphereGeometry(0.1, 20, 14), cloverCenterMaterial), false);
  center.position.z = 0.1;
  clover.add(center);
  clover.scale.setScalar(scale);
  return clover;
}

// Woven cloth texture shared by the wrapped scarf and the superhero cape.
function makeFabricTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#808080';
  ctx.fillRect(0, 0, 256, 256);
  ctx.lineWidth = 1;
  for (let i = 0; i < 256; i += 4) {
    ctx.strokeStyle = i % 8 === 0 ? '#929292' : '#747474';
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 256); ctx.stroke();
    ctx.strokeStyle = i % 8 === 0 ? '#898989' : '#787878';
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(256, i); ctx.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(5, 7);
  texture.anisotropy = Math.min(4, renderer.capabilities.getMaxAnisotropy());
  return texture;
}
const fabricTexture = makeFabricTexture();
const fabricMaterial = new THREE.MeshPhysicalMaterial({
  color: 0x405d22, roughness: 0.88, metalness: 0,
  clearcoat: 0, sheen: 1, sheenColor: 0x89a65b, sheenRoughness: 0.72,
  side: THREE.DoubleSide, envMapIntensity: 0.24,
  bumpMap: fabricTexture, bumpScale: 0.018
});

// A draped cloth grid with pleats at the shoulders and deeper folds toward the hem.
function capePoint(u, v) {
  const centered = u - 0.5;
  const width = 1.02 + v * 1.5;
  const edgeCurl = Math.pow(Math.abs(centered) * 2, 2) * 0.09 * v;
  const fold = Math.sin(u * TAU * 3.2 + v * 0.7) * (0.025 + v * 0.095);
  const x = centered * width + Math.sin(v * Math.PI) * Math.sin(u * TAU) * 0.025;
  const y = 2.53 - v * 1.55 + Math.sin(u * Math.PI) * 0.08 * v
    + Math.cos(u * TAU * 3) * 0.035 * v * v;
  const z = -0.29 - v * 0.39 - Math.sin(v * Math.PI) * 0.17 + fold + edgeCurl;
  return new THREE.Vector3(x, y, z);
}
function makeCape() {
  const columns = 48;
  const rows = 32;
  const positions = [];
  const uvs = [];
  const indices = [];
  for (let row = 0; row <= rows; row++) {
    const v = row / rows;
    for (let column = 0; column <= columns; column++) {
      const u = column / columns;
      const point = capePoint(u, v);
      positions.push(point.x, point.y, point.z);
      uvs.push(u, v);
    }
  }
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const a = row * (columns + 1) + column;
      const b = a + columns + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const capeMesh = shadowed(new THREE.Mesh(geometry, fabricMaterial));
  const hemPoints = Array.from({ length: 33 }, (_, i) => {
    const point = capePoint(i / 32, 1);
    return [point.x, point.y, point.z];
  });
  capeMesh.add(tubeFrom(hemPoints, 0.018, fabricMaterial, 96));
  return capeMesh;
}
const cape = makeCape();
const capeClover = makeClover(0.13);
capeClover.position.set(0.62, 1.35, -0.75);
capeClover.rotation.y = Math.PI;
cape.add(capeClover);
character.add(cape);

// Articulated standing legs: hips, knees, ankles and forward-facing boots.
function makeLeg(side) {
  const leg = new THREE.Group();
  leg.position.set(side * 0.38, 1.28, 0);
  leg.scale.set(1.02, 1.1, 1.02);
  leg.add(ellipsoid([0.24, 0.13, 0.24], darkJoint, [0, 0, 0], 32));
  const kneePosition = [side * 0.035, -0.48, 0];
  leg.add(capsuleBetween([0, -0.02, 0], kneePosition, 0.24, cream, 32));

  const lower = new THREE.Group();
  lower.position.set(...kneePosition);
  lower.add(ellipsoid([0.21, 0.1, 0.21], darkJoint, [0, 0, 0], 32));
  const anklePosition = [0, -0.45, 0.02];
  lower.add(capsuleBetween([0, -0.02, 0], anklePosition, 0.19, creamLight, 28));
  lower.add(ellipsoid([0.18, 0.08, 0.19], darkJoint, anklePosition, 28));

  const foot = new THREE.Group();
  foot.position.set(...anklePosition);
  const boot = ellipsoid([0.33, 0.18, 0.46], creamLight, [0, -0.04, 0.16], 48);
  boot.rotation.y = -side * 0.06;
  foot.add(boot);
  const sole = ellipsoid([0.31, 0.055, 0.4], handMaterial, [0, -0.115, 0.22], 36);
  sole.rotation.y = -side * 0.06;
  foot.add(sole);
  lower.add(foot);
  leg.add(lower);
  character.add(leg);
  return { leg, lower, foot, side };
}
const legs = [makeLeg(-1), makeLeg(1)];

// A child-sized rounded chair gives the sitting action a clear physical support.
function makeChair() {
  const chairGroup = new THREE.Group();
  const wood = new THREE.MeshPhysicalMaterial({
    color: 0x667645, roughness: 0.58, clearcoat: 0.18, clearcoatRoughness: 0.5
  });
  const cushion = new THREE.MeshPhysicalMaterial({
    color: 0xd8c7a4, roughness: 0.86, sheen: 0.45, sheenColor: 0xf2e2c4
  });
  const seat = shadowed(new THREE.Mesh(new RoundedBoxGeometry(1.26, 0.14, 0.9, 4, 0.07), wood));
  seat.position.set(0, 0.43, -0.12);
  chairGroup.add(seat);
  const pad = shadowed(new THREE.Mesh(new RoundedBoxGeometry(1.12, 0.08, 0.78, 4, 0.06), cushion));
  pad.position.set(0, 0.52, -0.1);
  chairGroup.add(pad);
  const back = shadowed(new THREE.Mesh(new RoundedBoxGeometry(1.42, 0.78, 0.12, 5, 0.08), wood));
  back.position.set(0, 1.02, -0.56);
  back.rotation.x = -0.06;
  chairGroup.add(back);
  for (const side of [-1, 1]) {
    chairGroup.add(capsuleBetween(
      [side * 0.58, 0.44, -0.5], [side * 0.58, 1.32, -0.54], 0.05, wood, 18
    ));
    const armrest = shadowed(new THREE.Mesh(new RoundedBoxGeometry(0.16, 0.12, 0.7, 4, 0.05), wood));
    armrest.position.set(side * 0.66, 0.84, -0.05);
    chairGroup.add(armrest);
    chairGroup.add(capsuleBetween(
      [side * 0.59, 0.43, 0.12], [side * 0.64, 0.79, 0.12], 0.045, wood, 16
    ));
    for (const z of [-0.43, 0.2]) {
      chairGroup.add(capsuleBetween(
        [side * 0.44, 0.42, z], [side * 0.5, 0.04, z + (z < 0 ? -0.04 : 0.04)],
        0.055, wood, 18
      ));
    }
  }
  chairGroup.visible = false;
  return chairGroup;
}
const chair = makeChair();
scene.add(chair);

// A gently bottom-weighted torso flows from the neck into the hips without a sharp bulge.
const torsoProfile = new THREE.SplineCurve([
  new THREE.Vector2(0.43, -0.78),
  new THREE.Vector2(0.48, -0.72),
  new THREE.Vector2(0.54, -0.58),
  new THREE.Vector2(0.59, -0.38),
  new THREE.Vector2(0.61, -0.16),
  new THREE.Vector2(0.6, 0.06),
  new THREE.Vector2(0.55, 0.28),
  new THREE.Vector2(0.45, 0.5),
  new THREE.Vector2(0.34, 0.7),
  new THREE.Vector2(0.28, 0.78)
]).getPoints(64);
const torso = shadowed(new THREE.Mesh(new THREE.LatheGeometry(torsoProfile, 64), cream));
torso.position.set(0, 1.8, 0);
torso.scale.z = 0.76;
character.add(torso);
character.add(ellipsoid([0.4, 0.1, 0.32], darkJoint, [0, 1.12, 0]));
const scarfMaterial = fabricMaterial.clone();
scarfMaterial.color.set(0x4b681f);
scarfMaterial.bumpScale = 0.012;
character.add(ellipsoid([0.27, 0.12, 0.23], darkJoint, [0, 2.48, 0], 32));
function makeScarfLayer(baseY, radius, phase) {
  const segments = 80;
  const rows = 6;
  const positions = [];
  const uvs = [];
  const indices = [];
  for (let row = 0; row <= rows; row++) {
    const cross = row / rows - 0.5;
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * TAU;
      const fold = Math.sin(theta * 3 + phase) * 0.014;
      const r = radius + cross * 0.14 + fold;
      positions.push(
        Math.cos(theta) * r,
        baseY + cross * 0.12 + Math.sin(theta * 2 + phase) * 0.014,
        Math.sin(theta) * r * 0.88
      );
      uvs.push(i / segments, row / rows);
    }
  }
  for (let row = 0; row < rows; row++) {
    for (let i = 0; i < segments; i++) {
      const a = row * (segments + 1) + i;
      const b = a + segments + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return shadowed(new THREE.Mesh(geometry, scarfMaterial));
}
character.add(makeScarfLayer(2.51, 0.37, 0.2));
character.add(makeScarfLayer(2.44, 0.39, 1.1));
const scarfKnot = ellipsoid([0.11, 0.065, 0.08], scarfMaterial, [0, 2.44, 0.41], 32);
scarfKnot.rotation.z = -0.18;
character.add(scarfKnot);
const cloverGlowCanvas = document.createElement('canvas');
cloverGlowCanvas.width = cloverGlowCanvas.height = 256;
const cloverGlowContext = cloverGlowCanvas.getContext('2d');
const cloverGlowGradient = cloverGlowContext.createRadialGradient(128, 128, 12, 128, 128, 126);
cloverGlowGradient.addColorStop(0, 'rgba(213, 242, 82, 0.34)');
cloverGlowGradient.addColorStop(0.45, 'rgba(199, 255, 65, 0.12)');
cloverGlowGradient.addColorStop(1, 'rgba(199, 255, 65, 0)');
cloverGlowContext.fillStyle = cloverGlowGradient;
cloverGlowContext.fillRect(0, 0, 256, 256);
const cloverGlowTexture = new THREE.CanvasTexture(cloverGlowCanvas);
cloverGlowTexture.colorSpace = THREE.SRGBColorSpace;
const chestGlow = new THREE.Mesh(new THREE.PlaneGeometry(1.05, 1.05), new THREE.MeshBasicMaterial({
  map: cloverGlowTexture, transparent: true, blending: THREE.AdditiveBlending,
  depthWrite: false, toneMapped: false
}));
chestGlow.position.set(0, 1.88, 0.535);
character.add(chestGlow);
const chestClover = makeClover(0.34);
chestClover.position.set(0, 1.93, 0.565);
chestClover.rotation.x = -0.06;
character.add(chestClover);

function makeHand(side) {
  const hand = new THREE.Group();
  hand.userData.fingers = [];
  hand.add(ellipsoid([0.17, 0.18, 0.105], creamLight, [0, -0.095, 0], 40));
  hand.add(ellipsoid([0.078, 0.115, 0.063], creamLight, [side * 0.072, -0.085, 0.045], 28));
  hand.add(ellipsoid([0.13, 0.135, 0.025], handMaterial, [0, -0.105, 0.096], 32));
  const fingerX = [-0.112, -0.037, 0.037, 0.112];
  const baseY = [-0.205, -0.22, -0.225, -0.205];
  const proximalLengths = [0.072, 0.09, 0.086, 0.064];
  const middleLengths = [0.053, 0.064, 0.061, 0.048];
  const tipLengths = [0.04, 0.047, 0.045, 0.037];
  for (let i = 0; i < 4; i++) {
    const finger = new THREE.Group();
    finger.position.set(fingerX[i], baseY[i], 0.016);
    finger.rotation.z = (i - 1.5) * 0.025;
    const radius = i === 3 ? 0.032 : 0.035;
    finger.add(ellipsoid([radius * 1.14, radius, radius * 1.08], darkJoint, [0, 0, 0], 16));
    finger.add(capsuleBetween([0, -0.005, 0], [0, -proximalLengths[i], 0.006], radius, handMaterial, 14));
    const middle = new THREE.Group();
    middle.position.set(0, -proximalLengths[i], 0.006);
    middle.add(ellipsoid([radius * 1.06, radius * 0.92, radius], darkJoint, [0, 0, 0], 16));
    middle.add(capsuleBetween([0, 0, 0], [0, -middleLengths[i], 0.01], radius * 0.92, handMaterial, 14));
    const tip = new THREE.Group();
    tip.position.set(0, -middleLengths[i], 0.01);
    tip.add(capsuleBetween([0, 0, 0], [side * 0.003, -tipLengths[i], 0.012], radius * 0.86, handMaterial, 14));
    tip.add(ellipsoid([radius * 0.94, radius * 1.08, radius * 0.86], fingertipMaterial, [side * 0.003, -tipLengths[i], 0.012], 16));
    middle.add(tip);
    finger.add(middle);
    finger.userData.middle = middle;
    finger.userData.tip = tip;
    hand.add(finger);
    hand.userData.fingers.push(finger);
  }
  const thumb = new THREE.Group();
  thumb.position.set(side * 0.135, -0.065, 0.025);
  thumb.rotation.z = side * 0.65;
  thumb.add(ellipsoid([0.045, 0.043, 0.041], darkJoint, [0, 0, 0], 16));
  thumb.add(capsuleBetween([0, 0, 0], [0, -0.08, 0.012], 0.039, handMaterial, 14));
  const thumbTip = new THREE.Group();
  thumbTip.position.set(0, -0.08, 0.012);
  thumbTip.add(capsuleBetween([0, 0, 0], [0, -0.06, 0.018], 0.035, handMaterial, 14));
  thumbTip.add(ellipsoid([0.037, 0.04, 0.034], fingertipMaterial, [0, -0.063, 0.018], 16));
  thumb.add(thumbTip);
  hand.add(thumb);
  hand.userData.thumb = thumb;
  hand.userData.thumbTip = thumbTip;
  hand.scale.set(1.04, 0.96, 1.04);
  return hand;
}

function makeArm(side) {
  const arm = new THREE.Group();
  arm.position.set(side * 0.7, 2.18, 0);
  arm.scale.setScalar(0.98);
  arm.add(ellipsoid([0.19, 0.19, 0.19], darkJoint, [0, 0, 0], 32));
  arm.add(ellipsoid([0.23, 0.22, 0.225], creamLight, [0, 0.025, 0], 40));
  const elbow = [side * 0.08, -0.43, 0.02];
  arm.add(capsuleBetween([0, -0.07, 0], elbow, 0.16, cream, 32));
  const hinge = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.155, 0.155, 0.22, 32), darkJoint));
  hinge.position.set(...elbow);
  hinge.rotation.z = Math.PI / 2;
  arm.add(hinge);
  const elbowRing = new THREE.Mesh(new THREE.TorusGeometry(0.105, 0.018, 12, 36), oliveMaterial);
  elbowRing.position.set(elbow[0] + side * 0.118, elbow[1], elbow[2]);
  elbowRing.rotation.y = Math.PI / 2;
  arm.add(elbowRing);

  const forearm = new THREE.Group();
  forearm.position.set(...elbow);
  const wrist = [side * 0.035, -0.4, 0.06];
  forearm.add(capsuleBetween([0, 0.025, 0], wrist, 0.155, creamLight, 32));
  const cuff = new THREE.Mesh(new THREE.TorusGeometry(0.125, 0.024, 12, 40), darkJoint);
  cuff.position.set(wrist[0], wrist[1] + 0.025, wrist[2]);
  cuff.rotation.x = Math.PI / 2;
  forearm.add(cuff);
  forearm.add(ellipsoid([0.115, 0.105, 0.105], darkJoint, wrist, 28));
  const hand = makeHand(side);
  hand.position.set(...wrist);
  forearm.add(hand);
  arm.add(forearm);
  character.add(arm);
  return { arm, forearm, hand, side };
}
const rightArm = makeArm(-1);
const leftArm = makeArm(1);

// Target-driven two-bone posing keeps the wrist on the prop while the pole
// controls a relaxed, outward elbow. All targets are in character space.
function handOrientation(fingerDirection, palmNormal) {
  const yAxis = fingerDirection.clone().negate().normalize();
  const zAxis = palmNormal.clone().addScaledVector(yAxis, -palmNormal.dot(yAxis)).normalize();
  const xAxis = new THREE.Vector3().crossVectors(yAxis, zAxis).normalize();
  const basis = new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis);
  return new THREE.Quaternion().setFromRotationMatrix(basis);
}
function poseArmOnTarget(rig, target, pole, fingerDirection, palmNormal, weight) {
  if (weight <= 0.0001) return;
  const { arm, forearm, hand, side } = rig;
  const scale = arm.scale.x;
  const wristTarget = target.clone().sub(arm.position).divideScalar(scale);
  const poleTarget = pole.clone().sub(arm.position).divideScalar(scale);
  const upperRest = new THREE.Vector3(side * 0.08, -0.43, 0.02);
  const lowerRest = new THREE.Vector3(side * 0.035, -0.4, 0.06);
  const upperLength = upperRest.length();
  const lowerLength = lowerRest.length();
  const distance = THREE.MathUtils.clamp(
    wristTarget.length(),
    Math.abs(upperLength - lowerLength) + 0.001,
    upperLength + lowerLength - 0.001
  );
  const direction = wristTarget.normalize();
  const along = (upperLength * upperLength + distance * distance - lowerLength * lowerLength) / (2 * distance);
  const bendHeight = Math.sqrt(Math.max(0, upperLength * upperLength - along * along));
  const bend = poleTarget.addScaledVector(direction, -poleTarget.dot(direction));
  if (bend.lengthSq() < 0.0001) bend.set(side, 0, 0);
  bend.normalize();
  const elbow = direction.clone().multiplyScalar(along).addScaledVector(bend, bendHeight);
  const reachableWrist = direction.clone().multiplyScalar(distance);

  const armTarget = new THREE.Quaternion().setFromUnitVectors(
    upperRest.normalize(), elbow.clone().normalize()
  );
  const lowerInArm = reachableWrist.sub(elbow).normalize().applyQuaternion(armTarget.clone().invert());
  const forearmTarget = new THREE.Quaternion().setFromUnitVectors(lowerRest.normalize(), lowerInArm);
  const desiredHand = handOrientation(fingerDirection, palmNormal);
  const handTarget = armTarget.clone().multiply(forearmTarget).invert().multiply(desiredHand);

  arm.quaternion.slerp(armTarget, weight);
  forearm.quaternion.slerp(forearmTarget, weight);
  hand.quaternion.slerp(handTarget, weight);
}

// Large cream head shell.
const HEAD_Y = 3.36;
const head = new THREE.Group();
head.position.set(0, HEAD_Y, 0);
head.scale.setScalar(0.96);
character.add(head);
const shell = ellipsoid([1.25, 1.02, 1.05], creamLight, [0, 0, 0], 80);
head.add(shell);

// Clover's signature curved antenna and five-leaf crown.
const antenna = new THREE.Group();
antenna.position.set(0, 0.82, 0.02);
antenna.add(tubeFrom(
  [[0, 0, 0], [-0.02, 0.2, 0.01], [0.12, 0.43, 0.025], [0.28, 0.56, 0.02]],
  0.052, oliveMaterial, 36
));
const antennaClover = makeClover(0.27);
antennaClover.position.set(0.28, 0.58, 0.02);
antennaClover.rotation.z = 0.12;
antenna.add(antennaClover);
head.add(antenna);

function facePanelGeometry() {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0.36);
  shape.bezierCurveTo(-0.25, 0.36, -0.52, 0.55, -0.78, 0.49);
  shape.bezierCurveTo(-1.02, 0.43, -1.08, 0.18, -1.06, -0.1);
  shape.bezierCurveTo(-1.03, -0.43, -0.61, -0.6, 0, -0.61);
  shape.bezierCurveTo(0.61, -0.6, 1.03, -0.43, 1.06, -0.1);
  shape.bezierCurveTo(1.08, 0.18, 1.02, 0.43, 0.78, 0.49);
  shape.bezierCurveTo(0.52, 0.55, 0.25, 0.36, 0, 0.36);
  return new THREE.ExtrudeGeometry(shape, {
    depth: 0.06, bevelEnabled: true, bevelSize: 0.045,
    bevelThickness: 0.07, bevelSegments: 6, curveSegments: 48
  });
}
const panelGeometry = facePanelGeometry();
const panelBorder = shadowed(new THREE.Mesh(panelGeometry, darkJoint));
panelBorder.position.set(0, -0.04, 0.9);
panelBorder.scale.set(1.018, 1.025, 1);
head.add(panelBorder);
const panel = shadowed(new THREE.Mesh(panelGeometry, faceMaterial));
panel.position.set(0, -0.04, 0.95);
head.add(panel);

// Compact olive side pods with luminous green rings; the rings answer the user's voice while listening.
const podGlow = greenGlow.clone();
const podRings = [];
for (const side of [-1, 1]) {
  const pod = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.18, 48), oliveMaterial));
  pod.position.set(side * 1.22, -0.02, 0);
  pod.rotation.z = Math.PI / 2;
  head.add(pod);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.27, 0.036, 16, 64), podGlow);
  ring.position.set(side * 1.32, -0.02, 0);
  ring.rotation.y = Math.PI / 2;
  head.add(ring);
  podRings.push(ring);
  const inner = shadowed(new THREE.Mesh(new THREE.CylinderGeometry(0.235, 0.235, 0.185, 40), faceMaterial));
  inner.position.set(side * 1.23, -0.02, 0);
  inner.rotation.z = Math.PI / 2;
  head.add(inner);
}

// Luminous 3D crescent cores with a soft, generated neon halo.
function makeEyeGlowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext('2d');
  ctx.lineCap = 'round';
  ctx.filter = 'blur(12px)';
  ctx.strokeStyle = 'rgba(142, 255, 32, 0.9)';
  ctx.lineWidth = 20;
  ctx.beginPath();
  ctx.moveTo(40, 88);
  ctx.quadraticCurveTo(128, 8, 216, 88);
  ctx.stroke();
  ctx.filter = 'blur(5px)';
  ctx.strokeStyle = 'rgba(184, 255, 80, 0.85)';
  ctx.lineWidth = 11;
  ctx.stroke();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
const eyeCurve = new THREE.QuadraticBezierCurve3(
  new THREE.Vector3(-0.255, -0.035, 0),
  new THREE.Vector3(0, 0.25, 0.02),
  new THREE.Vector3(0.255, -0.035, 0)
);
const eyeGeometry = new THREE.TubeGeometry(eyeCurve, 40, 0.037, 12, false);
const eyeSoftGlowMaterial = new THREE.MeshBasicMaterial({
  map: makeEyeGlowTexture(), transparent: true, opacity: 0.82,
  blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
  side: THREE.DoubleSide
});
const eyeSoftGlowGeometry = new THREE.PlaneGeometry(0.86, 0.46);
const eyes = [];
for (const side of [-1, 1]) {
  const eye = new THREE.Group();
  eye.position.set(side * 0.46, -0.08, 1.155);
  eye.userData.baseX = eye.position.x;
  const softGlow = new THREE.Mesh(eyeSoftGlowGeometry, eyeSoftGlowMaterial);
  softGlow.position.z = -0.03;
  eye.add(softGlow);
  eye.add(new THREE.Mesh(eyeGeometry, greenGlow));
  head.add(eye);
  eyes.push(eye);
}

// The reference's filled, open smile and soft cheek accents.
const mouthMaterial = greenGlow.clone();
mouthMaterial.emissiveIntensity = 2.2;
const mouthGlowMaterial = eyeGlowMaterial.clone();
mouthGlowMaterial.opacity = 0.15;
const mouthShape = new THREE.Shape();
mouthShape.moveTo(-0.16, 0.045);
mouthShape.quadraticCurveTo(0, 0.065, 0.16, 0.045);
mouthShape.bezierCurveTo(0.15, -0.025, 0.085, -0.105, 0, -0.12);
mouthShape.bezierCurveTo(-0.085, -0.105, -0.15, -0.025, -0.16, 0.045);
const mouthGeometry = new THREE.ShapeGeometry(mouthShape, 32);
const mouth = new THREE.Group();
mouth.position.set(0, -0.36, 1.165);
const mouthGlow = new THREE.Mesh(mouthGeometry, mouthGlowMaterial);
mouthGlow.scale.set(1.22, 1.18, 1);
mouthGlow.position.z = -0.006;
mouth.add(mouthGlow);
mouth.add(new THREE.Mesh(mouthGeometry, mouthMaterial));
head.add(mouth);

// A quiet mouth for focused activities; the open smile remains for social reactions.
const neutralMouth = new THREE.Group();
neutralMouth.position.set(0, -0.365, 1.17);
neutralMouth.add(tubeFrom([[-0.11, 0.012, 0], [0, -0.018, 0.012], [0.11, 0.012, 0]], 0.018, mouthMaterial, 24));
neutralMouth.visible = false;
head.add(neutralMouth);

// Three soft thought lights appear inside the visor during the thinking pose.
const thoughtDots = [];
const thoughtDotGeometry = new THREE.SphereGeometry(0.028, 16, 12);
for (let i = 0; i < 3; i++) {
  const dot = new THREE.Mesh(thoughtDotGeometry, greenGlow);
  dot.position.set(0.48 + i * 0.11, 0.31 + i * 0.075, 1.17);
  dot.visible = false;
  head.add(dot);
  thoughtDots.push(dot);
}

// Thought cloud with a two-bubble trail, attached to the character.
const thinkingCloud = new THREE.Group();
thinkingCloud.position.set(-0.88, 4.22, 0.12);
thinkingCloud.userData.baseY = thinkingCloud.position.y;
const cloudMaterial = new THREE.MeshPhysicalMaterial({
  color: 0xf1ead8, roughness: 0.76, clearcoat: 0.08, envMapIntensity: 0.45
});
const cloudParts = [
  [-0.18, 0, 0, 0.24], [0.08, 0.08, 0, 0.3], [0.34, 0, 0, 0.23],
  [0.06, -0.12, 0, 0.27], [-0.02, 0.23, 0, 0.2]
];
cloudParts.forEach(([x, y, z, radius]) => {
  thinkingCloud.add(ellipsoid([radius, radius * 0.82, radius * 0.72], cloudMaterial, [x, y, z], 28));
});
thinkingCloud.add(ellipsoid([0.075, 0.065, 0.055], cloudMaterial, [0.18, -0.43, 0], 20));
thinkingCloud.add(ellipsoid([0.045, 0.04, 0.035], cloudMaterial, [0.34, -0.65, 0], 16));
thinkingCloud.visible = false;
character.add(thinkingCloud);

// Working prop: a grounded child-height desk with the keyboard within reach.
const workingDesk = new THREE.Group();
workingDesk.position.z = 0.1;
const deskMaterial = new THREE.MeshPhysicalMaterial({
  color: 0x667645, roughness: 0.62, clearcoat: 0.16, clearcoatRoughness: 0.55
});
const laptopMaterial = new THREE.MeshPhysicalMaterial({
  color: 0x343c35, roughness: 0.3, metalness: 0.34, clearcoat: 0.38
});
const deskTop = shadowed(new THREE.Mesh(new RoundedBoxGeometry(1.45, 0.1, 0.82, 4, 0.05), deskMaterial));
deskTop.position.set(0, 1.15, 0.88);
workingDesk.add(deskTop);
for (const side of [-1, 1]) {
  for (const z of [0.62, 1.12]) {
    workingDesk.add(capsuleBetween([side * 0.56, 1.11, z], [side * 0.62, 0.03, z], 0.045, deskMaterial, 16));
  }
}
const laptopBase = shadowed(new THREE.Mesh(new RoundedBoxGeometry(0.88, 0.06, 0.58, 4, 0.025), laptopMaterial));
laptopBase.position.set(0, 1.26, 1.03);
workingDesk.add(laptopBase);
const keyboard = new THREE.Mesh(new RoundedBoxGeometry(0.66, 0.012, 0.3, 2, 0.01), darkJoint);
keyboard.position.set(0, 1.298, 0.91);
workingDesk.add(keyboard);
const laptopScreen = new THREE.Group();
laptopScreen.position.set(0, 1.57, 1.25);
laptopScreen.rotation.x = 0.12;
const laptopLid = shadowed(new THREE.Mesh(new RoundedBoxGeometry(0.88, 0.56, 0.055, 4, 0.035), laptopMaterial));
laptopScreen.add(laptopLid);
const displayMaterial = new THREE.MeshStandardMaterial({
  color: 0x10251c, emissive: 0x193d28, emissiveIntensity: 0.5, roughness: 0.28
});
const laptopDisplay = new THREE.Mesh(new RoundedBoxGeometry(0.74, 0.43, 0.012, 3, 0.025), displayMaterial);
laptopDisplay.position.z = -0.036;
laptopScreen.add(laptopDisplay);
for (let line = 0; line < 3; line++) {
  const displayLine = new THREE.Mesh(
    new RoundedBoxGeometry(line === 2 ? 0.24 : 0.4, 0.018, 0.008, 2, 0.004),
    new THREE.MeshBasicMaterial({ color: 0x9cff28, toneMapped: false })
  );
  displayLine.position.set(0, 0.1 - line * 0.08, -0.045);
  laptopScreen.add(displayLine);
}
const laptopClover = makeClover(0.075);
laptopClover.position.set(0, 0.06, 0.07);
laptopScreen.add(laptopClover);
workingDesk.add(laptopScreen);
workingDesk.visible = false;
scene.add(workingDesk);

// Reading prop: an open, slightly folded book with individual pages and ink lines.
const readingBook = new THREE.Group();
readingBook.position.set(0, 1.58, 0.88);
readingBook.rotation.set(0.32, Math.PI, 0);
const coverMaterial = new THREE.MeshPhysicalMaterial({ color: 0x4b681f, roughness: 0.72, sheen: 0.3 });
const pageMaterial = new THREE.MeshStandardMaterial({ color: 0xf4ead3, roughness: 0.9 });
for (const side of [-1, 1]) {
  const half = new THREE.Group();
  half.position.x = side * 0.235;
  half.rotation.y = -side * 0.18;
  half.add(shadowed(new THREE.Mesh(new RoundedBoxGeometry(0.48, 0.64, 0.045, 3, 0.025), coverMaterial)));
  const pages = shadowed(new THREE.Mesh(new RoundedBoxGeometry(0.43, 0.58, 0.035, 3, 0.018), pageMaterial));
  pages.position.z = 0.04;
  half.add(pages);
  for (let line = 0; line < 4; line++) {
    const ink = new THREE.Mesh(new RoundedBoxGeometry(line === 3 ? 0.18 : 0.28, 0.018, 0.008, 2, 0.004), oliveMaterial);
    ink.position.set(0, 0.16 - line * 0.1, 0.063);
    half.add(ink);
  }
  readingBook.add(half);
}
readingBook.visible = false;
character.add(readingBook);

// Sleepy prop: three dimensional Z letters that rise above the head.
const sleepyLetters = [];
for (let i = 0; i < 3; i++) {
  const letter = new THREE.Group();
  const material = new THREE.MeshBasicMaterial({ color: 0x6c9680, transparent: true, opacity: 0 });
  const size = 0.13 + i * 0.025;
  letter.add(capsuleBetween([-size, size, 0], [size, size, 0], 0.025, material, 12));
  letter.add(capsuleBetween([size, size, 0], [-size, -size, 0], 0.025, material, 12));
  letter.add(capsuleBetween([-size, -size, 0], [size, -size, 0], 0.025, material, 12));
  letter.position.set(0.68 + i * 0.22, 3.95 + i * 0.27, 0.22);
  letter.userData.baseY = letter.position.y;
  letter.userData.material = material;
  letter.visible = false;
  character.add(letter);
  sleepyLetters.push(letter);
}

const cheekMaterial = new THREE.MeshBasicMaterial({
  color: 0xcfff86, transparent: true, opacity: 0.09,
  blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
});
const cheeks = [];
for (const side of [-1, 1]) {
  const cheek = ellipsoid([0.075, 0.03, 0.012], cheekMaterial, [side * 0.64, -0.19, 1.15], 24);
  cheek.castShadow = false;
  head.add(cheek);
  cheeks.push(cheek);
}

// Deterministic sparkles for the happy reaction.
const sparkles = [];
const sparkleGeometry = new THREE.IcosahedronGeometry(0.06, 1);
const sparkleColors = [0x5bcf8a, 0xc4ed8a, 0xffd675];
for (let i = 0; i < 18; i++) {
  const material = new THREE.MeshStandardMaterial({
    color: sparkleColors[i % 3], roughness: 0.32,
    emissive: sparkleColors[i % 3], emissiveIntensity: 0.1
  });
  const sparkle = new THREE.Mesh(sparkleGeometry, material);
  sparkle.visible = false;
  sparkle.userData.angle = (i / 18) * TAU + (rand() - 0.5) * 0.22;
  sparkle.userData.radius = 1.65 + rand() * 0.65;
  sparkle.userData.lift = 0.3 + rand() * 1.25;
  sparkle.userData.delay = rand() * 0.18;
  scene.add(sparkle);
  sparkles.push(sparkle);
}

const pointer = new THREE.Vector2();
const look = new THREE.Vector2();
let oneShot = null;    // a timed reaction (a tap, the demo buttons) with its own entrance, hold and exit
let stateRx = null;    // a lasting state set by the host (listening, thinking, ...) until the next one
let stateFade = null;  // the previous lasting state, fading out
let speechState = null;
let level = 0;         // the host's live voice level, 0..1 (the user's microphone while listening)
let levelSmooth = 0;
let statusListener = null;
function notify(copy) {
  if (statusListener && copy) statusListener(copy[0], copy[1]);
}
const actionCopy = {
  stand: ['Ready when you are', 'Your clover robot is standing.'],
  sit: ['Taking a seat', 'Your clover robot is sitting with its legs bent.'],
  walk: ['Walking with you', 'Your clover robot is walking.'],
  dance: ['Dancing with you', 'Your clover robot is dancing.']
};
let action = 'stand';
let actionFrom = 'stand';
let actionStart = -1;
const reactionCopy = {
  wave: ['Hello there!', 'Your clover robot is waving hello.'],
  happy: ['So happy to see you!', 'Your clover robot is happy.'],
  nod: ['Following you…', 'Your clover robot nods attentively while listening.'],
  thinking: ['Thinking it through…', 'Your clover robot is thinking with its hand on its chin.'],
  working: ['Focused on the task…', 'Your clover robot is working with a focused expression.'],
  reading: ['Reading carefully…', 'Your clover robot is reading with lowered eyes.'],
  answering: ['Here is my answer!', 'Your clover robot is answering with a talking expression.'],
  sleepy: ['Taking a tiny rest…', 'Your clover robot is sleepy.'],
  listening: ['Listening…', 'Your clover robot is listening to you.']
};
const reactionDuration = {
  wave: 2.8, happy: 1.65, nod: 1.05, thinking: 5.6, working: 9.2,
  reading: 8.6, answering: 3.8, sleepy: 2.8, listening: 4.2
};
// Lasting states end only when the host sets another; their duration is just the period of their motion.
const LASTING = new Set(['listening', 'thinking', 'working', 'reading', 'answering']);
const STATE_ENTER = 0.45;
const STATE_EXIT = 0.55;

// Lightweight grapheme-to-viseme approximation. The host can synchronise this
// with any TTS engine by calling __startLipSync(text, audioDurationSeconds).
const VISEMES = {
  rest: { width: 0.012, open: 0.004 },
  closed: { width: 0.035, open: 0.003 },       // M, B, P
  teeth: { width: 0.052, open: 0.018 },        // F, V
  wide: { width: 0.072, open: 0.028 },         // EE, IH
  open: { width: 0.064, open: 0.082 },         // AH, AE
  round: { width: 0.046, open: 0.064 },        // OH
  smallRound: { width: 0.030, open: 0.045 },   // OO, W
  soft: { width: 0.050, open: 0.024 }
};
function charViseme(char) {
  const c = char.toLowerCase();
  if ('mbp'.includes(c)) return 'closed';
  if ('fv'.includes(c)) return 'teeth';
  if ('a'.includes(c)) return 'open';
  if ('eiy'.includes(c)) return 'wide';
  if ('o'.includes(c)) return 'round';
  if ('uwq'.includes(c)) return 'smallRound';
  if (/[.,!?;:]/.test(c)) return 'rest';
  return 'soft';
}
function buildVisemeTimeline(text, durationSeconds) {
  const chars = [...String(text || '')];
  if (!chars.length) return [];
  const weights = chars.map((c) => /[.,!?;:]/.test(c) ? 1.7 : /[aeiouy]/i.test(c) ? 1.55 : /\s/.test(c) ? 0.85 : 1);
  const total = weights.reduce((a, b) => a + b, 0);
  let cursor = 0;
  const timeline = [];
  chars.forEach((c, i) => {
    const span = durationSeconds * weights[i] / total;
    const viseme = /\s/.test(c) ? 'rest' : charViseme(c);
    const previous = timeline[timeline.length - 1];
    if (previous && previous.viseme === viseme && cursor - previous.end < 0.0001) previous.end += span;
    else timeline.push({ start: cursor, end: cursor + span, viseme });
    cursor += span;
  });
  return timeline;
}
function startLipSync(text, durationSeconds = Math.max(0.8, String(text || '').length / 12)) {
  const clean = String(text || '').trim();
  if (!clean) { speechState = null; return; }
  const prevPose = speechPoseAt(simT);
  const duration = Math.max(0.15, Number(durationSeconds) || 0.15);
  speechState = { text: clean, start: simT, duration, timeline: buildVisemeTimeline(clean, duration), prevPose };
}
function stopLipSync() { speechState = null; }
function speechPoseAt(t) {
  if (!speechState) return null;
  const elapsed = t - speechState.start;
  if (elapsed < 0 || elapsed > speechState.duration) {
    if (elapsed > speechState.duration) speechState = null;
    return null;
  }
  const idx = speechState.timeline.findIndex((item) => elapsed >= item.start && elapsed < item.end);
  if (idx < 0) return VISEMES.rest;
  const segment = speechState.timeline[idx];
  const pose = VISEMES[segment.viseme] || VISEMES.soft;
  const prev = idx > 0
    ? (VISEMES[speechState.timeline[idx - 1].viseme] || VISEMES.soft)
    : (speechState.prevPose || VISEMES.rest);
  const next = idx + 1 < speechState.timeline.length
    ? (VISEMES[speechState.timeline[idx + 1].viseme] || VISEMES.soft)
    : VISEMES.rest;
  // Smoothly coarticulate between adjacent visemes instead of snapping to rest between every character/word.
  const inBlend = THREE.MathUtils.smoothstep(Math.min(1, (elapsed - segment.start) / 0.045), 0, 1);
  const outBlend = idx + 1 === speechState.timeline.length
    ? THREE.MathUtils.smoothstep(Math.min(1, (segment.end - elapsed) / 0.055), 0, 1)
    : 1;
  const w = THREE.MathUtils.lerp(prev.width, pose.width, inBlend);
  const o = THREE.MathUtils.lerp(prev.open, pose.open, inBlend);
  return {
    width: THREE.MathUtils.lerp(next.width, w, outBlend),
    open: THREE.MathUtils.lerp(next.open, o, outBlend)
  };
}

function setReaction(type) {
  if (!reactionDuration[type]) return;
  if (oneShot && oneShot.type === type && (simT - oneShot.start) < reactionDuration[type] * 0.85) return;
  oneShot = { type, start: simT };
  notify(reactionCopy[type]);
}
function setState(name) {
  if (name && !LASTING.has(name) && name !== 'idle') {
    setReaction(name);
    return;
  }
  const next = LASTING.has(name) ? name : null;
  if (next !== 'listening' && oneShot && oneShot.type === 'nod') oneShot = null;
  if ((stateRx ? stateRx.type : null) === next) return;
  if (stateRx) {
    stateFade = { type: stateRx.type, start: stateRx.start, at: simT,
      from: smoothRange(simT - stateRx.start, 0, STATE_ENTER) };
  }
  stateRx = next ? { type: next, start: simT } : null;
  notify(next ? reactionCopy[next] : actionCopy[action]);
}
function setLevel(value) {
  level = THREE.MathUtils.clamp(Number(value) || 0, 0, 1);
}
function setAction(type) {
  if (!actionCopy[type]) return;
  oneShot = null;
  if (type === action) {
    notify(actionCopy[type]);
    return;
  }
  actionFrom = action;
  action = type;
  actionStart = simT;
  notify(actionCopy[type]);
}
renderer.domElement.addEventListener('pointerdown', () => {
  if (!stateRx && !speechState) setReaction('wave');
});
addEventListener('pointermove', (event) => {
  const rect = renderer.domElement.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  pointer.set(
    THREE.MathUtils.clamp(((event.clientX - rect.left) / rect.width) * 2 - 1, -1.2, 1.2),
    THREE.MathUtils.clamp(-(((event.clientY - rect.top) / rect.height) * 2 - 1), -1.2, 1.2)
  );
});
document.documentElement.addEventListener('mouseleave', () => pointer.set(0, 0));

function blinkAt(t) {
  // Irregular blinks with an occasional double blink feel less mechanical.
  const period = 13.4;
  const cycle = ((t % period) + period) % period;
  const events = [2.15, 5.85, 6.12, 10.95];
  let closure = 0;
  for (const event of events) {
    const delta = Math.abs(cycle - event);
    if (delta < 0.16) closure = Math.max(closure, Math.cos((delta / 0.16) * (Math.PI / 2)) * 0.92);
  }
  return 1 - closure;
}
function actionWeights(t) {
  const weights = { stand: 0, sit: 0, walk: 0, dance: 0 };
  const u = THREE.MathUtils.clamp((t - actionStart) / 0.9, 0, 1);
  const blend = u * u * (3 - 2 * u);
  weights[actionFrom] += 1 - blend;
  weights[action] += blend;
  return weights;
}
function smoothRange(value, start, end) {
  const u = THREE.MathUtils.clamp((value - start) / (end - start), 0, 1);
  return u * u * (3 - 2 * u);
}
function reactionValues(t) {
  const values = {
    wave: 0, happy: 0, nod: 0, sleepy: 0, thinking: 0, working: 0,
    reading: 0, answering: 0, listening: 0, u: 0, stateU: 0, jump: 0
  };
  // A lasting state's phase keeps growing (u > 1) so its periodic motion never jumps back.
  if (stateFade) {
    const k = stateFade.from * (1 - smoothRange(t, stateFade.at, stateFade.at + STATE_EXIT));
    if (k <= 0.0001) stateFade = null;
    else {
      values[stateFade.type] = k;
      values.u = (t - stateFade.start) / reactionDuration[stateFade.type];
      values.stateU = values.u;
    }
  }
  if (stateRx) {
    const elapsed = Math.max(0, t - stateRx.start);
    values[stateRx.type] = Math.max(values[stateRx.type], smoothRange(elapsed, 0, STATE_ENTER));
    values.u = elapsed / reactionDuration[stateRx.type];
    values.stateU = values.u;
  }
  if (!oneShot) return values;

  const elapsed = t - oneShot.start;
  if (elapsed < 0) return values;
  const duration = reactionDuration[oneShot.type];
  if (elapsed >= duration) {
    oneShot = null;
    notify(stateRx ? reactionCopy[stateRx.type] : actionCopy[action]);
    return values;
  }

  const u = THREE.MathUtils.clamp(elapsed / duration, 0, 1);
  const sustained = ['thinking', 'working', 'reading', 'sleepy', 'listening'].includes(oneShot.type);
  const enter = sustained ? 0.42 : oneShot.type === 'nod' ? 0.24 : 0.32;
  const exit = sustained ? 0.62 : oneShot.type === 'nod' ? 0.32 : 0.48;
  const fadeIn = smoothRange(elapsed, 0, enter);
  const fadeOut = 1 - smoothRange(elapsed, duration - exit, duration);
  const envelope = fadeIn * fadeOut;

  values.u = u;
  if (!stateRx && !stateFade) values.stateU = u;
  values[oneShot.type] = Math.max(values[oneShot.type], envelope);
  if (oneShot.type === 'happy') {
    // A soft anticipation, one small hop, then a cushioned landing.
    const hop = Math.sin(Math.PI * THREE.MathUtils.clamp((u - 0.12) / 0.52, 0, 1));
    values.jump = hop * 0.105 * motion * envelope;
  }
  return values;
}
function sample(t) {
  const idle = Math.sin((t / 5.8) * TAU) * motion;
  const slow = Math.sin((t / 8.2) * TAU) * motion;
  const halfLoop = (1 - Math.cos((t / DURATION) * TAU)) * motion;
  const r = reactionValues(t);
  const sU = r.stateU || r.u;
  const a = actionWeights(t);
  const sit = a.sit;
  const dance = a.dance;
  const danceMove = dance * motion;
  const danceBeat = Math.sin((t / 1.72) * TAU);
  const wave = r.wave * motion;
  const liveListen = r.listening * levelSmooth;
  const nodDip = r.nod * Math.sin(r.u * TAU * 1.5) * 0.11 * motion;
  const activity = Math.max(r.thinking, r.working, r.reading, r.answering, r.listening);
  const walkPhase = (t / 1.28) * TAU;
  const activityMotion = (1 - r.sleepy * 0.85) * (1 - activity * 0.78) * a.walk;
  const walk = Math.sin(walkPhase) * motion * activityMotion;
  const armFollow = Math.sin(walkPhase - 0.45) * motion * activityMotion;
  const walkBob = (0.5 - Math.cos(walkPhase * 2) * 0.5) * 0.032 * motion * activityMotion;
  const handWave = Math.sin(r.u * TAU * 2.6);
  const talk = 0.3 + Math.abs(Math.sin(sU * TAU * 3.4)) * 0.85;

  character.position.y = r.jump + walkBob - sit * 0.52 - r.working * 0.12
    + Math.abs(Math.sin((t / 1.2) * TAU)) * 0.07 * danceMove;
  character.rotation.y = slow * 0.018 + danceMove * danceBeat * 0.07;
  character.rotation.z = walk * 0.012 + idle * 0.006 + danceMove * danceBeat * 0.045;
  chair.visible = sit > 0.04;
  chair.scale.setScalar(1);
  torso.position.x = idle * 0.012;
  torso.rotation.x = liveListen * 0.065 + r.nod * 0.025;
  torso.scale.set(1 + idle * 0.006, 1 - idle * 0.004, 0.76 * (1 + idle * 0.008));
  head.position.x = idle * 0.012;
  head.position.y = HEAD_Y + idle * 0.022 + halfLoop * 0.018 + r.happy * 0.045 - r.sleepy * 0.085 - r.reading * 0.045 + r.working * 0.018
    + r.listening * (0.03 + levelSmooth * 0.018);
  head.rotation.y = look.x * 0.23 + slow * 0.018 + r.thinking * 0.14 + Math.sin(sU * TAU * 1.7) * r.answering * 0.028
    + Math.sin(t * 1.8) * liveListen * 0.025;
  head.rotation.x = -look.y * 0.085 + r.sleepy * 0.085 + Math.sin(walkPhase * 2 - 0.3) * 0.01 * motion
    + r.reading * 0.2 - r.thinking * 0.065 + r.working * 0.045
    + Math.sin(sU * TAU * 1.7) * r.answering * 0.035 - r.listening * (0.05 + levelSmooth * 0.04)
    + Math.sin(t * 3.0) * liveListen * 0.028 + nodDip;
  head.rotation.z = r.sleepy * 0.14 - wave * 0.045 - walk * 0.025 + slow * 0.012 - r.thinking * 0.1 + r.working * 0.018
    + r.listening * (0.08 + levelSmooth * 0.035);
  cape.rotation.x = 0.02 + armFollow * 0.04 + sit * 0.08 + danceMove * danceBeat * 0.04;
  cape.rotation.z = slow * 0.018 - walk * 0.028 + danceMove * danceBeat * 0.05;
  antenna.rotation.z = Math.sin(t * 1.15) * 0.025 * motion + wave * handWave * 0.035 + danceMove * danceBeat * 0.025
    + r.listening * levelSmooth * 0.09 + r.nod * 0.05;
  antenna.rotation.x = Math.sin(t * 0.8) * 0.012 * motion + liveListen * 0.03;

  legs.forEach(({ leg, lower, foot, side }) => {
    const stride = -walk * side * 0.31;
    const kneeLift = Math.max(0, -walk * side);
    leg.rotation.x = stride * (1 - sit) - sit * 1.18 + danceMove * side * danceBeat * 0.12;
    lower.rotation.x = kneeLift * 0.52 * (1 - sit) + sit * 1.08
      + danceMove * Math.max(0, side * danceBeat) * 0.18;
    foot.rotation.x = -(stride + lower.rotation.x) * 0.72 * (1 - sit) + sit * 0.1
      - danceMove * side * danceBeat * 0.05;
    foot.rotation.z = side * walk * 0.035 * (1 - sit) + side * danceMove * danceBeat * 0.035;
  });

  [rightArm, leftArm].forEach((rig) => {
    rig.arm.position.set(
      rig.side * (0.7 - r.working * 0.06 - r.reading * 0.02),
      2.18 - r.working * 0.16 - r.reading * 0.1,
      r.working * 0.48 + r.reading * 0.3
    );
  });
  rightArm.arm.rotation.x = armFollow * 0.2 * (1 - wave) - sit * 0.1 - danceMove * 0.12 - liveListen * 0.14;
  rightArm.arm.rotation.y = 0;
  rightArm.arm.rotation.z = -0.38 * (1 - wave) - 1.65 * wave - r.happy * 0.22 + idle * 0.008 * (1 - wave)
    - r.answering * 0.18 - liveListen * 0.22 + sit * 0.16 - dance * 0.55 - danceMove * danceBeat * 0.22;
  rightArm.forearm.rotation.x = -0.28 * (1 - wave) - sit * 0.15 - liveListen * 0.24;
  rightArm.forearm.rotation.z = -0.42 * (1 - wave) + wave * (-0.85 + handWave * 0.13)
    - r.answering * 0.1 - liveListen * 0.14 - dance * 0.25 + danceMove * danceBeat * 0.12;
  rightArm.hand.rotation.x = -0.45 * (1 - wave) + sit * 0.2;
  rightArm.hand.rotation.z = -wave * handWave * 0.2 + Math.sin(t * 4.2) * liveListen * 0.08;
  leftArm.arm.rotation.x = -armFollow * 0.2 - sit * 0.1 - danceMove * 0.12;
  leftArm.arm.rotation.z = 0.38 + r.happy * 0.22 - idle * 0.008 + r.sleepy * 0.04
    + r.answering * 0.12 - sit * 0.16 + dance * 0.55 - danceMove * danceBeat * 0.22;
  leftArm.forearm.rotation.x = -0.28 - r.happy * 0.1 - sit * 0.15;
  leftArm.forearm.rotation.z = 0.42 + r.sleepy * -0.06 + r.answering * 0.15
    + dance * 0.25 + danceMove * danceBeat * 0.12;
  leftArm.hand.rotation.x = -0.45 + sit * 0.2;
  leftArm.hand.rotation.z = -r.answering * 0.12;

  if (r.thinking > 0.0001) {
    head.updateMatrix();
    const chinTarget = new THREE.Vector3(-0.27, -1.1, 0.58).applyMatrix4(head.matrix);
    chinTarget.x += Math.sin(t * 1.15) * 0.006 * motion;
    poseArmOnTarget(
      rightArm, chinTarget, new THREE.Vector3(-1.12, 2.04, 0.28),
      new THREE.Vector3(0.06, 1, -0.08), new THREE.Vector3(0, -0.08, -1), r.thinking
    );
  }
  [rightArm, leftArm].forEach((rig, index) => {
    const side = rig.side;
    if (r.working > 0.0001) {
      const typing = Math.sin(t * TAU * 3.2 + index * Math.PI) * motion;
      const keyboardTarget = new THREE.Vector3(
        side * 0.23 + typing * 0.008,
        1.52 - character.position.y + Math.max(0, typing) * 0.012,
        0.85
      );
      poseArmOnTarget(
        rig, keyboardTarget, new THREE.Vector3(side * 1.08, 1.58, 0.3),
        new THREE.Vector3(0, -0.78, 0.62), new THREE.Vector3(0, -0.62, -0.78), r.working
      );
    }
    if (r.reading > 0.0001) {
      const bookEdge = new THREE.Vector3(side * 0.52, 1.66 + slow * 0.006 * motion, 0.86);
      poseArmOnTarget(
        rig, bookEdge, new THREE.Vector3(side * 1.05, 1.58, 0.18),
        new THREE.Vector3(0, -1, 0.02), new THREE.Vector3(-side, 0, 0), r.reading
      );
    }
  });

  [rightArm, leftArm].forEach(({ hand }, handIndex) => {
    const openWave = hand === rightArm.hand ? wave : 0;
    const relaxed = 1 - openWave;
    const thinkingGrip = hand === rightArm.hand ? r.thinking : 0;
    hand.userData.fingers.forEach((finger, index) => {
      const typing = r.working * Math.max(0, Math.sin(t * TAU * 3.2 + index * 0.9 + handIndex * Math.PI));
      const thinkingCurl = thinkingGrip * (index === 1 ? 0.04 : 0.38);
      finger.rotation.x = -0.025 - relaxed * 0.04 - r.happy * 0.08 - r.sleepy * 0.1
        + openWave * handWave * 0.015 * (index - 1.5) - typing * 0.12
        - r.reading * 0.2 - thinkingCurl;
      finger.userData.middle.rotation.x = -0.03 - relaxed * 0.08 - r.happy * 0.1 - r.sleepy * 0.12
        - typing * 0.1 - r.reading * 0.34 - thinkingGrip * (index === 1 ? 0.06 : 0.52);
      finger.userData.tip.rotation.x = -0.02 - relaxed * 0.05 - r.happy * 0.08 - r.sleepy * 0.1
        - typing * 0.06 - r.reading * 0.18 - thinkingGrip * (index === 1 ? 0.02 : 0.3);
    });
    hand.userData.thumb.rotation.x = -0.06 - r.happy * 0.08 - r.sleepy * 0.06 + openWave * 0.03
      - r.reading * 0.24 - thinkingGrip * 0.12;
    hand.userData.thumbTip.rotation.x = -0.08 - r.happy * 0.1 - r.sleepy * 0.08
      - r.reading * 0.2 - thinkingGrip * 0.08;
  });

  const openness = blinkAt(t) * (1 - r.sleepy * 0.72);
  eyes.forEach((eye) => {
    const side = eye.userData.baseX < 0 ? -1 : 1;
    eye.position.x = eye.userData.baseX + look.x * 0.022 + r.thinking * 0.045
      + Math.sin(t * 1.35) * 0.022 * r.reading * motion;
    eye.position.y = -0.08 + look.y * 0.015 - r.sleepy * 0.05 + r.thinking * 0.09 - r.reading * 0.1 - r.working * 0.025
      + r.listening * (0.02 + levelSmooth * 0.022) + r.nod * 0.015;
    eye.rotation.z = r.thinking * side * 0.12 - r.working * side * 0.1;
    eye.scale.set(
      1.05 + r.happy * 0.1 + wave * 0.05 + r.answering * 0.04 + liveListen * 0.06 - r.working * 0.06,
      Math.max(0.15, openness) * (1.05 + r.happy * 0.1)
        * (1 - r.reading * 0.42 - r.working * 0.22 - r.thinking * 0.12 - r.nod * 0.14 + r.answering * 0.05 + r.listening * (0.08 + levelSmooth * 0.12)),
      1
    );
  });
  greenGlow.emissiveIntensity = 3.4 + r.happy * 1.2 + wave * 0.35 - r.sleepy * 2.2 + r.answering * 0.6 + r.working * 0.2 - r.reading * 0.35 + liveListen * 0.45;
  podGlow.emissiveIntensity = greenGlow.emissiveIntensity + r.listening * (0.6 + levelSmooth * 4.4);
  podRings.forEach((ring) => ring.scale.setScalar(1 + r.listening * levelSmooth * 0.16));
  chestGlow.material.opacity = 0.72 + liveListen * 0.26;
  chestClover.rotation.z = liveListen * Math.sin(t * 4.8) * 0.07;
  eyeSoftGlowMaterial.opacity = Math.max(0.25, 0.82 + r.happy * 0.1 - r.sleepy * 0.45 + r.answering * 0.08 + liveListen * 0.08 - r.reading * 0.2);
  const speechPose = speechPoseAt(t) || (r.answering > 0.025 ? { width: 0.042 + talk * 0.012, open: 0.018 + talk * 0.035 } : null);
  const isSpeaking = Boolean(speechPose);
  const listenSmile = Math.max(r.nod * 0.65, liveListen > 0.12 ? (liveListen - 0.12) * 0.55 : 0);
  mouth.visible = isSpeaking || Math.max(r.happy, wave, r.answering, listenSmile, dance * 0.8) > 0.025;
  const smile = Math.max(r.happy, wave, listenSmile, dance * 0.8);
  if (speechPose) {  // speaking: the white light itself opens and closes, from a sliver at rest to the full smile
    mouth.scale.set(0.62 + speechPose.width * 6.5, 0.22 + speechPose.open * 9.5, 1);
  } else {
    mouth.scale.set(
      1 + smile * 0.22 + r.answering * 0.08,
      Math.max(0.18, 0.72 + smile * 0.25 - r.sleepy * 0.3),
      1
    );
  }
  mouth.position.y = -0.36 - r.sleepy * 0.02;
  mouthMaterial.emissiveIntensity = 2.0 + r.happy * 1.0 + wave * 0.25 - r.sleepy * 1.0 + r.answering * 0.3 + listenSmile * 0.4
    + (speechPose ? speechPose.open * 6 : 0);
  mouthGlowMaterial.opacity = 0.12 + r.happy * 0.07 - r.sleepy * 0.06 + r.answering * 0.03 + listenSmile * 0.04;
  neutralMouth.visible = !mouth.visible;
  neutralMouth.scale.set(1 - r.thinking * 0.25 - r.reading * 0.15, 1 - r.working * 0.25, 1);
  neutralMouth.rotation.z = -r.thinking * 0.12;
  thoughtDots.forEach((dot, index) => {
    dot.visible = r.thinking > 0.025;
    const pulse = 0.65 + 0.35 * Math.sin(r.u * TAU * 3 - index * 0.8);
    dot.scale.setScalar(Math.max(0.01, r.thinking * pulse));
  });
  thinkingCloud.visible = r.thinking > 0.025;
  thinkingCloud.scale.setScalar(Math.max(0.001, r.thinking));
  thinkingCloud.position.y = thinkingCloud.userData.baseY + Math.sin(r.u * TAU) * 0.05 * r.thinking;
  workingDesk.visible = r.working > 0.001;
  workingDesk.scale.setScalar(1);
  readingBook.visible = r.reading > 0.001;
  readingBook.scale.setScalar(1);
  readingBook.rotation.z = Math.sin(t * 0.85) * 0.012 * r.reading * motion;
  sleepyLetters.forEach((letter, index) => {
    const cycle = (r.u + index * 0.22) % 1;
    const fade = Math.sin(Math.PI * cycle);
    letter.visible = r.sleepy > 0.025;
    letter.userData.material.opacity = r.sleepy * (0.62 + index * 0.1) * fade;
    letter.scale.setScalar(Math.max(0.001, r.sleepy * (0.8 + index * 0.12)));
    letter.position.y = letter.userData.baseY + cycle * (0.1 + index * 0.04) * r.sleepy;
    letter.rotation.z = Math.sin(cycle * TAU + index) * 0.08 * r.sleepy;
  });
  cheekMaterial.opacity = Math.max(0, r.happy * 0.12 + wave * 0.025 + r.answering * 0.04 + liveListen * 0.07 + r.nod * 0.06);

  contactShadow.scale.setScalar(1 - r.jump * 0.18);
  contactShadow.material.opacity = 0.72 - r.jump * 0.3;

  sparkles.forEach((sparkle) => {
    if (r.happy <= 0 || reducedMotion) {
      sparkle.visible = false;
      return;
    }
    const localU = THREE.MathUtils.clamp((r.u - sparkle.userData.delay) / (1 - sparkle.userData.delay), 0, 1);
    const angle = sparkle.userData.angle + localU * 0.5;
    const radius = sparkle.userData.radius + localU * 0.5;
    sparkle.visible = localU > 0 && localU < 1;
    sparkle.position.set(
      Math.cos(angle) * radius,
      2.05 + localU * sparkle.userData.lift,
      Math.sin(angle) * radius * 0.45 + 0.45
    );
    sparkle.scale.setScalar(Math.sin(Math.PI * localU) * (0.7 + sparkle.userData.lift * 0.3));
    sparkle.rotation.set(localU * 3, localU * 4, localU * 2);
  });
}

const clock = new THREE.Clock();
const STEP = 1 / 60;
let acc = 0;
let simT = 0;
let paused = false;

function draw() {
  renderer.render(scene, camera);
}
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (!paused) {
    acc += dt;
    while (acc >= STEP) {
      acc -= STEP;
      simT += STEP;
      look.lerp(pointer, 0.075);
      // Fast rise, gentle release so syllable gaps never make the body/head/smile chatter.
      levelSmooth += (level - levelSmooth) * (level > levelSmooth ? 0.14 : 0.045);
    }
    sample(simT);
  }
  draw();
}

const seek = (t) => {
  const value = Number(t);
  paused = true;
  acc = 0;
  simT = Number.isFinite(value) && value > 0 ? value : 0;
  levelSmooth = level;
  sample(simT);
  draw();
};
const play = () => {
  paused = false;
  clock.getDelta();
};
const speakText = (text, options = {}) => {
  if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) {
    throw new Error('Browser speech synthesis is not available; use __startLipSync(text, durationSeconds) with your TTS audio.');
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(String(text || ''));
  utterance.lang = options.lang || 'en-IN';
  utterance.rate = options.rate || 1;
  utterance.pitch = options.pitch || 1.05;
  const estimatedDuration = Math.max(0.8, String(text || '').length / (12 * utterance.rate));
  utterance.onstart = () => startLipSync(text, estimatedDuration);
  utterance.onend = stopLipSync;
  utterance.onerror = stopLipSync;
  window.speechSynthesis.speak(utterance);
  return utterance;
};
const debugState = () => ({
  ready: true,
  state: stateRx?.type ?? 'idle',
  level: levelSmooth,
  running,
  action,
  sitWeight: actionWeights(simT).sit,
  chairVisible: chair.visible,
  danceWeight: actionWeights(simT).dance,
  reaction: oneShot?.type ?? 'idle',
  time: simT,
  waveAngle: rightArm.arm.rotation.z,
  headYaw: head.rotation.y,
  headPitch: head.rotation.x,
  eyeY: eyes[0].position.y,
  eyeScaleY: eyes[0].scale.y,
  mouthSmile: mouth.scale.x,
  mouthVisible: mouth.visible,
  mouthOpen: mouth.scale.y,
  speakingText: speechState?.text ?? null,
  neutralMouthVisible: neutralMouth.visible,
  thinkingCloudVisible: thinkingCloud.visible,
  workingDeskVisible: workingDesk.visible,
  readingBookVisible: readingBook.visible,
  sleepyLettersVisible: sleepyLetters.some((letter) => letter.visible),
  rightForearmAngle: rightArm.forearm.rotation.z,
  rightWrist: rightArm.hand.getWorldPosition(new THREE.Vector3()).toArray(),
  leftWrist: leftArm.hand.getWorldPosition(new THREE.Vector3()).toArray(),
  workingWeight: reactionValues(simT).working,
  readingWeight: reactionValues(simT).reading,
  listeningWeight: reactionValues(simT).listening,
  nodWeight: reactionValues(simT).nod,
  torsoLean: torso.rotation.x,
  podRingScale: podRings[0].scale.x,
  podGlowIntensity: podGlow.emissiveIntensity,
  rightLegAngle: legs[0].leg.rotation.x,
  leftLegAngle: legs[1].leg.rotation.x,
  calls: renderer.info.render.calls,
  triangles: renderer.info.render.triangles
});

// The host decides where the robot appears and when it is drawn; nothing renders while it is stopped.
let host = null;
let resizeObserver = null;
let running = false;
function resize() {
  if (!host) return;
  view.width = Math.max(1, host.clientWidth);
  view.height = Math.max(1, host.clientHeight);
  frameCamera();
  renderer.setPixelRatio(Math.min(devicePixelRatio, COARSE ? 1.5 : 2));
  renderer.setSize(view.width, view.height);
  if (!running) draw();
}
function stop() {
  running = false;
  renderer.setAnimationLoop(null);
}
function start() {
  if (running || !host) return;
  running = true;
  paused = false;
  clock.getDelta();
  renderer.setAnimationLoop(frame);
}
function unmount() {
  stop();
  if (resizeObserver) resizeObserver.disconnect();
  resizeObserver = null;
  renderer.domElement.remove();
  host = null;
}
function mount(container) {
  if (host === container) return api;
  unmount();
  host = container;
  host.appendChild(renderer.domElement);
  resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(host);
  resize();
  return api;
}

const api = {
  mount, unmount, start, stop,
  setState, react: setReaction, setAction, setLevel,
  startLipSync, stopLipSync, speakText,
  onStatus: (fn) => { statusListener = typeof fn === 'function' ? fn : null; },
  seek, play, debugState,
  get state() { return stateRx ? stateRx.type : 'idle'; },
  get canvas() { return renderer.domElement; }
};
sample(0);
export default api;
