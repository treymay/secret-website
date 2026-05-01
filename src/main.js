/**
 * Trey May — Narcissus Mirror activation
 * State machine + WebGL shader mirror + MediaPipe FaceDetector + post-process glitch.
 */
import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { GlitchPass } from 'three/examples/jsm/postprocessing/GlitchPass.js';
import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision';

const STATES = {
  INIT_SCAN: 'INIT_SCAN',
  CAMERA_ACTIVE: 'CAMERA_ACTIVE',
  FALLBACK_VIDEO: 'FALLBACK_VIDEO',
};

/** @typedef { keyof typeof STATES } AppState */

const MEDIAPIPE_WASM_BASE =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const FACE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.task';

const VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 1.0);
}
`;

const FRAG = /* glsl */ `
precision highp float;
varying vec2 vUv;
uniform sampler2D u_texture;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_tilt;
uniform vec2 u_videoSize;

const vec3 LUM_COEF = vec3(0.2126, 0.7152, 0.0722);

vec2 coverUv(vec2 uv) {
  if (u_videoSize.x < 1.0 || u_videoSize.y < 1.0) return uv;
  float vidAR = u_videoSize.x / u_videoSize.y;
  float screenAR = u_resolution.x / max(u_resolution.y, 1.0);
  if (vidAR > screenAR) {
    float s = screenAR / vidAR;
    float off = (1.0 - s) * 0.5;
    return vec2(off + uv.x * s, uv.y);
  }
  float s = vidAR / screenAR;
  float off = (1.0 - s) * 0.5;
  return vec2(uv.x, off + uv.y * s);
}

vec4 sampleBlur(vec2 uvIn) {
  vec2 px = vec2(1.0 / max(u_videoSize.x, 1.0), 1.0 / max(u_videoSize.y, 1.0)) * 14.0;
  vec2 uv = coverUv(uvIn);
  vec4 acc = vec4(0.0);
  // Clean gaussian-style blur (frosted mirror look).
  acc += texture2D(u_texture, uv) * 0.152;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(1.0, 0.0) * px)) * 0.108;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(-1.0, 0.0) * px)) * 0.108;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(0.0, 1.0) * px)) * 0.108;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(0.0, -1.0) * px)) * 0.108;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(1.0, 1.0) * px)) * 0.064;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(-1.0, -1.0) * px)) * 0.064;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(1.0, -1.0) * px)) * 0.064;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(-1.0, 1.0) * px)) * 0.064;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(2.0, 0.0) * px)) * 0.032;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(-2.0, 0.0) * px)) * 0.032;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(0.0, 2.0) * px)) * 0.032;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(0.0, -2.0) * px)) * 0.032;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(2.0, 2.0) * px)) * 0.016;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(-2.0, -2.0) * px)) * 0.016;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(2.0, -2.0) * px)) * 0.016;
  acc += texture2D(u_texture, coverUv(uvIn + vec2(-2.0, 2.0) * px)) * 0.016;
  return acc;
}

void main() {
  vec4 samp = sampleBlur(vUv);
  float lum = dot(samp.rgb, LUM_COEF);
  // High-contrast grayscale base.
  float c = smoothstep(0.06, 0.94, lum);
  c = clamp((c - 0.5) * 1.45 + 0.5, 0.0, 1.0);
  vec3 col = vec3(c);

  // Red only in deep shadows.
  float shadowMask = 1.0 - smoothstep(0.14, 0.42, c);
  col.r += 0.2 * shadowMask;
  col.g -= 0.035 * shadowMask;
  col.b -= 0.035 * shadowMask;

  gl_FragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`;

// —— Dom ————————————————————————————————————————————————————————

const els = {
  initScan: document.getElementById('init-scan'),
  initPrimary: document.getElementById('init-line-primary'),
  tapHint: document.getElementById('tap-hint'),
  mirrorStage: document.getElementById('mirror-stage'),
  webglCanvas: document.getElementById('webgl-canvas'),
  video: document.getElementById('source-video'),
  procCanvas: document.getElementById('procedural-source'),
  acquireForm: document.getElementById('acquire-form'),
  cupidLayer: document.getElementById('cupid-layer'),
  eyeFxLayer: document.getElementById('eye-fx-layer'),
  eyeLeft: document.getElementById('eye-left'),
  eyeRight: document.getElementById('eye-right'),
  archival: document.getElementById('archival-stage'),
};

/** @type {AppState} */
let currentState = STATES.INIT_SCAN;

let audioCtx;
let tiltTarget = new THREE.Vector2(0, 0);
let tiltSmooth = new THREE.Vector2(0, 0);
let faceDetectorPromise = /** @type {Promise<FaceDetector | null>} */ (Promise.resolve(null));
const neckSmooth = { x: 50, y: 42 };

const BASS_URL = '/assets/bass-hit.wav';
const FALLBACK_URL = '/assets/fallback.mp4';
const QR_URL = '/assets/qr.png';

// Kick off network fetch early; decode happens only after user gesture unlocks AudioContext.
const bassArrayBufferPromise = fetch(BASS_URL)
  .then((r) => (r.ok ? r.arrayBuffer() : null))
  .catch(() => null);

// Cache-bust preload for QR + fallback (ensures they’re “included” and warm in cache).
fetch(QR_URL, { cache: 'force-cache' }).catch(() => {});
fetch(FALLBACK_URL, { cache: 'force-cache' }).catch(() => {});

// —— Typed intro ————————————————————————————————————————————————

const PRIMARY_TEXT = '> SYSTEM REQUIRES OPTICAL CALIBRATION';

function typeTerminalLine(el, full, pace = 28) {
  return new Promise((resolve) => {
    let i = 0;
    el.textContent = '';
    function tick() {
      if (i <= full.length) {
        el.textContent = full.slice(0, i);
        i += 1;
        setTimeout(tick, pace);
      } else {
        resolve(undefined);
      }
    }
    tick();
  });
}

(async function bootIntro() {
  await typeTerminalLine(els.initPrimary, PRIMARY_TEXT, 24);
  els.tapHint.hidden = false;
})();

function isProbablySecureContext() {
  // `isSecureContext` is the truth, but Safari sometimes lies behind proxies.
  if (window.isSecureContext) return true;
  const host = window.location.hostname;
  return host === 'localhost' || host === '127.0.0.1';
}

function showInitStatus(line) {
  // Keep it “terminal” and obvious.
  els.initPrimary.textContent = line;
  els.tapHint.hidden = false;
}

/** @type {Promise<void> | null} */
let cameraStartPromise = null;

async function startCameraOnly() {
  if (currentState !== STATES.INIT_SCAN) return;
  if (cameraStartPromise) return cameraStartPromise;
  cameraStartPromise = activateCameraPipeline().catch((e) => {
    cameraStartPromise = null;
    throw e;
  });
  return cameraStartPromise;
}

async function unlockGesturePerks() {
  // Audio + orientation permissions MUST be tied to a user gesture (especially iOS).
  if (!audioCtx) audioCtx = new AudioContext();
  if (audioCtx.state !== 'running') await audioCtx.resume();
  try {
    await playBassHit(audioCtx);
  } catch {
    synthBassSlam(audioCtx);
  }

  await requestOrientationIOS();
  bindOrientation();
}

// Auto-start camera prompt ASAP, but never block tapping.
requestAnimationFrame(() => {
  if (currentState !== STATES.INIT_SCAN) return;
  if (!isProbablySecureContext()) {
    showInitStatus('> HTTPS REQUIRED FOR CAMERA (OR USE localhost)');
    return;
  }
  startCameraOnly().catch(() => {
    // Silent: user can still tap to initiate and/or go to fallback.
  });
});

// —— Bass (Web Audio synth — zero latency on gesture) ——————————————

function synthBassSlam(context) {
  const t0 = context.currentTime;
  const osc = context.createOscillator();
  osc.type = 'square';
  osc.frequency.setValueAtTime(61, t0);
  osc.frequency.exponentialRampToValueAtTime(38, t0 + 0.22);

  const sub = context.createOscillator();
  sub.type = 'sine';
  sub.frequency.setValueAtTime(61, t0);

  const shaper = context.createGain();
  shaper.gain.setValueAtTime(0.22, t0);

  const lp = context.createBiquadFilter();
  lp.type = 'lowpass';
  lp.frequency.setValueAtTime(980, t0);
  lp.frequency.exponentialRampToValueAtTime(120, t0 + 0.35);

  const g = context.createGain();
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(0.42, t0 + 0.02);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.52);

  const dist = context.createWaveShaper();
  const curve = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const x = (i / 127) - 1;
    curve[i] = Math.max(-1, Math.min(1, x * 1.85 + Math.sign(x) * 0.12));
  }
  dist.curve = curve;

  osc.connect(dist);
  sub.connect(dist);
  dist.connect(lp);
  lp.connect(shaper);
  shaper.connect(g);
  g.connect(context.destination);
  osc.start(t0);
  sub.start(t0);
  osc.stop(t0 + 0.55);
  sub.stop(t0 + 0.55);
}

/** @type {AudioBuffer | null} */
let bassBuffer = null;

async function playBassHit(context) {
  const t0 = context.currentTime;
  if (!bassBuffer) {
    const ab = await bassArrayBufferPromise;
    if (!ab) throw new Error('bass-hit.wav unavailable');
    bassBuffer = await context.decodeAudioData(ab.slice(0));
  }
  const src = context.createBufferSource();
  src.buffer = bassBuffer;
  const g = context.createGain();
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(1.0, t0 + 0.015);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + Math.min(1.2, bassBuffer.duration));
  src.connect(g);
  g.connect(context.destination);
  src.start(t0);
  src.stop(t0 + Math.min(1.25, bassBuffer.duration + 0.05));
}

// —— Device orientation ————————————————————————————————————————————

async function requestOrientationIOS() {
  if (typeof DeviceOrientationEvent !== 'undefined' && DeviceOrientationEvent.requestPermission) {
    try {
      const res = await DeviceOrientationEvent.requestPermission();
      return res === 'granted';
    } catch {
      return false;
    }
  }
  return true;
}

function bindOrientation() {
  window.addEventListener(
    'deviceorientation',
    (e) => {
      if (e.beta === null || e.gamma === null) return;
      const nx = THREE.MathUtils.clamp(e.gamma / 45, -1, 1);
      const ny = THREE.MathUtils.clamp((e.beta - 35) / 45, -1, 1);
      tiltTarget.set(nx * 0.7, ny * 0.6);
    },
    false,
  );
}

// —— Procedural fallback texture ————————————————————————————————————

let procTime = 0;

function resizeProcCanvas() {
  const c = els.procCanvas;
  const size = Math.min(512, Math.max(256, window.innerWidth));
  if (c.width !== size || c.height !== size) {
    c.width = size;
    c.height = size;
  }
}

function animateProceduralFrame() {
  const c = els.procCanvas;
  const ctx = c.getContext('2d');
  if (!ctx) return;
  const w = c.width;
  const h = c.height;
  procTime += 0.016;
  ctx.fillStyle = '#0f0303';
  ctx.fillRect(0, 0, w, h);
  const g = ctx.createRadialGradient(w * 0.5, h * 0.35, 0, w * 0.5, h * 0.55, h * 0.9);
  g.addColorStop(0, 'rgba(139,20,22,0.35)');
  g.addColorStop(0.5, 'rgba(40,42,46,0.9)');
  g.addColorStop(1, '#050608');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);

  ctx.globalAlpha = 0.08;
  for (let i = 0; i < 520; i++) {
    const x = (Math.sin(procTime + i * 0.91) * 0.5 + 0.5) * w;
    const y = (Math.cos(procTime * 0.7 + i * 0.71) * 0.5 + 0.5) * h;
    ctx.fillStyle = i % 2 ? '#dcdcdc' : '#6b0004';
    ctx.fillRect(x, y, 1.5 + (i % 3), 1.5);
  }
  ctx.globalAlpha = 1;
}

// —— Three.js mirror ——————————————————————————————————————————————

let renderer;
let composer;
/** @type {GlitchPass | null} */
let glitchPassRef = /** @type {GlitchPass | null} */ (null);
/** @type {THREE.ShaderMaterial | undefined} */
let shaderMat;
/** @type {THREE.Texture | THREE.VideoTexture} */
let mainTexture;
let scene;
let camera;
/** @type {number | undefined} */
let rafId;

let glitchUntil = 0;
let glitching = false;

/** @type {(() => void) | null} */
let boundResizeHandler = null;

function initTHREE() {
  renderer = new THREE.WebGLRenderer({
    canvas: els.webglCanvas,
    antialias: false,
    alpha: false,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  scene = new THREE.Scene();
  camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.z = 1;

  shaderMat = new THREE.ShaderMaterial({
    uniforms: {
      u_texture: { value: null },
      u_time: { value: 0 },
      u_resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
      u_tilt: { value: new THREE.Vector2(0, 0) },
      u_videoSize: { value: new THREE.Vector2(1, 1) },
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
    depthTest: false,
    depthWrite: false,
  });

  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), shaderMat);
  scene.add(quad);

  boundResizeHandler = () => {
    if (!renderer || !shaderMat) return;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    shaderMat.uniforms.u_resolution.value.set(window.innerWidth, window.innerHeight);
    resizeProcCanvas();
    composer?.setSize(window.innerWidth, window.innerHeight);
  };
  window.addEventListener('resize', boundResizeHandler);
}

function attachTexture(tex) {
  if (mainTexture && mainTexture !== tex && mainTexture.dispose) {
    mainTexture.dispose?.();
  }
  mainTexture = tex;
  if (shaderMat) shaderMat.uniforms.u_texture.value = tex;

  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
}

function refreshVideoUniformSize() {
  const v = els.video;
  const w = useProcedural ? els.procCanvas.width : v.videoWidth;
  const h = useProcedural ? els.procCanvas.height : v.videoHeight;
  if (w && h) {
    shaderMat.uniforms.u_videoSize.value.set(w, h);
  }
}

/** @type {boolean} */
let useProcedural = false;

async function bootFaceDetector() {
  try {
    const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_BASE);
    const det = await FaceDetector.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: FACE_MODEL,
      },
      runningMode: 'VIDEO',
      minDetectionConfidence: 0.35,
      minSuppressionThreshold: 0.2,
    });
    return det;
  } catch {
    console.warn('[TreyMay] FaceDetector unavailable — overlay uses default anchor.');
    return null;
  }
}

let syncedFaceDetector = /** @type {FaceDetector | null} */ (null);
let frameCounter = 0;
const targetNeckPct = { x: 50, y: 44 };
const eyeTarget = {
  left: { x: 46, y: 40, size: 30, visible: false },
  right: { x: 54, y: 40, size: 30, visible: false },
};
const eyeSmooth = {
  left: { x: 46, y: 40, size: 30, visible: false },
  right: { x: 54, y: 40, size: 30, visible: false },
};
let eyeLastSeenAt = 0;
const FRONT_CAMERA_MIRROR_X = true;
const lastFaceBox = { x: 0, y: 0, w: 0, h: 0, valid: false };

function toScreenPercentFromVideo(videoX, videoY, vw, vh) {
  const videoU = THREE.MathUtils.clamp(videoX / Math.max(vw, 1), 0, 1);
  const videoV = THREE.MathUtils.clamp(videoY / Math.max(vh, 1), 0, 1);
  const screenW = window.innerWidth;
  const screenH = window.innerHeight;
  const vidAR = vw / Math.max(vh, 1);
  const screenAR = screenW / Math.max(screenH, 1);

  let screenU = videoU;
  let screenV = videoV;
  if (vidAR > screenAR) {
    const s = screenAR / vidAR;
    const off = (1 - s) * 0.5;
    screenU = (videoU - off) / s;
  } else {
    const s = vidAR / screenAR;
    const off = (1 - s) * 0.5;
    screenV = (videoV - off) / s;
  }
  return {
    x: THREE.MathUtils.clamp((FRONT_CAMERA_MIRROR_X ? 1 - screenU : screenU) * 100, 0, 100),
    y: THREE.MathUtils.clamp(screenV * 100, 0, 100),
  };
}

function animateLoop(now) {
  if (!shaderMat || !renderer) {
    rafId = undefined;
    return;
  }
  rafId = requestAnimationFrame(animateLoop);
  shaderMat.uniforms.u_time.value = now * 0.001;

  tiltSmooth.lerp(tiltTarget, 0.08);
  shaderMat.uniforms.u_tilt.value.copy(tiltSmooth);

  if (useProcedural) {
    animateProceduralFrame();
    mainTexture.needsUpdate = true;
  } else if (els.video.readyState >= 2) {
    mainTexture.needsUpdate = true;
  }

  refreshVideoUniformSize();

  if (currentState !== STATES.INIT_SCAN) {
    eyeTarget.left.visible = false;
    eyeTarget.right.visible = false;
    /** @type {FaceDetector | null} */
    const fd = syncedFaceDetector;
    if (fd && !useProcedural) {
      let r;
      try {
        r = fd.detectForVideo(els.video, performance.now());
      } catch {
        r = null;
      }
      if (!r) {
        r = { detections: [] };
      }
      const d = r.detections[0];
      const box = d?.boundingBox;
      if (box && els.video.videoWidth && els.video.videoHeight) {
        const vw = els.video.videoWidth;
        const vh = els.video.videoHeight;
        const boxOriginX = box.width <= 1 ? box.originX * vw : box.originX;
        const boxOriginY = box.height <= 1 ? box.originY * vh : box.originY;
        const boxW = box.width <= 1 ? box.width * vw : box.width;
        const boxH = box.height <= 1 ? box.height * vh : box.height;
        lastFaceBox.x = boxOriginX;
        lastFaceBox.y = boxOriginY;
        lastFaceBox.w = boxW;
        lastFaceBox.h = boxH;
        lastFaceBox.valid = true;
        const centerPt = toScreenPercentFromVideo(boxOriginX + boxW / 2, boxOriginY + boxH / 2, vw, vh);
        const neckPt = toScreenPercentFromVideo(
          boxOriginX + boxW / 2,
          boxOriginY + boxH + boxH * 0.42,
          vw,
          vh,
        );
        const cxPct = centerPt.x;
        const neckPct = neckPt.y;
        targetNeckPct.x = THREE.MathUtils.clamp(cxPct, 20, 80);
        targetNeckPct.y = THREE.MathUtils.clamp(neckPct, 28, 78);
        const eyeSizePx = THREE.MathUtils.clamp(boxW * 0.2, 26, 84);
        const k = d?.keypoints ?? [];
        const kLeft = k[0];
        const kRight = k[1];
        let leftPt;
        let rightPt;
        if (kLeft && kRight) {
          const leftRawX = kLeft.x <= 1 ? kLeft.x * vw : kLeft.x;
          const leftRawY = kLeft.y <= 1 ? kLeft.y * vh : kLeft.y;
          const rightRawX = kRight.x <= 1 ? kRight.x * vw : kRight.x;
          const rightRawY = kRight.y <= 1 ? kRight.y * vh : kRight.y;
          leftPt = toScreenPercentFromVideo(leftRawX, leftRawY, vw, vh);
          rightPt = toScreenPercentFromVideo(rightRawX, rightRawY, vw, vh);
        } else {
          leftPt = toScreenPercentFromVideo(
            boxOriginX + boxW * 0.36,
            boxOriginY + boxH * 0.41,
            vw,
            vh,
          );
          rightPt = toScreenPercentFromVideo(
            boxOriginX + boxW * 0.64,
            boxOriginY + boxH * 0.41,
            vw,
            vh,
          );
        }
        eyeTarget.left.x = leftPt.x;
        eyeTarget.left.y = leftPt.y;
        eyeTarget.right.x = rightPt.x;
        eyeTarget.right.y = rightPt.y;
        eyeTarget.left.size = eyeSizePx;
        eyeTarget.right.size = eyeSizePx;
        eyeTarget.left.visible = true;
        eyeTarget.right.visible = true;
        eyeLastSeenAt = performance.now();
      }

      // If detector misses for a frame, keep using the most recent valid face box.
      if (!box && lastFaceBox.valid && els.video.videoWidth && els.video.videoHeight) {
        const vw = els.video.videoWidth;
        const vh = els.video.videoHeight;
        const eyeSizePx = THREE.MathUtils.clamp(lastFaceBox.w * 0.2, 26, 84);
        const leftPt = toScreenPercentFromVideo(
          lastFaceBox.x + lastFaceBox.w * 0.36,
          lastFaceBox.y + lastFaceBox.h * 0.41,
          vw,
          vh,
        );
        const rightPt = toScreenPercentFromVideo(
          lastFaceBox.x + lastFaceBox.w * 0.64,
          lastFaceBox.y + lastFaceBox.h * 0.41,
          vw,
          vh,
        );
        eyeTarget.left.x = leftPt.x;
        eyeTarget.left.y = leftPt.y;
        eyeTarget.right.x = rightPt.x;
        eyeTarget.right.y = rightPt.y;
        eyeTarget.left.size = eyeSizePx;
        eyeTarget.right.size = eyeSizePx;
        eyeTarget.left.visible = true;
        eyeTarget.right.visible = true;
      }
    }
  }

  neckSmooth.x += (targetNeckPct.x - neckSmooth.x) * 0.16;
  neckSmooth.y += (targetNeckPct.y - neckSmooth.y) * 0.14;
  els.cupidLayer.style.setProperty('--center-x', `${neckSmooth.x}%`);
  els.cupidLayer.style.setProperty('--neck-y', `${neckSmooth.y}%`);

  eyeSmooth.left.x += (eyeTarget.left.x - eyeSmooth.left.x) * 0.35;
  eyeSmooth.left.y += (eyeTarget.left.y - eyeSmooth.left.y) * 0.35;
  eyeSmooth.left.size += ((eyeTarget.left.size ?? 30) - eyeSmooth.left.size) * 0.3;
  eyeSmooth.right.x += (eyeTarget.right.x - eyeSmooth.right.x) * 0.35;
  eyeSmooth.right.y += (eyeTarget.right.y - eyeSmooth.right.y) * 0.35;
  eyeSmooth.right.size += ((eyeTarget.right.size ?? 30) - eyeSmooth.right.size) * 0.3;
  els.eyeLeft.style.setProperty('--eye-x', `${eyeSmooth.left.x}%`);
  els.eyeLeft.style.setProperty('--eye-y', `${eyeSmooth.left.y}%`);
  els.eyeLeft.style.setProperty('--eye-size', `${eyeSmooth.left.size}px`);
  els.eyeRight.style.setProperty('--eye-x', `${eyeSmooth.right.x}%`);
  els.eyeRight.style.setProperty('--eye-y', `${eyeSmooth.right.y}%`);
  els.eyeRight.style.setProperty('--eye-size', `${eyeSmooth.right.size}px`);
  const eyeVisible = performance.now() - eyeLastSeenAt < 1200;
  els.eyeLeft.classList.toggle('active', eyeVisible);
  els.eyeRight.classList.toggle('active', eyeVisible);

  if (glitching && composer && glitchPassRef) {
    glitchPassRef.goWild = performance.now() < glitchUntil - 220;
    composer.render();
    if (performance.now() >= glitchUntil) {
      teardownGlitch();
    }
    return;
  }

  renderer.render(scene, camera);
}

function teardownGlitch() {
  glitching = false;
  if (rafId !== undefined) {
    cancelAnimationFrame(rafId);
    rafId = undefined;
  }

  if (boundResizeHandler) {
    window.removeEventListener('resize', boundResizeHandler);
    boundResizeHandler = null;
  }

  if (composer) {
    glitchPassRef?.dispose();
    composer.dispose();
    composer = /** @type {never} */ (null);
    glitchPassRef = null;
  }

  els.video.pause?.();
  els.video.srcObject?.getTracks()?.forEach((t) => t.stop());
  els.video.srcObject = null;
  els.video.removeAttribute('src');

  els.mirrorStage.classList.add('hidden');
  els.webglCanvas.style.display = 'none';

  scene?.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      obj.geometry.dispose();
      if (obj.material && obj.material !== shaderMat) obj.material.dispose?.();
    }
  });
  if (shaderMat) shaderMat.dispose();
  shaderMat = /** @type {never} */ (null);
  if (mainTexture?.dispose) mainTexture.dispose();
  mainTexture = /** @type {never} */ (null);

  syncedFaceDetector?.close();
  syncedFaceDetector = null;

  if (renderer) {
    renderer.dispose();
    renderer = /** @type {never} */ (null);
  }
  scene = /** @type {never} */ (null);
  camera = /** @type {never} */ (null);

  els.archival.classList.remove('hidden');
}

function startMirrorLoop() {
  if (rafId !== undefined) cancelAnimationFrame(rafId);
  frameCounter = 0;
  rafId = requestAnimationFrame(animateLoop);

  faceDetectorPromise.then((fd) => {
    syncedFaceDetector = fd;
  });
}

// —— Transitions ——————————————————————————————————————————————————

function showMirrorUI() {
  els.initScan.classList.add('hidden');
  els.mirrorStage.classList.remove('hidden');
}

async function activateCameraPipeline() {
  currentState = STATES.CAMERA_ACTIVE;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 720 }, height: { ideal: 1280 } },
    audio: false,
  });

  els.video.srcObject = stream;
  useProcedural = false;
  els.video.removeAttribute('src');
  await els.video.play().catch(() => {});

  if (!renderer) initTHREE();
  attachTexture(new THREE.VideoTexture(els.video));
  showMirrorUI();
  startMirrorLoop();
  faceDetectorPromise = bootFaceDetector();
}

async function activateFallbackPipeline() {
  currentState = STATES.FALLBACK_VIDEO;
  els.video.pause();
  els.video.srcObject?.getTracks().forEach((t) => t.stop());
  els.video.srcObject = null;

  /** @returns {Promise<boolean>} */
  const tryMp4 = () =>
    new Promise((resolve) => {
      const v = els.video;
      const onDone = () => {
        cleanup();
        resolve(true);
      };
      const onFail = () => {
        cleanup();
        resolve(false);
      };
      function cleanup() {
        v.removeEventListener('loadeddata', onDone);
        v.removeEventListener('error', onFail);
      }
      v.addEventListener('loadeddata', onDone);
      v.addEventListener('error', onFail);
      v.loop = true;
      v.muted = true;
      v.playsInline = true;
      v.crossOrigin = 'anonymous';
      v.src = `${window.location.origin}${FALLBACK_URL}`;
      v.load();
    });

  useProcedural = !(await tryMp4());

  if (!renderer) initTHREE();

  if (useProcedural) {
    resizeProcCanvas();
    animateProceduralFrame();
    attachTexture(new THREE.CanvasTexture(els.procCanvas));
  } else {
    await els.video.play().catch(() => {});
    attachTexture(new THREE.VideoTexture(els.video));
  }

  showMirrorUI();
  startMirrorLoop();
  faceDetectorPromise = bootFaceDetector();
}

// —— Init tap ——————————————————————————————————————————————————————

async function handleInitiate() {
  if (currentState !== STATES.INIT_SCAN) return;
  try {
    if (!isProbablySecureContext()) {
      showInitStatus('> HTTPS REQUIRED FOR CAMERA (OR USE localhost)');
      return;
    }

    await unlockGesturePerks();
    await startCameraOnly();
  } catch (err) {
    console.warn('[TreyMay] Camera denied/unavailable — fallback pipeline.', err);
    await activateFallbackPipeline();
  }
}

els.initScan.addEventListener('pointerdown', handleInitiate, { passive: true });
els.initScan.addEventListener('click', handleInitiate);

// —— Form → glitch → archival ——————————————————————————————————————

els.acquireForm.addEventListener('submit', (e) => {
  e.preventDefault();
  if (!shaderMat || !renderer || glitching) return;

  const emailInput = /** @type {HTMLInputElement} */ (document.getElementById('email-input'));
  console.info('[ARCHIVE ENTRY]', emailInput.value);
  localStorage.setItem(
    'treymay_archive_echo',
    JSON.stringify({
      ts: Date.now(),
      email: emailInput.value,
    }),
  );

  glitchUntil = performance.now() + 1500;
  glitching = true;

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  glitchPassRef = new GlitchPass();
  glitchPassRef.goWild = true;
  glitchPassRef.renderToScreen = true;

  composer.addPass(glitchPassRef);
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(window.innerWidth, window.innerHeight);
});
