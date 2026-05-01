# Trey May — Narcissus Mirror (WebGL Activation)

Interactive single-page experience described in `Masterclass_ Interactive WebGL Mobile Experience.md`: a black “optical calibration” entry, synced low-latency bass + camera unlock on tap, liquid-metal WebGL mirror with device tilt and face-anchored Cupid overlay, optional fallback reel, glitch post-processing on signup, then archival typography and Instagram handoff.

## Stack

- [Vite](https://vitejs.dev/) — dev server & build
- [Three.js](https://threejs.org/) — WebGL fullscreen plane, custom GLSL shader, `EffectComposer` + `GlitchPass`
- [@mediapipe/tasks-vision](https://developers.google.com/mediapipe) — BlazeFace (`FaceDetector`) for overlay alignment
- Vanilla JS (no SPA framework)

## Quick start

```bash
npm install
npm run dev
```

Then open the URL Vite prints (typically `http://localhost:5173`).

Production build:

```bash
npm run build
npm run preview
```

Output lives in `dist/`.

## Camera and HTTPS

`getUserMedia` only works in a [secure context](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts) (`https://`, or `localhost` for development). Deploy to HTTPS hosting (Vercel, Netlify, etc.) before QR-code / field use.

Safari/iOS may ask for Device Orientation alongside camera; both are wired to the initial tap gesture per the blueprint.

## Custom assets

| Asset | Location | Notes |
|--------|-----------|-------|
| Cupid overlay | `public/assets/cupid.svg` | Stylized SVG placeholder — replace with a transparent PNG/WebP (`<img>` in `index.html`) for production art. |
| Fallback loop | `public/fallback.mp4` | Optional **muted** MP4 (~10 s loop per spec). If missing or unloadable, a procedural canvas feeds the **same shader** pipeline. |
| Bass hit | Implemented in-browser | Synthetic Web Audio oscillator chain for zero buffering on first tap (replace with `decodeAudioData` + your file if preferred). |

## Asset checklist (add these for production)

Drop these files into `public/` (or adjust paths in `index.html` / `src/main.js`).

- **Cupid overlay art (required for final look)**: `public/assets/cupid.png` (or `.webp`)
  - **Format**: transparent PNG-24 or WebP (alpha)
  - **Suggested size**: ~1200–2000 px tall (crisp edges matter because background is blurred)
  - **Note**: update the `<img src>` in `index.html` to point to your file

- **Fallback muse loop (optional but recommended)**: `public/fallback.mp4`
  - **Length**: ~10 seconds, seamless loop
  - **Codec**: H.264 MP4 (baseline/high), **no audio track** (or muted)
  - **Size target**: under ~3 MB for fast cellular load

- **Bass hit audio (optional upgrade)**: `public/audio/bass-hit.ogg` (or `.m4a`)
  - **Use**: replace the in-browser synth with `fetch` → `decodeAudioData` for your exact sound
  - **Note**: you’ll wire this in `src/main.js` (search for `synthBassSlam`)

- **Sticker / QR print files (for the physical entry point)**:
  - `public/press/qr.png` (high-res QR code)
  - `public/press/sticker-eyes.png` (Cupid eyes crop / sticker art)

## Flow (state machine)

1. **`INIT_SCAN`** — Terminal line `> SYSTEM REQUIRES OPTICAL CALIBRATION`; tap invokes audio + orientation permission (where needed) + `getUserMedia`.
2. **`CAMERA_ACTIVE`** — Mirror: video → `ShaderMaterial` (ripple, multi-tap blur, blood/silver grade), tilt uniforms, mascot overlay smoothed toward face bbox neck estimate.
3. **`FALLBACK_VIDEO`** — If camera denied: try `fallback.mp4`, else procedural texture through the identical shader stack.
4. **Submit `[ > ARM < ]`** — ~1.5 s `GlitchPass`, teardown WebGL, show `> ARCHIVE ENTRY LOGGED. / > PREPARE FOR 0.925.` plus pulsing [Instagram `@TreyMayCo`](https://www.instagram.com/TreyMayCo) link (HTTPS universal link behavior).

Signup data is echoed to `console` and `localStorage` under `treymay_archive_echo`; wire your API or newsletter provider where the form submits in `src/main.js`.

## Project layout

```
index.html           # Structure & fonts
src/main.js          # State machine, WebGL, audio, Mediapipe, glitch
src/styles.css       # Brutalist / frosted overlay styles
public/assets/cupid.svg
```

## Mediapipe

WASM binaries load from jsDelivr; the BlazeFace `.task` model loads from Google’s bucket. Offline or locked-down networks may skip face alignment (overlay falls back toward default CSS variables).

## License / brand

Implements an internal creative-architecture brief for **Trey May**; mascot and marketing rights remain with you.
