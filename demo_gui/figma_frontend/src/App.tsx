import { useState, useRef, useEffect } from 'react'
import {
  SatImageBefore, SatImageAfter, SatImageChangeMask,
  SatImageSingle, SatImageSAR, SatImageFused,
  MiniSat, HomeSatPanel
} from './SatImages'

type Screen = 'home' | 'upload' | 'validate' | 'query' | 'workflow' | 'results' | 'report'
type AnalysisMode = 'single' | 'bitemporal' | 'sar'

const BG = '#080c18'
const CARD = '#0d1525'
const ELEVATED = '#121c32'
const HOVER = '#162038'
const BORDER = '#1a2840'
const BORDER_STRONG = '#213050'

const modeConfig = {
  single: {
    label: 'Single Image Analysis',
    workflow: 'Single-image VQA + Scene Description',
    answer: 'The image contains agricultural land, a water body in the eastern zone, scattered built-up areas, and a road network running through the center of the scene.',
    confidence: 'High',
    fileLabel: 'Image',
    files: ['scene_before.tif'],
    exampleQuery: 'Describe the land-cover and major objects visible in this image.',
  },
  bitemporal: {
    label: 'Bi-temporal Change Analysis',
    workflow: 'Change-detection model + Change-description model',
    answer: 'Built-up area increased in the southern and eastern portions of the scene. Expansion is most pronounced south of the primary road and along the eastern edge. Vegetation cover in the affected zones has decreased proportionally.',
    confidence: 'High',
    fileLabel: 'Before / After',
    files: ['scene_before.tif', 'scene_after.tif'],
    exampleQuery: 'What changed between these two dates, and where did the change occur?',
  },
  sar: {
    label: 'Optical + SAR Fusion',
    workflow: 'Cross-modal Optical–SAR Analysis',
    answer: 'The fused analysis identifies water-covered regions in the western zone and dense built-up structures in the central-east zone. SAR backscatter confirms surface roughness consistent with urban structures.',
    confidence: 'High',
    fileLabel: 'Optical / SAR',
    files: ['optical.tif', 'sar_scene.tif'],
    exampleQuery: 'Use the optical and SAR images together to identify built-up and water-covered regions.',
  },
}

const exampleQueries: Record<AnalysisMode, string[]> = {
  single: [
    'Describe the land-cover and major objects visible in this image.',
    'What proportion of the scene is built-up vs vegetation?',
    'Identify all water bodies and their approximate locations.',
  ],
  bitemporal: [
    'What changed between these two dates, and where did the change occur?',
    'Identify areas of new construction or urban development.',
    'Describe any changes in vegetation or water coverage.',
  ],
  sar: [
    'Use the optical and SAR images together to identify built-up and water-covered regions.',
    'Identify flooded areas using SAR and confirm with optical imagery.',
    'Detect structures not clearly visible in optical imagery using SAR.',
  ],
}

const DEMO_IMAGES = [
  { id: 'scene01' as const, name: 'Scene 01', filename: 'scene_01.tif', desc: 'Agricultural + water body' },
  { id: 'scene02' as const, name: 'Scene 02', filename: 'scene_02.tif', desc: 'Mixed land-cover' },
  { id: 'urban' as const, name: 'Urban Scene', filename: 'urban_scene.tif', desc: 'Dense built-up + roads' },
]

// ── Shared components ─────────────────────────────────────────────────────────

function Header() {
  return (
    <header style={{ background: CARD, borderBottom: `1px solid ${BORDER}` }}
      className="flex items-center justify-between px-6 py-3 shrink-0">
      <div className="flex items-center gap-3">
        <div>
          <div className="flex items-center gap-2">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <circle cx="9" cy="9" r="8" stroke="#3b82f6" strokeWidth="1.5" />
              <circle cx="9" cy="9" r="4" stroke="#22d3ee" strokeWidth="1" />
              <line x1="9" y1="1" x2="9" y2="4" stroke="#3b82f6" strokeWidth="1.5" />
              <line x1="9" y1="14" x2="9" y2="17" stroke="#3b82f6" strokeWidth="1.5" />
              <line x1="1" y1="9" x2="4" y2="9" stroke="#3b82f6" strokeWidth="1.5" />
              <line x1="14" y1="9" x2="17" y2="9" stroke="#3b82f6" strokeWidth="1.5" />
            </svg>
            <span className="text-slate-100 font-semibold text-sm tracking-tight">SatQuery AI</span>
          </div>
          <div className="text-slate-500 text-xs mt-0.5 font-mono">Satellite Query Assistant</div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
          <span className="text-slate-400 text-xs">System Ready</span>
        </div>
      </div>
    </header>
  )
}

function Progress({ step }: { step: 1 | 2 | 3 | 4 }) {
  const steps = ['01 Upload', '02 Validate', '03 Query', '04 Results']
  return (
    <div className="flex items-center gap-1" style={{ color: '#64748b' }}>
      {steps.map((s, i) => (
        <div key={s} className="flex items-center gap-1">
          <span
            className={`text-xs font-mono ${i + 1 === step ? 'text-blue-400 font-medium' : i + 1 < step ? 'text-slate-500' : 'text-slate-600'}`}
          >{s}</span>
          {i < steps.length - 1 && (
            <span className="text-slate-700 text-xs mx-0.5">→</span>
          )}
        </div>
      ))}
    </div>
  )
}

function PageTitle({ children }: { children: React.ReactNode }) {
  return <h1 className="text-slate-100 font-semibold text-lg">{children}</h1>
}

function Subtitle({ children }: { children: React.ReactNode }) {
  return <p className="text-slate-400 text-sm mt-1">{children}</p>
}

function Btn({
  children, onClick, variant = 'primary', disabled = false, className = ''
}: {
  children: React.ReactNode
  onClick?: () => void
  variant?: 'primary' | 'secondary' | 'ghost'
  disabled?: boolean
  className?: string
}) {
  const base = 'inline-flex items-center gap-2 text-sm font-medium rounded transition-colors cursor-pointer px-4 py-2'
  const styles = {
    primary: disabled
      ? 'bg-blue-800/50 text-blue-400/50 cursor-not-allowed'
      : 'bg-blue-600 hover:bg-blue-500 text-white',
    secondary: `text-slate-300 hover:text-slate-100 hover:bg-white/5 border border-[${BORDER_STRONG}]`,
    ghost: 'text-slate-400 hover:text-slate-200 hover:bg-white/5',
  }
  return (
    <button
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      className={`${base} ${styles[variant]} ${className}`}
      style={variant === 'secondary' ? { border: `1px solid ${BORDER_STRONG}` } : undefined}
    >
      {children}
    </button>
  )
}

function Card({ children, className = '', selected = false, onClick, style: styleProp }: {
  children: React.ReactNode
  className?: string
  selected?: boolean
  onClick?: () => void
  style?: React.CSSProperties
}) {
  const [pressed, setPressed] = useState(false)
  const [hovered, setHovered] = useState(false)

  const bg = pressed
    ? '#1e3050'
    : hovered && onClick
    ? HOVER
    : selected
    ? HOVER
    : CARD

  const borderColor = selected
    ? '#3b82f6'
    : hovered && onClick
    ? '#2d4a70'
    : BORDER

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => { setHovered(false); setPressed(false) }}
      onMouseDown={() => onClick && setPressed(true)}
      onMouseUp={() => setPressed(false)}
      style={{
        background: bg,
        border: `1px solid ${borderColor}`,
        transform: pressed ? 'scale(0.98)' : 'scale(1)',
        transition: 'background 0.12s, border-color 0.12s, transform 0.08s',
        ...styleProp,
      }}
      className={`rounded p-4 ${onClick ? 'cursor-pointer select-none' : ''} ${className}`}
    >
      {children}
    </div>
  )
}

// ── Screen 1: Home ────────────────────────────────────────────────────────────

function ScreenHome({ onStart, onSelectMode, selectedMode }: { onStart: () => void; onSelectMode: (m: AnalysisMode) => void; selectedMode: AnalysisMode | null }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Main two-column — fills height */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 380px', gap: '28px', padding: '32px 40px 20px', minHeight: 0 }}>
        {/* Left */}
        <div className="flex flex-col justify-center gap-4">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <svg width="20" height="20" viewBox="0 0 18 18" fill="none">
                <circle cx="9" cy="9" r="8" stroke="#3b82f6" strokeWidth="1.5" />
                <circle cx="9" cy="9" r="4" stroke="#22d3ee" strokeWidth="1" />
                <line x1="9" y1="1" x2="9" y2="4" stroke="#3b82f6" strokeWidth="1.5" />
                <line x1="9" y1="14" x2="9" y2="17" stroke="#3b82f6" strokeWidth="1.5" />
                <line x1="1" y1="9" x2="4" y2="9" stroke="#3b82f6" strokeWidth="1.5" />
                <line x1="14" y1="9" x2="17" y2="9" stroke="#3b82f6" strokeWidth="1.5" />
              </svg>
              <span className="text-slate-400 text-sm font-mono">SatQuery AI</span>
            </div>
            <h1 className="text-slate-100 font-semibold text-2xl leading-tight mb-2">
              Ask questions.<br />Understand Earth from space.
            </h1>
            <p className="text-slate-400 text-sm leading-relaxed max-w-sm">
              A natural-language interface for exploring satellite imagery through automated task selection and visual evidence.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Btn onClick={onStart}>Start Analysis →</Btn>
          </div>
          {/* Feature list */}
          <div className="flex flex-col gap-1.5">
            {[
              'Upload satellite imagery (GeoTIFF, PNG, JPEG)',
              'Ask questions in plain language',
              'Automatic specialist workflow selection',
              'Visual evidence with change detection',
            ].map(f => (
              <div key={f} className="flex items-center gap-2 text-slate-500 text-xs">
                <span className="text-blue-500 text-xs">─</span>
                {f}
              </div>
            ))}
          </div>
        </div>

        {/* Right: satellite panel — fixed height, compact */}
        <div style={{ background: CARD, border: `1px solid ${BORDER}`, height: '340px', alignSelf: 'center' }} className="rounded overflow-hidden flex flex-col shrink-0">
          {/* Panel header */}
          <div style={{ background: ELEVATED, borderBottom: `1px solid ${BORDER}` }}
            className="flex items-center justify-between px-3 py-2 shrink-0">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
              <span className="text-slate-300 text-xs font-mono">Preview — Bi-temporal Mode</span>
            </div>
            <div className="flex items-center gap-2 text-slate-600 text-xs font-mono">
              <span>WGS84</span>
            </div>
          </div>
          {/* Satellite image */}
          <div className="flex-1 overflow-hidden">
            <HomeSatPanel />
          </div>
          {/* Panel footer */}
          <div style={{ background: ELEVATED, borderTop: `1px solid ${BORDER}` }}
            className="flex items-center justify-between px-3 py-1.5 shrink-0">
            <span className="text-slate-600 text-xs font-mono">Sentinel-2 · 2024</span>
            <span className="text-slate-600 text-xs font-mono">GSD: 10m</span>
          </div>
        </div>
      </div>

      {/* Bottom cards */}
      <div style={{ borderTop: `1px solid ${BORDER}`, padding: '20px 40px' }} className="shrink-0">
        <div className="grid grid-cols-3 gap-4">
          {[
            { icon: '□', label: 'Single Image', mode: 'single' as AnalysisMode, desc: 'Analyse one satellite image and describe land-cover and major objects visible.' },
            { icon: '◫', label: 'Bi-temporal Change', mode: 'bitemporal' as AnalysisMode, desc: 'Compare imagery from two dates to identify and describe changes.' },
            { icon: '◈', label: 'Optical + SAR', mode: 'sar' as AnalysisMode, desc: 'Combine optical and SAR imagery to identify complementary land-cover information.' },
          ].map(c => (
            <Card key={c.label} selected={selectedMode === c.mode} onClick={() => onSelectMode(c.mode)} className="!p-4" style={{ minHeight: '100px' }}>
              <div className="flex items-center gap-2 mb-2.5">
                <span className={`text-lg ${selectedMode === c.mode ? 'text-blue-400' : 'text-slate-500'}`}>{c.icon}</span>
                <span className={`text-sm font-medium ${selectedMode === c.mode ? 'text-slate-100' : 'text-slate-300'}`}>{c.label}</span>
              </div>
              <p className="text-slate-500 text-sm leading-relaxed">{c.desc}</p>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Screen 2: Upload ──────────────────────────────────────────────────────────

function DemoImageModal({
  onClose,
  onSelect,
  slot,
}: {
  onClose: () => void
  onSelect: (filename: string, id: 'scene01' | 'scene02' | 'urban') => void
  slot: string
}) {
  const [sel, setSel] = useState<'scene01' | 'scene02' | 'urban'>('scene01')
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(4,7,16,0.8)' }}>
      <div style={{ background: CARD, border: `1px solid ${BORDER_STRONG}`, width: '360px' }} className="rounded-lg shadow-2xl">
        <div style={{ borderBottom: `1px solid ${BORDER}` }} className="px-5 py-4">
          <div className="text-slate-100 font-semibold text-sm">Select Image</div>
          <div className="text-slate-500 text-xs mt-0.5">
            {slot === 'before' || slot === 'optical' ? 'Before / Optical slot' : 'After / SAR slot'} — Choose an example image.
          </div>
        </div>
        <div className="p-4 flex flex-col gap-2">
          {DEMO_IMAGES.map(img => (
            <div
              key={img.id}
              onClick={() => setSel(img.id)}
              style={{
                background: sel === img.id ? HOVER : ELEVATED,
                border: `1px solid ${sel === img.id ? '#3b82f6' : BORDER}`,
              }}
              className="flex items-center gap-3 p-3 rounded cursor-pointer transition-colors"
            >
              <div style={{ border: `1px solid ${BORDER}` }} className="rounded overflow-hidden shrink-0">
                <MiniSat variant={img.id} />
              </div>
              <div>
                <div className="text-slate-200 text-sm font-medium">{img.name}</div>
                <div className="text-slate-500 text-xs font-mono mt-0.5">{img.filename}</div>
                <div className="text-slate-600 text-xs mt-0.5">{img.desc}</div>
              </div>
              {sel === img.id && (
                <div className="ml-auto w-4 h-4 rounded-full bg-blue-600 flex items-center justify-center shrink-0">
                  <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
                    <path d="M1 4L3 6L7 2" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
        <div style={{ borderTop: `1px solid ${BORDER}` }} className="px-5 py-3 flex justify-end gap-2">
          <Btn variant="ghost" onClick={onClose}>Cancel</Btn>
          <Btn onClick={() => {
            const img = DEMO_IMAGES.find(d => d.id === sel)!
            onSelect(img.filename, sel)
          }}>Use Selected Image</Btn>
        </div>
      </div>
    </div>
  )
}

function UploadSlot({
  label,
  filename,
  imgId,
  onBrowse,
  onRemove,
}: {
  label: string
  filename: string | null
  imgId: 'scene01' | 'scene02' | 'urban' | null
  onBrowse: () => void
  onRemove: () => void
}) {
  if (filename && imgId) {
    return (
      <div style={{ background: ELEVATED, border: `1px solid ${BORDER_STRONG}` }} className="rounded p-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-emerald-400 text-xs">✓</span>
          <span className="text-emerald-400 text-xs font-medium">Uploaded</span>
        </div>
        <div className="flex items-center gap-3">
          <div style={{ border: `1px solid ${BORDER}` }} className="rounded overflow-hidden shrink-0">
            <MiniSat variant={imgId} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="text-slate-200 text-sm font-medium truncate">{filename}</div>
            <div className="text-slate-500 text-xs font-mono mt-0.5">GeoTIFF</div>
            <div className="text-slate-600 text-xs mt-0.5">{label}</div>
          </div>
        </div>
        <div className="flex items-center gap-3 mt-3">
          <button onClick={onBrowse} className="text-blue-400 text-xs hover:text-blue-300 cursor-pointer transition-colors">Replace</button>
          <span className="text-slate-700 text-xs">|</span>
          <button onClick={onRemove} className="text-slate-500 text-xs hover:text-slate-300 cursor-pointer transition-colors">Remove</button>
        </div>
      </div>
    )
  }

  return (
    <div
      onClick={onBrowse}
      style={{ border: `1.5px dashed ${BORDER_STRONG}` }}
      className="rounded p-4 flex flex-col items-center justify-center gap-2 cursor-pointer hover:border-slate-500 transition-colors group"
    >
      <svg width="22" height="22" viewBox="0 0 28 28" fill="none" className="text-slate-600 group-hover:text-slate-500 transition-colors">
        <path d="M14 18V10M14 10L10 14M14 10L18 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M6 22C4.34315 22 3 20.6569 3 19C3 17.5 4 16.2 5.4 16C5.8 13.2 8.1 11 11 11C11.6 11 12.2 11.1 12.8 11.3C13.6 9.3 15.6 8 18 8C21.3 8 24 10.7 24 14C25.7 14.4 27 15.9 27 17.8C27 20.1 25.1 22 22.8 22" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
      <div className="text-center">
        <div className="text-slate-300 text-sm font-medium">{label}</div>
        <div className="text-slate-500 text-xs mt-1">Drag and drop your image here</div>
        <div className="text-slate-600 text-xs">or</div>
        <div className="text-blue-400 text-xs mt-0.5 group-hover:text-blue-300 transition-colors">Browse Files</div>
      </div>
      <div style={{ background: ELEVATED, border: `1px solid ${BORDER}` }} className="text-slate-600 text-xs px-2.5 py-1 rounded font-mono">
        GeoTIFF / TIFF · PNG / JPEG
      </div>
    </div>
  )
}

function ScreenUpload({
  mode, setMode, onContinue,
  beforeFile, setBeforeFile, beforeId, setBeforeId,
  afterFile, setAfterFile, afterId, setAfterId,
  singleFile, setSingleFile, singleId, setSingleId,
}: {
  mode: AnalysisMode
  setMode: (m: AnalysisMode) => void
  onContinue: () => void
  beforeFile: string | null; setBeforeFile: (f: string | null) => void
  beforeId: 'scene01' | 'scene02' | 'urban' | null; setBeforeId: (id: 'scene01' | 'scene02' | 'urban' | null) => void
  afterFile: string | null; setAfterFile: (f: string | null) => void
  afterId: 'scene01' | 'scene02' | 'urban' | null; setAfterId: (id: 'scene01' | 'scene02' | 'urban' | null) => void
  singleFile: string | null; setSingleFile: (f: string | null) => void
  singleId: 'scene01' | 'scene02' | 'urban' | null; setSingleId: (id: 'scene01' | 'scene02' | 'urban' | null) => void
}) {
  const [modal, setModal] = useState<'before' | 'after' | 'single' | null>(null)

  const isReady =
    mode === 'single' ? !!singleFile :
    !!(beforeFile && afterFile)

  const readyCount =
    mode === 'single' ? (singleFile ? 1 : 0) :
    (beforeFile ? 1 : 0) + (afterFile ? 1 : 0)

  const MODES: { id: AnalysisMode; label: string; desc: string; badge?: string }[] = [
    { id: 'single', label: 'Single Image Analysis', desc: 'Analyse one satellite image and describe the land-cover and major objects visible.' },
    { id: 'bitemporal', label: 'Bi-temporal Change Analysis', desc: 'Compare imagery from two dates to identify and describe changes.', badge: 'Recommended' },
    { id: 'sar', label: 'Optical + SAR Fusion', desc: 'Combine optical and SAR imagery to identify complementary land-cover information.' },
  ]

  const slotLabels = {
    single: ['Upload Image'],
    bitemporal: ['Upload Before Image', 'Upload After Image'],
    sar: ['Upload Optical Image', 'Upload SAR Image'],
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Top bar */}
      <div style={{ borderBottom: `1px solid ${BORDER}`, padding: '14px 32px' }}
        className="flex items-center justify-between shrink-0">
        <div>
          <PageTitle>{modeConfig[mode].label}</PageTitle>
          <Subtitle>Configure your inputs before starting the analysis pipeline.</Subtitle>
        </div>
        <Progress step={1} />
      </div>

      {/* Main — sidebar + content */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '300px 1fr', minHeight: 0 }}>
        {/* Left: mode selector */}
        <div style={{ borderRight: `1px solid ${BORDER}`, padding: '24px', overflow: 'hidden', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-2">Analysis Mode</div>
          {MODES.map(m => (
            <div
              key={m.id}
              onClick={() => setMode(m.id)}
              style={{
                background: mode === m.id ? HOVER : ELEVATED,
                border: `1px solid ${mode === m.id ? '#3b82f6' : BORDER}`,
                padding: '10px 12px',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
              className="transition-colors"
            >
              <div className="flex items-start justify-between gap-2 mb-1.5">
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full border shrink-0 flex items-center justify-center ${mode === m.id ? 'border-blue-500 bg-blue-600' : 'border-slate-600'}`}>
                    {mode === m.id && <div className="w-1.5 h-1.5 rounded-full bg-white" />}
                  </div>
                  <span className="text-slate-100 text-sm font-medium">{m.label}</span>
                </div>
              </div>
              <p className="text-slate-500 text-xs leading-relaxed pl-5">{m.desc}</p>
              {m.badge && (
                <div className="pl-5 mt-2">
                  <span style={{ background: 'rgba(59,130,246,0.12)', border: '1px solid rgba(59,130,246,0.28)' }}
                    className="text-blue-400 text-xs px-2 py-0.5 rounded font-mono">{m.badge}</span>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Right: upload area */}
        <div style={{ padding: '24px 32px', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-4">
            {mode === 'single' ? 'Upload Imagery' : 'Upload Imagery — ' + (mode === 'bitemporal' ? '2 images required' : 'Optical + SAR')}
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            {mode === 'single' ? (
              <div style={{ maxWidth: '480px' }}>
                <UploadSlot
                  label="Upload Image"
                  filename={singleFile}
                  imgId={singleId}
                  onBrowse={() => setModal('single')}
                  onRemove={() => { setSingleFile(null); setSingleId(null) }}
                />
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4" style={{ maxWidth: '680px' }}>
                <UploadSlot
                  label={slotLabels[mode][0]}
                  filename={beforeFile}
                  imgId={beforeId}
                  onBrowse={() => setModal('before')}
                  onRemove={() => { setBeforeFile(null); setBeforeId(null) }}
                />
                <UploadSlot
                  label={slotLabels[mode][1]}
                  filename={afterFile}
                  imgId={afterId}
                  onBrowse={() => setModal('after')}
                  onRemove={() => { setAfterFile(null); setAfterId(null) }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom action bar */}
      <div style={{ borderTop: `1px solid ${BORDER}`, padding: '16px 32px' }}
        className="flex items-center justify-between shrink-0">
        <div>
          {isReady ? (
            <div className="flex items-center gap-2 text-emerald-400 text-sm">
              <span>✓</span>
              <span>{readyCount} {readyCount === 1 ? 'image' : 'images'} ready for analysis</span>
            </div>
          ) : (
            <span className="text-slate-600 text-sm font-mono">
              {readyCount}/{mode === 'single' ? 1 : 2} images selected
            </span>
          )}
        </div>
        <Btn disabled={!isReady} onClick={onContinue}>Continue →</Btn>
      </div>

      {/* Modal */}
      {modal && (
        <DemoImageModal
          slot={modal}
          onClose={() => setModal(null)}
          onSelect={(filename, id) => {
            if (modal === 'single') { setSingleFile(filename); setSingleId(id) }
            else if (modal === 'before') { setBeforeFile(filename); setBeforeId(id) }
            else { setAfterFile(filename); setAfterId(id) }
            setModal(null)
          }}
        />
      )}
    </div>
  )
}

// ── Screen 3: Validate ────────────────────────────────────────────────────────

function ScreenValidate({
  mode, beforeFile, afterFile, singleFile, onProceed,
}: {
  mode: AnalysisMode
  beforeFile: string | null
  afterFile: string | null
  singleFile: string | null
  onProceed: () => void
}) {
  const cfg = modeConfig[mode]
  const checks = [
    { label: 'File format', detail: 'Valid' },
    { label: 'Image count', detail: 'Valid' },
    { label: 'Sensor modality', detail: 'Compatible' },
    { label: 'Acquisition dates', detail: 'Available' },
    { label: 'Scene compatibility', detail: 'Compatible' },
    { label: 'Co-registration / alignment', detail: 'Verified where relevant' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Top bar */}
      <div style={{ borderBottom: `1px solid ${BORDER}`, padding: '20px 32px' }}
        className="flex items-center justify-between shrink-0">
        <div>
          <PageTitle>Check Inputs</PageTitle>
          <Subtitle>SatQuery AI checks the selected imagery before starting the analysis.</Subtitle>
        </div>
        <Progress step={2} />
      </div>

      {/* Main — two columns */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 300px', minHeight: 0 }}>
        {/* Left: checklist */}
        <div style={{ padding: '28px 32px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ background: CARD, border: `1px solid ${BORDER}` }} className="rounded p-5">
            <div className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-5">Validation Checks</div>
            <div className="flex flex-col gap-4">
              {checks.map(c => (
                <div key={c.label} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-5 h-5 rounded-full flex items-center justify-center shrink-0"
                      style={{ background: 'rgba(52,211,153,0.1)', border: '1px solid rgba(52,211,153,0.35)' }}>
                      <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
                        <path d="M1.5 4.5L3.5 6.5L7.5 2.5" stroke="#34d399" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                    <span className="text-slate-200 text-sm">{c.label}</span>
                  </div>
                  <span className="text-emerald-400 text-xs font-mono">{c.detail}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Info box */}
          <div style={{ background: 'rgba(59,130,246,0.07)', border: '1px solid rgba(59,130,246,0.2)', borderRadius: '4px', padding: '14px 16px' }}
            className="flex items-start gap-3">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="mt-0.5 shrink-0">
              <circle cx="7" cy="7" r="6" stroke="#3b82f6" strokeWidth="1.2" />
              <line x1="7" y1="6" x2="7" y2="10" stroke="#3b82f6" strokeWidth="1.2" strokeLinecap="round" />
              <circle cx="7" cy="4" r="0.5" fill="#3b82f6" />
            </svg>
            <div>
              <div className="text-blue-300 text-sm font-medium">Validation complete</div>
              <div className="text-slate-400 text-xs mt-0.5">The selected inputs are ready for query-based analysis.</div>
            </div>
          </div>
        </div>

        {/* Right: summary sidebar */}
        <div style={{ borderLeft: `1px solid ${BORDER}`, padding: '28px 24px', overflowY: 'auto' }}>
          <div className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-5">Input Summary</div>
          <div className="flex flex-col gap-4">
            {mode === 'single' ? (
              <Row label="Image" value={singleFile || 'scene_before.tif'} mono />
            ) : (
              <>
                <Row label="Before" value={beforeFile || 'scene_before.tif'} mono />
                <Row label={mode === 'sar' ? 'SAR' : 'After'} value={afterFile || 'scene_after.tif'} mono />
              </>
            )}
            <div style={{ borderTop: `1px solid ${BORDER}`, paddingTop: '16px' }}>
              <Row label="Mode" value={cfg.label} />
            </div>
            <div>
              <div className="text-slate-600 text-xs mb-1.5">Status</div>
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span className="text-emerald-400 text-xs">Inputs validated</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom action bar */}
      <div style={{ borderTop: `1px solid ${BORDER}`, padding: '16px 32px' }}
        className="flex justify-end shrink-0">
        <Btn onClick={onProceed}>Proceed to Query →</Btn>
      </div>
    </div>
  )
}

function Row({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex flex-col gap-0.5 mb-2">
      <span className="text-slate-600 text-xs">{label}</span>
      <span className={`text-slate-300 text-xs ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  )
}

// ── Screen 4: Query ───────────────────────────────────────────────────────────

function ScreenQuery({
  mode, query, setQuery, onRun,
}: {
  mode: AnalysisMode
  query: string
  setQuery: (q: string) => void
  onRun: () => void
}) {
  const examples = exampleQueries[mode]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Top bar */}
      <div style={{ borderBottom: `1px solid ${BORDER}`, padding: '20px 32px' }}
        className="flex items-center justify-between shrink-0">
        <div>
          <PageTitle>Ask a Question</PageTitle>
          <Subtitle>Ask about your imagery in natural language. SatQuery AI will automatically determine the appropriate analysis workflow.</Subtitle>
        </div>
        <Progress step={3} />
      </div>

      {/* Main — textarea left, examples right */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 320px', minHeight: 0 }}>
        {/* Left: query input */}
        <div style={{ padding: '28px 32px', display: 'flex', flexDirection: 'column' }}>
          <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-3">Your Query</div>
          <textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="What would you like to know about these images?"
            style={{
              flex: 1,
              background: CARD,
              border: `1px solid ${BORDER_STRONG}`,
              color: '#e2e8f0',
              resize: 'none',
              width: '100%',
              padding: '16px',
              borderRadius: '4px',
              fontSize: '14px',
              lineHeight: '1.7',
              outline: 'none',
              fontFamily: 'Inter, sans-serif',
            }}
            onFocus={e => { e.target.style.borderColor = '#3b82f6' }}
            onBlur={e => { e.target.style.borderColor = BORDER_STRONG }}
          />
        </div>

        {/* Right: examples */}
        <div style={{ borderLeft: `1px solid ${BORDER}`, padding: '28px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-2">Try an example</div>
          {examples.map(ex => (
            <button
              key={ex}
              onClick={() => setQuery(ex)}
              style={{ background: ELEVATED, border: `1px solid ${BORDER}`, textAlign: 'left', padding: '12px 14px', borderRadius: '4px' }}
              className="text-slate-400 text-xs hover:text-slate-200 hover:border-slate-600 transition-colors cursor-pointer leading-relaxed w-full"
            >
              "{ex}"
            </button>
          ))}
          <div style={{ marginTop: 'auto', paddingTop: '24px', borderTop: `1px solid ${BORDER}` }}>
            <div className="flex items-start gap-2 text-slate-600 text-xs leading-relaxed">
              <svg width="11" height="11" viewBox="0 0 11 11" fill="none" className="mt-0.5 shrink-0">
                <circle cx="5.5" cy="5.5" r="4.5" stroke="#475569" strokeWidth="1" />
                <line x1="5.5" y1="5" x2="5.5" y2="8" stroke="#475569" strokeWidth="1" strokeLinecap="round" />
                <circle cx="5.5" cy="3.5" r="0.4" fill="#475569" />
              </svg>
              You don't need to select a model or tool manually.
            </div>
          </div>
        </div>
      </div>

      {/* Bottom action bar */}
      <div style={{ borderTop: `1px solid ${BORDER}`, padding: '16px 32px' }}
        className="flex justify-end shrink-0">
        <Btn onClick={() => { if (!query.trim()) setQuery(examples[0]); setTimeout(onRun, 0) }}>
          Run Analysis →
        </Btn>
      </div>
    </div>
  )
}

// ── Screen 5: Workflow ────────────────────────────────────────────────────────

function ScreenWorkflow({ mode, onResults }: { mode: AnalysisMode; onResults: () => void }) {
  const cfg = modeConfig[mode]
  const steps = [
    { num: '01', label: 'Interpreting query', desc: 'Understanding the user\'s intent.' },
    { num: '02', label: 'Validating inputs', desc: 'Checking imagery type, dates, and compatibility.' },
    { num: '03', label: 'Selecting task', desc: 'Identifying the required analysis task.' },
    { num: '04', label: 'Selecting specialist tools/models', desc: 'Choosing suitable specialist workflows.' },
    { num: '05', label: 'Generating visual evidence', desc: 'Preparing comparison views and change masks.' },
    { num: '06', label: 'Preparing answer', desc: 'Converting the analysis into a plain-language response.' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '0' }}>
      {/* Top bar */}
      <div style={{ borderBottom: `1px solid ${BORDER}`, padding: '20px 32px' }}
        className="flex items-center justify-between shrink-0">
        <div>
          <PageTitle>Analysis Workflow</PageTitle>
          <Subtitle>SatQuery AI is interpreting your request and selecting the appropriate specialist workflow.</Subtitle>
        </div>
        <Progress step={4} />
      </div>

      {/* Main content — fills remaining height */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 300px', gap: '0', minHeight: 0 }}>
        {/* Steps panel */}
        <div style={{ borderRight: `1px solid ${BORDER}`, padding: '28px 32px', overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
          <div className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-6">Pipeline Execution</div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            {steps.map((s, i) => (
              <div key={s.num} className="flex gap-4 relative" style={{ flex: 1 }}>
                {i < steps.length - 1 && (
                  <div style={{ position: 'absolute', left: '15px', top: '32px', bottom: 0, width: '1px', background: BORDER_STRONG }} />
                )}
                <div className="shrink-0 z-10">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center"
                    style={{ background: 'rgba(52,211,153,0.1)', border: '1px solid rgba(52,211,153,0.32)' }}>
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                      <path d="M2 5L4 7L8 3" stroke="#34d399" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </div>
                </div>
                <div style={{ paddingBottom: i < steps.length - 1 ? '0' : '0', flex: 1 }}>
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-slate-600 text-xs font-mono">{s.num}</span>
                    <span className="text-slate-200 text-sm font-medium">{s.label}</span>
                  </div>
                  <div className="text-slate-500 text-xs leading-relaxed">{s.desc}</div>
                  <div className="flex items-center gap-1.5 mt-1.5">
                    <span className="text-emerald-400 text-xs font-mono">✓ Completed</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right sidebar */}
        <div style={{ padding: '28px 24px', display: 'flex', flexDirection: 'column', gap: '16px', overflowY: 'auto' }}>
          <div style={{ background: CARD, border: `1px solid ${BORDER}` }} className="rounded p-4">
            <div className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-3">Selected Workflow</div>
            <div style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.22)' }}
              className="rounded p-3 mb-3">
              <div className="text-blue-300 text-sm font-medium leading-snug">{cfg.workflow}</div>
            </div>
            <div className="text-slate-600 text-xs mb-1">Selection method</div>
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
              <span className="text-slate-400 text-xs">Workflow selected automatically</span>
            </div>
          </div>

          <div style={{ background: CARD, border: `1px solid ${BORDER}` }} className="rounded p-4">
            <div className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-3">Status</div>
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                <span className="text-slate-300 text-xs">Processing completed</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />
                <span className="text-slate-500 text-xs">Analysis output</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom action bar */}
      <div style={{ borderTop: `1px solid ${BORDER}`, padding: '16px 32px' }}
        className="flex justify-end shrink-0">
        <Btn onClick={onResults}>View Results →</Btn>
      </div>
    </div>
  )
}

// ── Screen 6: Results ─────────────────────────────────────────────────────────

function ScreenResults({
  mode, query,
  onDownload, onAskAnother, onNew,
}: {
  mode: AnalysisMode
  query: string
  onDownload: () => void
  onAskAnother: () => void
  onNew: () => void
}) {
  const [execOpen, setExecOpen] = useState(false)
  const cfg = modeConfig[mode]
  const displayQuery = query.trim() || cfg.exampleQuery

  const execSteps = [
    'Query interpreted', 'Inputs validated', 'Task selected',
    'Specialist workflow selected', 'Visual evidence generated', 'Answer prepared',
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Top bar */}
      <div style={{ borderBottom: `1px solid ${BORDER}`, padding: '16px 32px' }}
        className="flex items-center justify-between shrink-0">
        <div>
          <div className="flex items-center gap-3">
            <PageTitle>Analysis Results</PageTitle>
            <span style={{ background: ELEVATED, border: `1px solid ${BORDER}` }}
              className="text-slate-600 text-xs px-2 py-0.5 rounded font-mono">Analysis output</span>
          </div>
          <div className="text-slate-500 text-xs mt-0.5">{cfg.label}</div>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
          <span className="text-slate-400 text-xs">Processing completed</span>
        </div>
      </div>

      {/* Main: left content + right sidebar */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 280px', minHeight: 0 }}>

        {/* Left — query/answer + evidence filling height */}
        <div style={{ display: 'flex', flexDirection: 'column', borderRight: `1px solid ${BORDER}`, minHeight: 0 }}>
          {/* Query + Answer strip */}
          <div style={{ borderBottom: `1px solid ${BORDER}`, display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
            <div style={{ borderRight: `1px solid ${BORDER}`, padding: '16px 20px' }}>
              <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-2">User Query</div>
              <p className="text-slate-200 text-sm leading-relaxed">"{displayQuery}"</p>
            </div>
            <div style={{ padding: '16px 20px' }}>
              <div className="flex items-center justify-between mb-2">
                <div className="text-slate-500 text-xs font-medium uppercase tracking-wider">AI Answer</div>
                <span style={{ background: 'rgba(52,211,153,0.1)', border: '1px solid rgba(52,211,153,0.3)' }}
                  className="text-emerald-400 text-xs px-2 py-0.5 rounded font-mono">Confidence: {cfg.confidence}</span>
              </div>
              <p className="text-slate-200 text-sm leading-relaxed">{cfg.answer}</p>
            </div>
          </div>

          {/* Visual Evidence — fills remaining height */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div style={{ borderBottom: `1px solid ${BORDER}`, padding: '10px 20px' }}
              className="flex items-center justify-between shrink-0">
              <span className="text-slate-400 text-xs font-medium uppercase tracking-wider">Visual Evidence</span>
              <span className="text-slate-600 text-xs font-mono">Visual output</span>
            </div>
            <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0', minHeight: 0 }}>
              {mode === 'single' && (
                <>
                  <EvidencePanel label="SCENE VIEW"><SatImageSingle width="100%" height="100%" /></EvidencePanel>
                  <EvidencePanel label="LAND COVER" border><SatImageSingle width="100%" height="100%" /></EvidencePanel>
                  <EvidencePanel label="OBJECT MAP" border><SatImageSingle width="100%" height="100%" /></EvidencePanel>
                </>
              )}
              {mode === 'bitemporal' && (
                <>
                  <EvidencePanel label="BEFORE"><SatImageBefore width="100%" height="100%" /></EvidencePanel>
                  <EvidencePanel label="AFTER" border><SatImageAfter width="100%" height="100%" /></EvidencePanel>
                  <EvidencePanel label="DETECTED CHANGE" border><SatImageChangeMask width="100%" height="100%" /></EvidencePanel>
                </>
              )}
              {mode === 'sar' && (
                <>
                  <EvidencePanel label="OPTICAL"><SatImageSingle width="100%" height="100%" /></EvidencePanel>
                  <EvidencePanel label="SAR" border><SatImageSAR width="100%" height="100%" /></EvidencePanel>
                  <EvidencePanel label="FUSED RESULT" border><SatImageFused width="100%" height="100%" /></EvidencePanel>
                </>
              )}
            </div>
            {/* Legend */}
            {mode === 'bitemporal' && (
              <div style={{ borderTop: `1px solid ${BORDER}`, padding: '8px 20px' }}
                className="flex items-center gap-5 shrink-0">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm" style={{ background: 'rgba(239,68,68,0.7)' }} />
                  <span className="text-slate-500 text-xs">Built-up expansion</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm border border-yellow-500" />
                  <span className="text-slate-500 text-xs">Change boundary</span>
                </div>
                <span className="text-slate-700 text-xs ml-auto font-mono">Primary change: southern + eastern zone</span>
              </div>
            )}
            {mode === 'sar' && (
              <div style={{ borderTop: `1px solid ${BORDER}`, padding: '8px 20px' }}
                className="flex items-center gap-5 shrink-0">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm" style={{ background: 'rgba(26,94,138,0.7)' }} />
                  <span className="text-slate-500 text-xs">Water</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-sm" style={{ background: 'rgba(180,58,38,0.7)' }} />
                  <span className="text-slate-500 text-xs">Built-up</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right sidebar */}
        <div style={{ overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
          {/* Input Details */}
          <div style={{ borderBottom: `1px solid ${BORDER}`, padding: '16px 20px' }}>
            <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-3">Input Details</div>
            <div className="flex flex-col gap-2.5">
              {mode === 'single' ? (
                <MetaRow label="Image" value="scene_before.tif" mono />
              ) : (
                <>
                  <MetaRow label="Before" value="scene_before.tif" mono />
                  <MetaRow label={mode === 'sar' ? 'SAR' : 'After'} value="scene_after.tif" mono />
                </>
              )}
              <MetaRow label="Analysis Mode" value={cfg.label} />
              <MetaRow label="Status" value="Processing completed" />
            </div>
          </div>

          {/* Selected Workflow */}
          <div style={{ borderBottom: `1px solid ${BORDER}`, padding: '16px 20px' }}>
            <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-3">Selected Workflow</div>
            <div style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.22)', borderRadius: '4px', padding: '10px 12px', marginBottom: '8px' }}>
              <span className="text-blue-300 text-sm font-medium leading-snug">{cfg.workflow}</span>
            </div>
            <p className="text-slate-500 text-xs leading-relaxed">Automatically selected based on the user's question and available imagery.</p>
          </div>

          {/* Execution Summary */}
          <div style={{ padding: '0' }}>
            <button
              onClick={() => setExecOpen(o => !o)}
              style={{ width: '100%', padding: '14px 20px', borderBottom: execOpen ? `1px solid ${BORDER}` : 'none' }}
              className="flex items-center justify-between cursor-pointer"
            >
              <span className="text-slate-500 text-xs font-medium uppercase tracking-wider">Execution Summary</span>
              <span className="text-slate-500 text-base leading-none">{execOpen ? '−' : '+'}</span>
            </button>
            {execOpen && (
              <div style={{ background: ELEVATED, padding: '12px 20px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {execSteps.map((s, i) => (
                  <div key={s} className="flex items-center gap-2.5">
                    <span className="text-slate-700 text-xs font-mono">{String(i + 1).padStart(2, '0')}</span>
                    <span className="text-emerald-400 text-xs font-mono">✓</span>
                    <span className="text-slate-400 text-xs font-mono">{s}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom action bar */}
      <div style={{ borderTop: `1px solid ${BORDER}`, padding: '14px 32px' }}
        className="flex items-center gap-3 shrink-0">
        <Btn onClick={onDownload}>
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M6.5 1v7M3.5 5l3 3 3-3M1 9.5v1A1.5 1.5 0 002.5 12h8A1.5 1.5 0 0012 10.5v-1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Download Report
        </Btn>
        <Btn variant="secondary" onClick={onAskAnother}>Ask Another Question</Btn>
        <Btn variant="ghost" onClick={onNew}>New Analysis</Btn>
      </div>
    </div>
  )
}

function EvidencePanel({ label, children, border = false }: { label: string; children: React.ReactNode; border?: boolean }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column',
      borderLeft: border ? `1px solid ${BORDER}` : 'none',
    }}>
      <div style={{ borderBottom: `1px solid ${BORDER}`, padding: '8px 12px', background: ELEVATED, shrink: 0 } as React.CSSProperties}
        className="shrink-0">
        <span className="text-slate-500 text-xs font-mono">{label}</span>
      </div>
      <div style={{ flex: 1, overflow: 'hidden', minHeight: 0, lineHeight: 0 }}>
        {children}
      </div>
    </div>
  )
}

function MetaRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <span className="text-slate-600 text-xs shrink-0">{label}</span>
      <span className={`text-slate-300 text-xs text-right ${mono ? 'font-mono' : ''}`}>{value}</span>
    </div>
  )
}

// ── Screen 7: Report ──────────────────────────────────────────────────────────

function ScreenReport({ mode, query, onNew }: { mode: AnalysisMode; query: string; onNew: () => void }) {
  const [notif, setNotif] = useState(false)
  const cfg = modeConfig[mode]
  const displayQuery = query.trim() || cfg.exampleQuery

  const handleDownload = () => {
    setNotif(true)
    setTimeout(() => setNotif(false), 3000)
  }

  return (
    <div style={{ maxWidth: '640px', margin: '0 auto', padding: '48px 24px' }}>
      {/* Download notification */}
      {notif && (
        <div
          style={{ background: ELEVATED, border: `1px solid ${BORDER_STRONG}`, position: 'fixed', top: '72px', right: '24px', zIndex: 100 }}
          className="flex items-center gap-2.5 px-4 py-3 rounded shadow-xl"
        >
          <span className="text-emerald-400 text-sm">✓</span>
          <span className="text-slate-300 text-sm">Report downloaded</span>
        </div>
      )}

      {/* Success state */}
      <div className="flex flex-col items-center text-center mb-8">
        <div className="w-12 h-12 rounded-full bg-emerald-500/12 border border-emerald-500/35 flex items-center justify-center mb-4">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M4 10L8 14L16 6" stroke="#34d399" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <h2 className="text-slate-100 font-semibold text-xl mb-2">Analysis Report Ready</h2>
        <p className="text-slate-400 text-sm max-w-md leading-relaxed">
          The report contains the query, inputs, selected workflow, result, and visual evidence.
        </p>
      </div>

      {/* Report preview */}
      <div style={{ background: CARD, border: `1px solid ${BORDER}` }} className="rounded mb-6">
        <div style={{ background: ELEVATED, borderBottom: `1px solid ${BORDER}` }} className="px-5 py-3 flex items-center justify-between rounded-t">
          <span className="text-slate-300 text-sm font-semibold">SatQuery AI Analysis Report</span>
          <span style={{ background: CARD, border: `1px solid ${BORDER}` }}
            className="text-slate-600 text-xs px-2 py-0.5 rounded font-mono">Analysis output</span>
        </div>
        <div className="p-5 flex flex-col gap-4">
          <ReportSection label="Query" value={displayQuery} />
          <ReportSection label="Inputs" value={
            mode === 'single' ? 'Single satellite image' : 'Before + After satellite imagery'
          } />
          <ReportSection label="Workflow" value={cfg.workflow} />
          <ReportSection label="Result" value={cfg.answer} />
          <div>
            <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">Evidence</div>
            <div className="text-slate-300 text-sm">
              {mode === 'single' && 'Scene View / Land Cover / Object Map'}
              {mode === 'bitemporal' && 'Before / After / Detected Change'}
              {mode === 'sar' && 'Optical / SAR / Fused Result'}
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-center gap-3">
        <Btn onClick={handleDownload}>
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
            <path d="M6.5 1v7M3.5 5l3 3 3-3M1 9.5v1A1.5 1.5 0 002.5 12h8A1.5 1.5 0 0012 10.5v-1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Download Analysis Report
        </Btn>
        <Btn variant="secondary" onClick={onNew}>New Analysis</Btn>
      </div>
    </div>
  )
}

function ReportSection({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">{label}</div>
      <div className="text-slate-300 text-sm leading-relaxed">{value}</div>
    </div>
  )
}

// ── App root ──────────────────────────────────────────────────────────────────

export default function App() {
  const [screen, setScreen] = useState<Screen>('home')
  const [mode, setMode] = useState<AnalysisMode>('bitemporal')
  const [homeSelectedMode, setHomeSelectedMode] = useState<AnalysisMode | null>(null)
  const [query, setQuery] = useState('')

  // Upload state
  const [beforeFile, setBeforeFile] = useState<string | null>(null)
  const [beforeId, setBeforeId] = useState<'scene01' | 'scene02' | 'urban' | null>(null)
  const [afterFile, setAfterFile] = useState<string | null>(null)
  const [afterId, setAfterId] = useState<'scene01' | 'scene02' | 'urban' | null>(null)
  const [singleFile, setSingleFile] = useState<string | null>(null)
  const [singleId, setSingleId] = useState<'scene01' | 'scene02' | 'urban' | null>(null)

  const resetUploads = () => {
    setBeforeFile(null); setBeforeId(null)
    setAfterFile(null); setAfterId(null)
    setSingleFile(null); setSingleId(null)
    setQuery('')
    setHomeSelectedMode(null)
  }

  const go = (s: Screen) => {
    setScreen(s)
    window.scrollTo(0, 0)
  }

  return (
    <div style={{ background: BG, minHeight: '100%', display: 'flex', flexDirection: 'column' }}>
      <Header />
      <div style={{ flex: 1, overflow: ['home','upload','validate','query','workflow','results'].includes(screen) ? 'hidden' : 'auto', display: 'flex', flexDirection: 'column' }}>
        {screen === 'home' && (
          <ScreenHome
            onStart={() => go('upload')}
            selectedMode={homeSelectedMode}
            onSelectMode={(m) => { setHomeSelectedMode(m); setMode(m); setQuery(''); go('upload') }}
          />
        )}
        {screen === 'upload' && (
          <ScreenUpload
            mode={mode} setMode={m => { setMode(m); setQuery('') }}
            onContinue={() => go('validate')}
            beforeFile={beforeFile} setBeforeFile={setBeforeFile}
            beforeId={beforeId} setBeforeId={setBeforeId}
            afterFile={afterFile} setAfterFile={setAfterFile}
            afterId={afterId} setAfterId={setAfterId}
            singleFile={singleFile} setSingleFile={setSingleFile}
            singleId={singleId} setSingleId={setSingleId}
          />
        )}
        {screen === 'validate' && (
          <ScreenValidate
            mode={mode}
            beforeFile={beforeFile}
            afterFile={afterFile}
            singleFile={singleFile}
            onProceed={() => go('query')}
          />
        )}
        {screen === 'query' && (
          <ScreenQuery
            mode={mode}
            query={query}
            setQuery={setQuery}
            onRun={() => go('workflow')}
          />
        )}
        {screen === 'workflow' && (
          <ScreenWorkflow mode={mode} onResults={() => go('results')} />
        )}
        {screen === 'results' && (
          <ScreenResults
            mode={mode}
            query={query}
            onDownload={() => go('report')}
            onAskAnother={() => { setQuery(''); go('query') }}
            onNew={() => { resetUploads(); go('upload') }}
          />
        )}
        {screen === 'report' && (
          <ScreenReport
            mode={mode}
            query={query}
            onNew={() => { resetUploads(); go('upload') }}
          />
        )}
      </div>
    </div>
  )
}
