// Mock satellite imagery SVG components for SatQuery AI prototype

export function SatImageBefore({ width = 320, height = 256 }: { width?: number | string; height?: number | string }) {
  return (
    <svg width={width} height={height} viewBox="0 0 320 256" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block', width: '100%', height: '100%' }}>
      <rect width="320" height="256" fill="#7a6848" />
      <rect x="0" y="0" width="190" height="110" fill="#2d6a4f" />
      <rect x="0" y="108" width="85" height="85" fill="#40916c" />
      <rect x="8" y="180" width="130" height="76" fill="#2d6a4f" />
      <rect x="65" y="138" width="55" height="52" fill="#52b788" />
      <ellipse cx="32" cy="158" rx="26" ry="38" fill="#1a5e8a" />
      <ellipse cx="26" cy="174" rx="17" ry="26" fill="#164e77" />
      <rect x="200" y="8" width="58" height="44" fill="#5a6830" />
      <rect x="200" y="54" width="58" height="42" fill="#72804a" />
      <rect x="200" y="98" width="58" height="44" fill="#5a6830" />
      <rect x="200" y="144" width="58" height="40" fill="#68783e" />
      <rect x="205" y="128" width="12" height="10" fill="#8a8a8a" />
      <rect x="220" y="126" width="10" height="12" fill="#7e7e7e" />
      <rect x="215" y="142" width="9" height="8" fill="#959595" />
      <rect x="228" y="138" width="14" height="10" fill="#8a8a8a" />
      <rect x="242" y="130" width="8" height="9" fill="#7a7a7a" />
      <line x1="160" y1="0" x2="160" y2="256" stroke="#b09872" strokeWidth="3" />
      <line x1="0" y1="128" x2="320" y2="128" stroke="#b09872" strokeWidth="3" />
      <line x1="85" y1="0" x2="85" y2="256" stroke="#c0a882" strokeWidth="1.5" />
      <line x1="0" y1="72" x2="320" y2="72" stroke="#c0a882" strokeWidth="1.5" />
      <line x1="240" y1="0" x2="240" y2="256" stroke="#c0a882" strokeWidth="1.5" />
      {[0, 64, 128, 192, 256, 320].map(x => (
        <line key={`v${x}`} x1={x} y1="0" x2={x} y2="256" stroke="rgba(255,255,255,0.055)" strokeWidth="0.5" />
      ))}
      {[0, 64, 128, 192, 256].map(y => (
        <line key={`h${y}`} x1="0" y1={y} x2="320" y2={y} stroke="rgba(255,255,255,0.055)" strokeWidth="0.5" />
      ))}
      <text x="4" y="9" fill="rgba(255,255,255,0.3)" fontSize="5.5" fontFamily="monospace">28.62°N</text>
      <text x="4" y="249" fill="rgba(255,255,255,0.3)" fontSize="5.5" fontFamily="monospace">28.58°N</text>
      <text x="260" y="9" fill="rgba(255,255,255,0.3)" fontSize="5.5" fontFamily="monospace">77.14°E</text>
      <text x="4" y="22" fill="rgba(255,255,255,0.3)" fontSize="5.5" fontFamily="monospace">77.10°E</text>
      <rect x="6" y="237" width="42" height="13" fill="rgba(0,0,0,0.65)" rx="1" />
      <text x="10" y="246" fill="#7c8fa8" fontSize="6" fontFamily="monospace" fontWeight="500">BEFORE</text>
      <line x1="258" y1="246" x2="306" y2="246" stroke="rgba(255,255,255,0.45)" strokeWidth="1" />
      <line x1="258" y1="243" x2="258" y2="249" stroke="rgba(255,255,255,0.45)" strokeWidth="1" />
      <line x1="306" y1="243" x2="306" y2="249" stroke="rgba(255,255,255,0.45)" strokeWidth="1" />
      <text x="268" y="242" fill="rgba(255,255,255,0.35)" fontSize="5" fontFamily="monospace">500 m</text>
    </svg>
  )
}

export function SatImageAfter({ width = 320, height = 256 }: { width?: number | string; height?: number | string }) {
  return (
    <svg width={width} height={height} viewBox="0 0 320 256" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block', width: '100%', height: '100%' }}>
      <rect width="320" height="256" fill="#7a6848" />
      <rect x="0" y="0" width="170" height="100" fill="#2d6a4f" />
      <rect x="0" y="98" width="72" height="78" fill="#40916c" />
      <rect x="8" y="168" width="96" height="88" fill="#2d6a4f" />
      <ellipse cx="32" cy="155" rx="26" ry="38" fill="#1a5e8a" />
      <ellipse cx="26" cy="170" rx="17" ry="26" fill="#164e77" />
      <rect x="196" y="8" width="58" height="44" fill="#5a6830" />
      <rect x="196" y="54" width="58" height="42" fill="#72804a" />
      {/* New built-up - south */}
      <rect x="108" y="168" width="80" height="88" fill="#707070" />
      <rect x="158" y="140" width="70" height="60" fill="#686868" />
      {/* New built-up - east */}
      <rect x="220" y="124" width="100" height="132" fill="#727272" />
      <rect x="196" y="144" width="40" height="72" fill="#6a6a6a" />
      {/* Building details */}
      <rect x="112" y="172" width="16" height="12" fill="#525252" />
      <rect x="132" y="172" width="14" height="12" fill="#5c5c5c" />
      <rect x="150" y="170" width="18" height="14" fill="#545454" />
      <rect x="112" y="190" width="18" height="13" fill="#5e5e5e" />
      <rect x="162" y="143" width="16" height="12" fill="#4e4e4e" />
      <rect x="182" y="143" width="14" height="12" fill="#585858" />
      <rect x="224" y="127" width="20" height="15" fill="#4a4a4a" />
      <rect x="248" y="127" width="18" height="15" fill="#545454" />
      <rect x="270" y="128" width="22" height="14" fill="#4e4e4e" />
      <rect x="224" y="148" width="24" height="16" fill="#505050" />
      <rect x="254" y="148" width="20" height="16" fill="#5a5a5a" />
      <line x1="160" y1="0" x2="160" y2="256" stroke="#b09872" strokeWidth="3" />
      <line x1="0" y1="128" x2="320" y2="128" stroke="#b09872" strokeWidth="3" />
      <line x1="85" y1="0" x2="85" y2="256" stroke="#c0a882" strokeWidth="1.5" />
      <line x1="0" y1="72" x2="320" y2="72" stroke="#c0a882" strokeWidth="1.5" />
      <line x1="210" y1="128" x2="210" y2="256" stroke="#c0a882" strokeWidth="1.5" />
      {[0, 64, 128, 192, 256, 320].map(x => (
        <line key={`v${x}`} x1={x} y1="0" x2={x} y2="256" stroke="rgba(255,255,255,0.055)" strokeWidth="0.5" />
      ))}
      {[0, 64, 128, 192, 256].map(y => (
        <line key={`h${y}`} x1="0" y1={y} x2="320" y2={y} stroke="rgba(255,255,255,0.055)" strokeWidth="0.5" />
      ))}
      <text x="4" y="9" fill="rgba(255,255,255,0.3)" fontSize="5.5" fontFamily="monospace">28.62°N</text>
      <text x="4" y="249" fill="rgba(255,255,255,0.3)" fontSize="5.5" fontFamily="monospace">28.58°N</text>
      <text x="260" y="9" fill="rgba(255,255,255,0.3)" fontSize="5.5" fontFamily="monospace">77.14°E</text>
      <rect x="6" y="237" width="36" height="13" fill="rgba(0,0,0,0.65)" rx="1" />
      <text x="10" y="246" fill="#7c8fa8" fontSize="6" fontFamily="monospace" fontWeight="500">AFTER</text>
      <line x1="258" y1="246" x2="306" y2="246" stroke="rgba(255,255,255,0.45)" strokeWidth="1" />
      <line x1="258" y1="243" x2="258" y2="249" stroke="rgba(255,255,255,0.45)" strokeWidth="1" />
      <line x1="306" y1="243" x2="306" y2="249" stroke="rgba(255,255,255,0.45)" strokeWidth="1" />
      <text x="268" y="242" fill="rgba(255,255,255,0.35)" fontSize="5" fontFamily="monospace">500 m</text>
    </svg>
  )
}

export function SatImageChangeMask({ width = 320, height = 256 }: { width?: number | string; height?: number | string }) {
  return (
    <svg width={width} height={height} viewBox="0 0 320 256" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block', width: '100%', height: '100%' }}>
      <rect width="320" height="256" fill="#090e1c" />
      <rect x="0" y="0" width="170" height="100" fill="#111f12" />
      <rect x="0" y="98" width="72" height="78" fill="#152316" />
      <rect x="8" y="168" width="96" height="88" fill="#111f12" />
      <ellipse cx="32" cy="155" rx="26" ry="38" fill="#0f1929" />
      {/* Change region - south */}
      <rect x="108" y="168" width="80" height="88" fill="rgba(239,68,68,0.68)" />
      <rect x="158" y="140" width="70" height="60" fill="rgba(239,68,68,0.58)" />
      {/* Change region - east */}
      <rect x="220" y="124" width="100" height="132" fill="rgba(239,68,68,0.62)" />
      <rect x="196" y="144" width="40" height="72" fill="rgba(239,68,68,0.52)" />
      {/* Minor changes */}
      <rect x="172" y="96" width="28" height="22" fill="rgba(234,179,8,0.38)" />
      {/* Yellow dashed boundaries */}
      <rect x="106" y="166" width="124" height="90" fill="none" stroke="#eab308" strokeWidth="1.5" strokeDasharray="4,3" />
      <rect x="194" y="122" width="126" height="134" fill="none" stroke="#eab308" strokeWidth="1.5" strokeDasharray="4,3" />
      <line x1="160" y1="0" x2="160" y2="256" stroke="rgba(176,152,114,0.12)" strokeWidth="2" />
      <line x1="0" y1="128" x2="320" y2="128" stroke="rgba(176,152,114,0.12)" strokeWidth="2" />
      {[0, 64, 128, 192, 256, 320].map(x => (
        <line key={`v${x}`} x1={x} y1="0" x2={x} y2="256" stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
      ))}
      {[0, 64, 128, 192, 256].map(y => (
        <line key={`h${y}`} x1="0" y1={y} x2="320" y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
      ))}
      {/* Legend */}
      <rect x="6" y="5" width="9" height="9" fill="rgba(239,68,68,0.7)" rx="1" />
      <text x="18" y="12" fill="rgba(255,255,255,0.45)" fontSize="6" fontFamily="monospace">Built-up expansion</text>
      <rect x="6" y="18" width="9" height="6" fill="none" stroke="#eab308" strokeWidth="1.2" />
      <text x="18" y="25" fill="rgba(255,255,255,0.45)" fontSize="6" fontFamily="monospace">Change boundary</text>
      <rect x="6" y="237" width="68" height="13" fill="rgba(0,0,0,0.75)" rx="1" />
      <text x="10" y="246" fill="#7c8fa8" fontSize="6" fontFamily="monospace" fontWeight="500">CHANGE MASK</text>
    </svg>
  )
}

export function SatImageSingle({ width = 320, height = 256 }: { width?: number | string; height?: number | string }) {
  return (
    <svg width={width} height={height} viewBox="0 0 320 256" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block', width: '100%', height: '100%' }}>
      <rect width="320" height="256" fill="#7a6848" />
      <rect x="0" y="0" width="140" height="118" fill="#2d6a4f" />
      <rect x="0" y="112" width="108" height="88" fill="#40916c" />
      <rect x="18" y="188" width="88" height="68" fill="#2d6a4f" />
      <ellipse cx="218" cy="88" rx="48" ry="38" fill="#1a5e8a" />
      <ellipse cx="228" cy="102" rx="32" ry="24" fill="#164e77" />
      <rect x="168" y="152" width="66" height="56" fill="#787878" />
      <rect x="230" y="158" width="46" height="50" fill="#6e6e6e" />
      <rect x="174" y="157" width="16" height="12" fill="#555" />
      <rect x="195" y="157" width="14" height="12" fill="#5f5f5f" />
      <rect x="216" y="157" width="16" height="12" fill="#515151" />
      <line x1="140" y1="0" x2="140" y2="256" stroke="#b09872" strokeWidth="2.5" />
      <line x1="0" y1="122" x2="320" y2="122" stroke="#b09872" strokeWidth="2.5" />
      <line x1="0" y1="65" x2="320" y2="65" stroke="#c0a882" strokeWidth="1" />
      <line x1="215" y1="0" x2="215" y2="256" stroke="#c0a882" strokeWidth="1" />
      {/* Overlay labels */}
      <rect x="32" y="42" width="60" height="17" fill="rgba(52,211,153,0.2)" stroke="rgba(52,211,153,0.55)" strokeWidth="1" rx="2" />
      <text x="37" y="53" fill="#34d399" fontSize="6.5" fontFamily="monospace">Vegetation</text>
      <rect x="190" y="74" width="44" height="17" fill="rgba(59,130,246,0.2)" stroke="rgba(96,165,250,0.55)" strokeWidth="1" rx="2" />
      <text x="196" y="85" fill="#60a5fa" fontSize="6.5" fontFamily="monospace">Water</text>
      <rect x="176" y="162" width="48" height="17" fill="rgba(148,163,184,0.2)" stroke="rgba(148,163,184,0.45)" strokeWidth="1" rx="2" />
      <text x="180" y="173" fill="#cbd5e1" fontSize="6.5" fontFamily="monospace">Built-up</text>
      <rect x="128" y="115" width="34" height="17" fill="rgba(251,191,36,0.2)" stroke="rgba(251,191,36,0.45)" strokeWidth="1" rx="2" />
      <text x="132" y="126" fill="#fbbf24" fontSize="6.5" fontFamily="monospace">Roads</text>
      {[0, 64, 128, 192, 256, 320].map(x => (
        <line key={`v${x}`} x1={x} y1="0" x2={x} y2="256" stroke="rgba(255,255,255,0.055)" strokeWidth="0.5" />
      ))}
      {[0, 64, 128, 192, 256].map(y => (
        <line key={`h${y}`} x1="0" y1={y} x2="320" y2={y} stroke="rgba(255,255,255,0.055)" strokeWidth="0.5" />
      ))}
      <rect x="6" y="237" width="60" height="13" fill="rgba(0,0,0,0.65)" rx="1" />
      <text x="10" y="246" fill="#7c8fa8" fontSize="6" fontFamily="monospace" fontWeight="500">SCENE VIEW</text>
    </svg>
  )
}

export function SatImageSAR({ width = 320, height = 256 }: { width?: number | string; height?: number | string }) {
  return (
    <svg width={width} height={height} viewBox="0 0 320 256" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block', width: '100%', height: '100%' }}>
      <rect width="320" height="256" fill="#1e1e1e" />
      <rect x="0" y="0" width="170" height="108" fill="#4a4a4a" />
      <rect x="0" y="106" width="110" height="88" fill="#565656" />
      <rect x="8" y="180" width="96" height="76" fill="#4a4a4a" />
      <ellipse cx="32" cy="155" rx="26" ry="38" fill="#0d0d0d" />
      <rect x="168" y="148" width="152" height="108" fill="#808080" />
      <rect x="196" y="8" width="58" height="80" fill="#626262" />
      <rect x="172" y="100" width="40" height="52" fill="#707070" />
      {Array.from({ length: 100 }, (_, i) => {
        const x = (i * 137.5) % 320
        const y = (i * 89.3) % 256
        return <rect key={i} x={x} y={y} width="1.5" height="1.5" fill={`rgba(255,255,255,${0.03 + (i % 5) * 0.025})`} />
      })}
      <line x1="160" y1="0" x2="160" y2="256" stroke="rgba(176,152,114,0.18)" strokeWidth="1.5" />
      <line x1="0" y1="128" x2="320" y2="128" stroke="rgba(176,152,114,0.18)" strokeWidth="1.5" />
      {[0, 64, 128, 192, 256, 320].map(x => (
        <line key={`v${x}`} x1={x} y1="0" x2={x} y2="256" stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
      ))}
      {[0, 64, 128, 192, 256].map(y => (
        <line key={`h${y}`} x1="0" y1={y} x2="320" y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
      ))}
      <rect x="6" y="237" width="34" height="13" fill="rgba(0,0,0,0.75)" rx="1" />
      <text x="10" y="246" fill="#7c8fa8" fontSize="6" fontFamily="monospace" fontWeight="500">SAR</text>
    </svg>
  )
}

export function SatImageFused({ width = 320, height = 256 }: { width?: number | string; height?: number | string }) {
  return (
    <svg width={width} height={height} viewBox="0 0 320 256" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block', width: '100%', height: '100%' }}>
      <rect width="320" height="256" fill="#0b1020" />
      <rect x="0" y="0" width="170" height="108" fill="#131f15" />
      <rect x="0" y="106" width="110" height="88" fill="#182218" />
      <rect x="8" y="180" width="96" height="76" fill="#131f15" />
      <ellipse cx="32" cy="155" rx="26" ry="38" fill="rgba(26,94,138,0.72)" />
      <ellipse cx="26" cy="170" rx="17" ry="26" fill="rgba(22,78,119,0.8)" />
      <rect x="168" y="148" width="152" height="108" fill="rgba(180,58,38,0.58)" />
      <rect x="196" y="8" width="58" height="80" fill="rgba(155,78,38,0.48)" />
      <rect x="172" y="100" width="40" height="52" fill="rgba(168,68,38,0.52)" />
      {/* Overlay labels */}
      <rect x="10" y="144" width="38" height="17" fill="rgba(26,94,138,0.45)" stroke="rgba(96,165,250,0.6)" strokeWidth="1" rx="2" />
      <text x="14" y="155" fill="#60a5fa" fontSize="6.5" fontFamily="monospace">Water</text>
      <rect x="188" y="168" width="52" height="17" fill="rgba(180,58,38,0.4)" stroke="rgba(248,113,113,0.55)" strokeWidth="1" rx="2" />
      <text x="192" y="179" fill="#f87171" fontSize="6.5" fontFamily="monospace">Built-up</text>
      {[0, 64, 128, 192, 256, 320].map(x => (
        <line key={`v${x}`} x1={x} y1="0" x2={x} y2="256" stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
      ))}
      {[0, 64, 128, 192, 256].map(y => (
        <line key={`h${y}`} x1="0" y1={y} x2="320" y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="0.5" />
      ))}
      <rect x="6" y="237" width="46" height="13" fill="rgba(0,0,0,0.75)" rx="1" />
      <text x="10" y="246" fill="#7c8fa8" fontSize="6" fontFamily="monospace" fontWeight="500">FUSED</text>
    </svg>
  )
}

export function MiniSat({ variant }: { variant: 'scene01' | 'scene02' | 'urban' }) {
  const configs = {
    scene01: { bg: '#7a6848', veg: '#2d6a4f', water: '#1a5e8a', builtup: '#8a8a8a' },
    scene02: { bg: '#7a7055', veg: '#52b788', water: '#1a5e8a', builtup: '#6a6a6a' },
    urban: { bg: '#8a8278', veg: '#3a5a30', water: '#1a4068', builtup: '#6e6e6e' },
  }
  const c = configs[variant]
  return (
    <svg width="60" height="48" viewBox="0 0 60 48" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block', width: '100%', height: '100%' }}>
      <rect width="60" height="48" fill={c.bg} />
      <rect x="0" y="0" width="32" height="24" fill={c.veg} />
      <ellipse cx="46" cy="16" rx="10" ry="8" fill={c.water} />
      <rect x="32" y="28" width="28" height="20" fill={c.builtup} />
      <rect x="4" y="28" width="22" height="20" fill={c.veg} />
      <line x1="30" y1="0" x2="30" y2="48" stroke="rgba(176,152,114,0.4)" strokeWidth="1" />
      <line x1="0" y1="24" x2="60" y2="24" stroke="rgba(176,152,114,0.4)" strokeWidth="1" />
    </svg>
  )
}

export function HomeSatPanel() {
  return (
    <svg width="100%" height="100%" viewBox="0 0 480 340" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block', width: '100%', height: '100%' }}>
      <rect width="480" height="340" fill="#7a6848" />
      <rect x="0" y="0" width="240" height="150" fill="#2d6a4f" />
      <rect x="0" y="148" width="120" height="110" fill="#40916c" />
      <rect x="10" y="246" width="160" height="94" fill="#2d6a4f" />
      <rect x="85" y="180" width="72" height="66" fill="#52b788" />
      <ellipse cx="42" cy="200" rx="34" ry="50" fill="#1a5e8a" />
      <ellipse cx="34" cy="222" rx="22" ry="34" fill="#164e77" />
      <rect x="260" y="10" width="76" height="58" fill="#5a6830" />
      <rect x="260" y="70" width="76" height="56" fill="#72804a" />
      <rect x="260" y="128" width="76" height="58" fill="#5a6830" />
      <rect x="260" y="188" width="76" height="56" fill="#68783e" />
      <rect x="250" y="170" width="16" height="13" fill="#8a8a8a" />
      <rect x="270" y="168" width="13" height="15" fill="#7e7e7e" />
      <rect x="266" y="188" width="12" height="11" fill="#959595" />
      <rect x="282" y="182" width="18" height="13" fill="#8a8a8a" />
      <rect x="352" y="175" width="11" height="12" fill="#7a7a7a" />
      <line x1="210" y1="0" x2="210" y2="340" stroke="#b09872" strokeWidth="3.5" />
      <line x1="0" y1="170" x2="480" y2="170" stroke="#b09872" strokeWidth="3.5" />
      <line x1="110" y1="0" x2="110" y2="340" stroke="#c0a882" strokeWidth="1.5" />
      <line x1="0" y1="92" x2="480" y2="92" stroke="#c0a882" strokeWidth="1.5" />
      <line x1="350" y1="0" x2="350" y2="340" stroke="#c0a882" strokeWidth="1.5" />
      <line x1="0" y1="262" x2="480" y2="262" stroke="#c0a882" strokeWidth="1.5" />
      {[0, 80, 160, 240, 320, 400, 480].map(x => (
        <line key={`v${x}`} x1={x} y1="0" x2={x} y2="340" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
      ))}
      {[0, 68, 136, 204, 272, 340].map(y => (
        <line key={`h${y}`} x1="0" y1={y} x2="480" y2={y} stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
      ))}
      <text x="5" y="11" fill="rgba(255,255,255,0.28)" fontSize="7" fontFamily="monospace">28.62°N  77.10°E</text>
      <text x="5" y="333" fill="rgba(255,255,255,0.28)" fontSize="7" fontFamily="monospace">28.57°N  77.10°E</text>
      <text x="360" y="11" fill="rgba(255,255,255,0.28)" fontSize="7" fontFamily="monospace">77.16°E</text>
      <line x1="380" y1="326" x2="460" y2="326" stroke="rgba(255,255,255,0.4)" strokeWidth="1.2" />
      <line x1="380" y1="322" x2="380" y2="330" stroke="rgba(255,255,255,0.4)" strokeWidth="1.2" />
      <line x1="460" y1="322" x2="460" y2="330" stroke="rgba(255,255,255,0.4)" strokeWidth="1.2" />
      <text x="396" y="321" fill="rgba(255,255,255,0.35)" fontSize="6.5" fontFamily="monospace">1 km</text>
    </svg>
  )
}
