/* ==========================================================================
   SATQUERY AI — APPLICATION LOGIC (HUMAN-ENGINEERED GIS WORKSTATION)
   Vanilla JavaScript for State Engine, GIS Vector Satellite Viewports,
   Pipeline Orchestration, and Report Generation.
   ========================================================================== */

(function () {
  'use strict';

  // -------------------------------------------------------------------------
  // GOLDEN SCENE (Part 2: real backend imagery, single-image flow only)
  // -------------------------------------------------------------------------
  // Hardcoded for now purely to prove the image pipe works end-to-end.
  // Part 3 will feed the scene_id from the actual analyze request.
  const GOLDEN_SCENE_ID = 'loveda_LoveDA_images_png_0_gsd0.3';

  // -------------------------------------------------------------------------
  // GLOBAL STATE
  // -------------------------------------------------------------------------
  const state = {
    currentMode: 'bitemporal', // 'single' | 'bitemporal' | 'optical_sar'
    currentScreen: 'screen-home',
    queryText: 'What changed between these two dates, and where did the change occur?',
    accordionOpen: false
  };

  // -------------------------------------------------------------------------
  // DEMO DATA CONFIGURATIONS FOR THE THREE FLOWS
  // -------------------------------------------------------------------------
  const flowConfig = {
    single: {
      modeName: 'Single Image Analysis',
      sceneId: 'SAT-2026-SGL-014',
      modalityTag: 'OPTICAL MULTISPECTRAL',
      files: [
        { name: 'satellite_scene_demo.tif', tags: ['GeoTIFF', 'Optical'] }
      ],
      valImageCount: '1 image loaded for single-scene evaluation.',
      valModality: 'High-resolution optical multispectral satellite sensor.',
      defaultQuery: 'Describe the land-cover and major objects visible in this image.',
      workflowName: 'Single-image VQA + scene description',
      workflowDetail: 'Selected specialist workflow: Scene-classification model + object-detection model',
      aiAnswer: 'The image contains agricultural land, a water body, scattered built-up areas, and a road network.',
      metaInputs: 'Single Optical imagery',
      metaWorkflow: 'Single-image VQA + scene description',
      metaTools: 'Scene-classification model + object-detection model',
      audit: [
        { title: 'Query interpreted', desc: 'Extracted intent: Scene description & VQA' },
        { title: 'Inputs validated', desc: 'GeoTIFF single image verification passed' },
        { title: 'Task selected', desc: 'Single-image scene classification & object detection' },
        { title: 'Specialist workflow selected', desc: 'Scene-classification model + object-detection model' },
        { title: 'Visual evidence prepared', desc: 'Land-cover overlay & object bounding boxes rendered' },
        { title: 'Answer generated', desc: 'Synthesized natural-language description response' }
      ],
      legendItems: [
        { colorClass: 'color-water', label: 'Water Body' },
        { colorClass: 'color-veg', label: 'Agricultural Vegetation' },
        { colorClass: 'color-build', label: 'Built-up Areas' },
        { colorClass: 'color-road', label: 'Road Network' }
      ]
    },

    bitemporal: {
      modeName: 'Bi-temporal Change Analysis',
      sceneId: 'SAT-2026-CHG-089',
      modalityTag: 'BI-TEMPORAL OPTICAL PAIR',
      files: [
        { name: 'before_scene_demo.tif', tags: ['GeoTIFF', 'Optical', 'Before'] },
        { name: 'after_scene_demo.tif', tags: ['GeoTIFF', 'Optical', 'After'] }
      ],
      valImageCount: '2 images loaded for dual-timestamp comparative evaluation.',
      valModality: 'High-resolution multispectral optical satellite sensor.',
      defaultQuery: 'What changed between these two dates, and where did the change occur?',
      workflowName: 'Bi-temporal change VQA',
      workflowDetail: 'Selected specialist workflow: Change-detection model + change-description model',
      aiAnswer: 'Built-up area increased in the southern and eastern portions of the scene.',
      metaInputs: 'Before + After imagery',
      metaWorkflow: 'Bi-temporal change VQA',
      metaTools: 'Change-detection model + change-description model',
      audit: [
        { title: 'Query interpreted', desc: 'Extracted intent: Bi-temporal change query' },
        { title: 'Inputs validated', desc: 'GeoTIFF pair co-registration & timestamp check passed' },
        { title: 'Change-analysis task selected', desc: 'Spatial change detection & change VQA' },
        { title: 'Specialist workflow selected', desc: 'Change-detection model + change-description model' },
        { title: 'Visual evidence prepared', desc: 'Change mask & highlight overlays rendered' },
        { title: 'Answer generated', desc: 'Synthesized natural-language change response' }
      ],
      legendItems: [
        { colorClass: 'color-red', label: 'Red = Change area' },
        { colorClass: 'color-yellow', label: 'Yellow = Highlighted change area' }
      ]
    },

    optical_sar: {
      modeName: 'Optical + SAR Fusion',
      sceneId: 'SAT-2026-FUS-042',
      modalityTag: 'OPTICAL + C-BAND SAR FUSION',
      files: [
        { name: 'optical_scene_demo.tif', tags: ['Optical', 'GeoTIFF'] },
        { name: 'sar_scene_demo.tif', tags: ['SAR', 'GeoTIFF'] }
      ],
      valImageCount: '2 images loaded for complementary multi-modal fusion.',
      valModality: 'Optical Multispectral + Synthetic Aperture Radar (SAR).',
      defaultQuery: 'Use the optical and SAR images together to identify built-up and water-covered regions.',
      workflowName: 'Cross-modal optical–SAR analysis',
      workflowDetail: 'Selected specialist workflow: Optical-SAR feature fusion + semantic segmentation model',
      aiAnswer: 'The fused analysis identifies water-covered regions in the western zone and dense built-up structures in the central-east zone.',
      metaInputs: 'Optical + SAR imagery',
      metaWorkflow: 'Cross-modal optical–SAR analysis',
      metaTools: 'Optical-SAR feature fusion + semantic segmentation',
      audit: [
        { title: 'Query interpreted', desc: 'Extracted intent: Multi-modal land-cover & structure identification' },
        { title: 'Inputs validated', desc: 'Optical & SAR co-registration verified' },
        { title: 'Task selected', desc: 'Cross-modal feature extraction & classification' },
        { title: 'Specialist workflow selected', desc: 'Optical-SAR feature fusion + semantic segmentation model' },
        { title: 'Visual evidence prepared', desc: 'Fused classification map & specular overlays generated' },
        { title: 'Answer generated', desc: 'Synthesized multi-modal fused description response' }
      ],
      legendItems: [
        { colorClass: 'color-water', label: 'Water-covered (Western zone)' },
        { colorClass: 'color-yellow', label: 'Dense built-up structures (Central-east zone)' }
      ]
    }
  };

  // -------------------------------------------------------------------------
  // DOM ELEMENT REFERENCES
  // -------------------------------------------------------------------------
  const dom = {
    navItems: document.querySelectorAll('.nav-item'),
    screens: document.querySelectorAll('.screen'),
    sidebar: document.getElementById('sidebar'),
    sidebarOverlay: document.getElementById('sidebarOverlay'),
    menuToggleBtn: document.getElementById('menuToggleBtn'),
    currentModeLabel: document.getElementById('currentModeLabel'),
    apiStatus: document.getElementById('apiStatus'),
    apiDot: document.getElementById('apiDot'),
    apiBranding: document.getElementById('apiBranding'),

    // Screen 1
    btnHomeStart: document.getElementById('btnHomeStart'),
    featureCards: document.querySelectorAll('.feature-card'),

    // Screen 2
    modeCards: document.querySelectorAll('.mode-card'),
    demoFileList: document.getElementById('demoFileList'),
    btnUploadBack: document.getElementById('btnUploadBack'),
    btnUploadContinue: document.getElementById('btnUploadContinue'),

    // Screen 3
    valImageCountText: document.getElementById('valImageCountText'),
    valModalityText: document.getElementById('valModalityText'),
    btnValidationBack: document.getElementById('btnValidationBack'),
    btnValidationProceed: document.getElementById('btnValidationProceed'),

    // Screen 4
    queryTextarea: document.getElementById('queryTextarea'),
    queryModeTag: document.getElementById('queryModeTag'),
    btnClearQuery: document.getElementById('btnClearQuery'),
    promptChips: document.querySelectorAll('.prompt-chip'),
    btnQueryBack: document.getElementById('btnQueryBack'),
    btnQueryRun: document.getElementById('btnQueryRun'),

    // Screen 5
    analysisWorkflowName: document.getElementById('analysisWorkflowName'),
    analysisWorkflowDetails: document.getElementById('analysisWorkflowDetails'),
    step3Detail: document.getElementById('step3Detail'),
    step4Detail: document.getElementById('step4Detail'),
    btnAnalysisBack: document.getElementById('btnAnalysisBack'),
    btnAnalysisViewResults: document.getElementById('btnAnalysisViewResults'),

    // Screen 6
    resSceneId: document.getElementById('resSceneId'),
    resModalityTag: document.getElementById('resModalityTag'),
    resultUserQuery: document.getElementById('resultUserQuery'),
    resultAiAnswer: document.getElementById('resultAiAnswer'),
    metaInputs: document.getElementById('metaInputs'),
    metaWorkflow: document.getElementById('metaWorkflow'),
    metaTools: document.getElementById('metaTools'),
    visualEvidenceGrid: document.getElementById('visualEvidenceGrid'),
    visualLegendBar: document.getElementById('visualLegendBar'),
    accordionToggle: document.getElementById('accordionToggle'),
    accordionChevron: document.getElementById('accordionChevron'),
    accordionContent: document.getElementById('accordionContent'),
    auditFlowTree: document.getElementById('auditFlowTree'),
    btnResultsNewAnalysis: document.getElementById('btnResultsNewAnalysis'),
    btnResultsAskAnother: document.getElementById('btnResultsAskAnother'),
    btnResultsDownload: document.getElementById('btnResultsDownload'),

    // Screen 7
    reportPreviewContent: document.getElementById('reportPreviewContent'),
    btnReportNewAnalysis: document.getElementById('btnReportNewAnalysis'),
    btnReportDownload: document.getElementById('btnReportDownload')
  };

  // -------------------------------------------------------------------------
  // INITIALIZATION
  // -------------------------------------------------------------------------
  function init() {
    setupEventListeners();
    setMode('bitemporal'); // Default primary flow
    navigateTo('screen-home');
    checkApiHealth();
    if (typeof lucide !== 'undefined') {
      lucide.createIcons();
    }
  }

  async function checkApiHealth() {
    if (!dom.apiStatus || !dom.apiDot) return;
    dom.apiStatus.textContent = 'CONNECTING';
    dom.apiDot.className = 'indicator-dot connecting';
    if (dom.apiBranding) dom.apiBranding.textContent = 'SIH26167 · CONNECTING';
    try {
      const health = await window.SatQueryApi.getHealth();
      if (health.status === 'ready') {
        dom.apiStatus.textContent = 'ONLINE';
        dom.apiDot.className = 'indicator-dot online';
        if (dom.apiBranding) dom.apiBranding.textContent = 'SIH26167 · LIVE API CONNECTED';
      } else {
        dom.apiStatus.textContent = 'OFFLINE';
        dom.apiDot.className = 'indicator-dot offline';
        if (dom.apiBranding) dom.apiBranding.textContent = 'SIH26167 · API UNAVAILABLE';
      }
    } catch {
      dom.apiStatus.textContent = 'OFFLINE';
      dom.apiDot.className = 'indicator-dot offline';
      if (dom.apiBranding) dom.apiBranding.textContent = 'SIH26167 · API UNAVAILABLE';
    }
  }

  // -------------------------------------------------------------------------
  // NAVIGATION CONTROLLER
  // -------------------------------------------------------------------------
  function navigateTo(screenId) {
    state.currentScreen = screenId;

    // Update screen visibility
    dom.screens.forEach(screen => {
      if (screen.id === screenId) {
        screen.classList.add('active');
      } else {
        screen.classList.remove('active');
      }
    });

    // Update sidebar navigation active highlight
    dom.navItems.forEach(item => {
      if (item.getAttribute('data-screen') === screenId) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    // Scroll main panel to top
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Close mobile menu if open
    closeMobileMenu();

    // Trigger screen specific refresh
    if (screenId === 'screen-results') {
      renderVisualEvidence();
      renderAuditTrail();
    } else if (screenId === 'screen-report') {
      renderReportPreview();
    }
  }

  function setMode(modeKey) {
    if (!flowConfig[modeKey]) return;
    state.currentMode = modeKey;
    const config = flowConfig[modeKey];

    // Update Mode selector cards on Screen 2
    dom.modeCards.forEach(card => {
      if (card.getAttribute('data-mode') === modeKey) {
        card.classList.add('selected');
      } else {
        card.classList.remove('selected');
      }
    });

    // Update Sidebar mode badge
    dom.currentModeLabel.textContent = `Mode: ${config.modeName}`;

    // Update Upload Screen File List
    renderUploadFiles(config.files);

    // Update Input Validation Screen Texts
    dom.valImageCountText.textContent = config.valImageCount;
    dom.valModalityText.textContent = config.valModality;

    // Update Ask Query Screen
    dom.queryTextarea.value = config.defaultQuery;
    state.queryText = config.defaultQuery;
    dom.queryModeTag.textContent = `Mode: ${config.modeName}`;

    // Update prompt chips active state
    dom.promptChips.forEach(chip => {
      if (chip.getAttribute('data-prompt') === config.defaultQuery) {
        chip.classList.add('active');
      } else {
        chip.classList.remove('active');
      }
    });

    // Update Agentic Analysis Screen
    dom.analysisWorkflowName.textContent = config.workflowName;
    dom.analysisWorkflowDetails.textContent = config.workflowDetail;
    dom.step3Detail.textContent = `Task selected: ${config.workflowName}`;
    dom.step4Detail.textContent = `Tool chain: ${config.metaTools}`;

    // Update Results Screen Telemetry & Content
    dom.resSceneId.textContent = config.sceneId;
    dom.resModalityTag.textContent = config.modalityTag;
    dom.resultUserQuery.textContent = `“${state.queryText}”`;
    dom.resultAiAnswer.textContent = `“${config.aiAnswer}”`;
    dom.metaInputs.textContent = config.metaInputs;
    dom.metaWorkflow.textContent = config.metaWorkflow;
    dom.metaTools.textContent = config.metaTools;
  }

  // -------------------------------------------------------------------------
  // RENDER HELPERS
  // -------------------------------------------------------------------------
  function renderUploadFiles(files) {
    dom.demoFileList.innerHTML = '';
    files.forEach(file => {
      const item = document.createElement('div');
      item.className = 'demo-file-item';
      
      const tagsHtml = file.tags.map(t => {
        let tagClass = 'file-tag';
        if (t === 'Before' || t === 'Optical') tagClass += ' tag-cyan';
        if (t === 'After' || t === 'SAR') tagClass += ' tag-yellow';
        return `<span class="${tagClass}">${t}</span>`;
      }).join('');

      item.innerHTML = `
        <div class="file-info">
          <div class="file-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
            </svg>
          </div>
          <span class="file-name">${file.name}</span>
        </div>
        <div class="file-tags">
          ${tagsHtml}
        </div>
      `;
      dom.demoFileList.appendChild(item);
    });
  }

  function renderVisualEvidence() {
    const mode = state.currentMode;
    const config = flowConfig[mode];
    dom.visualEvidenceGrid.innerHTML = '';

    // Render legend
    dom.visualLegendBar.innerHTML = config.legendItems.map(item => `
      <div class="legend-item">
        <span class="legend-color ${item.colorClass}"></span>
        <span>${item.label}</span>
      </div>
    `).join('');

    if (mode === 'bitemporal') {
      // FLOW B: 3 Panels — BEFORE, AFTER, CHANGE MASK (MAIN SHOWCASE)
      dom.visualEvidenceGrid.appendChild(createPanelElement('1. BEFORE (T1)', 'TIMESTAMP: T1 (BASELINE)', generateSatelliteSvg('before')));
      dom.visualEvidenceGrid.appendChild(createPanelElement('2. AFTER (T2)', 'TIMESTAMP: T2 (UPDATED)', generateSatelliteSvg('after')));
      dom.visualEvidenceGrid.appendChild(createPanelElement('3. CHANGE MASK', 'RASTER MASK · SOUTH & EAST', generateSatelliteSvg('changemask')));
    } else if (mode === 'single') {
      // FLOW A: 1 Panel — REAL BACKEND SCENE IMAGERY (golden scene).
      // Pulls the actual PNG from GET /api/scenes/{scene_id}/image via the
      // API client. No SVG fallback: if pixels are missing we surface an
      // explicit "unavailable" state instead of fabricating a viewport.
      dom.visualEvidenceGrid.appendChild(
        createSceneImagePanelElement('SINGLE SCENE · SOURCE IMAGERY', 'BACKEND PNG · /api/scenes', GOLDEN_SCENE_ID)
      );
    } else if (mode === 'optical_sar') {
      // FLOW C: 3 Panels — OPTICAL, SAR, FUSED RESULT
      dom.visualEvidenceGrid.appendChild(createPanelElement('1. OPTICAL MODALITY', 'VNIR/SWIR BANDS', generateSatelliteSvg('optical')));
      dom.visualEvidenceGrid.appendChild(createPanelElement('2. SAR MODALITY', 'C-BAND VV BACKSCATTER', generateSatelliteSvg('sar')));
      dom.visualEvidenceGrid.appendChild(createPanelElement('3. FUSED RESULT', 'CROSS-MODAL SEGMENTATION', generateSatelliteSvg('fused')));
    }
  }

  function createPanelElement(title, subtag, svgContent) {
    const panel = document.createElement('div');
    panel.className = 'evidence-panel';
    panel.innerHTML = `
      <div class="panel-header">
        <span>${title}</span>
        <span class="panel-subtag">${subtag}</span>
      </div>
      <div class="panel-body">
        ${svgContent}
      </div>
    `;
    return panel;
  }

  // Builds an evidence panel backed by a REAL <img> pointed at the backend
  // scene endpoint. On load failure (404, network, missing pixels) the body
  // is replaced with an explicit unavailable state — never the generated SVG,
  // never a silent broken-image icon, never a fabricated substitute.
  function createSceneImagePanelElement(title, subtag, sceneId) {
    const panel = document.createElement('div');
    panel.className = 'evidence-panel';

    const header = document.createElement('div');
    header.className = 'panel-header';
    const titleSpan = document.createElement('span');
    titleSpan.textContent = title;
    const subtagSpan = document.createElement('span');
    subtagSpan.className = 'panel-subtag';
    subtagSpan.textContent = subtag;
    header.appendChild(titleSpan);
    header.appendChild(subtagSpan);

    const body = document.createElement('div');
    body.className = 'panel-body';

    function showUnavailable() {
      body.innerHTML = '';
      const state = document.createElement('div');
      state.className = 'panel-unavailable';
      const heading = document.createElement('div');
      heading.className = 'panel-unavailable-title';
      heading.textContent = 'Scene imagery unavailable';
      const detail = document.createElement('div');
      detail.className = 'panel-unavailable-detail';
      detail.textContent = `No pixels returned for ${sceneId}.`;
      state.appendChild(heading);
      state.appendChild(detail);
      body.appendChild(state);
    }

    let imageUrl;
    try {
      imageUrl = window.SatQueryApi.getSceneImageUrl(sceneId);
    } catch {
      showUnavailable();
      panel.appendChild(header);
      panel.appendChild(body);
      return panel;
    }

    const img = document.createElement('img');
    img.className = 'panel-scene-image';
    img.alt = `Source satellite scene ${sceneId}`;
    img.addEventListener('error', showUnavailable);
    img.src = imageUrl;
    body.appendChild(img);

    panel.appendChild(header);
    panel.appendChild(body);
    return panel;
  }

  // HIGH-PRECISION GIS SVG GRAPHICS GENERATOR (ENGINEERED VIEWPORTS)
  function generateSatelliteSvg(type) {
    const gisOverlay = `
      <!-- GIS Grid Lines -->
      <line x1="0" y1="50" x2="400" y2="50" stroke="rgba(56,189,248,0.15)" stroke-dasharray="3 3" stroke-width="1"/>
      <line x1="0" y1="150" x2="400" y2="150" stroke="rgba(56,189,248,0.15)" stroke-dasharray="3 3" stroke-width="1"/>
      <line x1="0" y1="250" x2="400" y2="250" stroke="rgba(56,189,248,0.15)" stroke-dasharray="3 3" stroke-width="1"/>
      <line x1="100" y1="0" x2="100" y2="300" stroke="rgba(56,189,248,0.15)" stroke-dasharray="3 3" stroke-width="1"/>
      <line x1="200" y1="0" x2="200" y2="300" stroke="rgba(56,189,248,0.15)" stroke-dasharray="3 3" stroke-width="1"/>
      <line x1="300" y1="0" x2="300" y2="300" stroke="rgba(56,189,248,0.15)" stroke-dasharray="3 3" stroke-width="1"/>
      
      <!-- Coordinate Labels (WGS84 EPSG:4326) -->
      <text x="5" y="45" fill="rgba(255,255,255,0.4)" font-size="7" font-family="monospace">18.54°N</text>
      <text x="5" y="145" fill="rgba(255,255,255,0.4)" font-size="7" font-family="monospace">18.52°N</text>
      <text x="5" y="245" fill="rgba(255,255,255,0.4)" font-size="7" font-family="monospace">18.50°N</text>
      <text x="105" y="295" fill="rgba(255,255,255,0.4)" font-size="7" font-family="monospace">73.85°E</text>
      <text x="205" y="295" fill="rgba(255,255,255,0.4)" font-size="7" font-family="monospace">73.88°E</text>

      <!-- Center Target Reticle -->
      <circle cx="200" cy="150" r="5" fill="none" stroke="#0284C7" stroke-width="1" opacity="0.8"/>
      <line x1="190" y1="150" x2="210" y2="150" stroke="#0284C7" stroke-width="1" opacity="0.8"/>
      <line x1="200" y1="140" x2="200" y2="160" stroke="#0284C7" stroke-width="1" opacity="0.8"/>

      <!-- North Compass & Scale Bar -->
      <g transform="translate(368, 22)">
        <circle cx="0" cy="0" r="10" fill="#080C17" stroke="rgba(255,255,255,0.25)"/>
        <path d="M0 -6 L3 3 L0 1 L-3 3 Z" fill="#0284C7"/>
        <text x="-3" y="8" font-size="6" fill="#FFF" font-family="sans-serif" font-weight="bold">N</text>
      </g>
      
      <g transform="translate(315, 282)">
        <rect x="0" y="0" width="45" height="2" fill="#FFF" opacity="0.8"/>
        <text x="10" y="-3" fill="rgba(255,255,255,0.6)" font-size="6" font-family="monospace">500m</text>
      </g>
    `;

    if (type === 'before') {
      return `
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
          <rect width="400" height="300" fill="#0B1A14"/>
          <path d="M0 0 L180 0 L150 140 L0 100 Z" fill="#133323"/>
          <path d="M180 0 L400 0 L400 120 L150 140 Z" fill="#0E261A"/>
          
          <path d="M0 180 Q120 160 220 220 T400 240 L400 300 L0 300 Z" fill="#072B47"/>

          <rect x="50" y="40" width="16" height="12" fill="#334155" stroke="#64748B"/>
          <rect x="80" y="35" width="20" height="14" fill="#334155" stroke="#64748B"/>
          <rect x="65" y="70" width="14" height="14" fill="#334155" stroke="#64748B"/>

          <path d="M220 60 L380 40 L360 160 L200 130 Z" fill="#183A29" stroke="rgba(255,255,255,0.05)"/>
          <path d="M240 160 L390 170 L380 230 L220 200 Z" fill="#143123"/>

          <path d="M100 0 L100 300" stroke="#1E293B" stroke-width="3.5"/>
          <path d="M0 80 Q100 100 400 100" stroke="#1E293B" stroke-width="2.5" fill="none"/>

          ${gisOverlay}
          <text x="12" y="282" fill="rgba(255,255,255,0.6)" font-size="8.5" font-family="monospace">T1 BASELINE SCENE · 2025-10-12</text>
        </svg>
      `;
    }

    if (type === 'after') {
      return `
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
          <rect width="400" height="300" fill="#0B1A14"/>
          <path d="M0 0 L180 0 L150 140 L0 100 Z" fill="#133323"/>
          <path d="M180 0 L400 0 L400 120 L150 140 Z" fill="#0E261A"/>
          
          <path d="M0 180 Q120 160 220 220 T400 240 L400 300 L0 300 Z" fill="#072B47"/>

          <rect x="50" y="40" width="16" height="12" fill="#334155" stroke="#64748B"/>
          <rect x="80" y="35" width="20" height="14" fill="#334155" stroke="#64748B"/>
          <rect x="65" y="70" width="14" height="14" fill="#334155" stroke="#64748B"/>

          <path d="M100 0 L100 300" stroke="#1E293B" stroke-width="3.5"/>
          <path d="M0 80 Q100 100 400 100" stroke="#1E293B" stroke-width="2.5" fill="none"/>
          <path d="M100 100 Q240 115 350 175" stroke="#CBD5E1" stroke-width="3" stroke-dasharray="5 2" fill="none"/>

          <!-- NEW URBAN STRUCTURES (Southern & Eastern Zones) -->
          <g stroke="#DC2626" stroke-width="1.2" fill="#991B1B">
            <rect x="250" y="68" width="22" height="18" rx="1"/>
            <rect x="280" y="62" width="26" height="20" rx="1"/>
            <rect x="315" y="72" width="20" height="24" rx="1"/>
            <rect x="260" y="96" width="30" height="16" rx="1"/>

            <rect x="260" y="165" width="28" height="22" rx="1"/>
            <rect x="295" y="170" width="22" height="20" rx="1"/>
            <rect x="325" y="160" width="32" height="25" rx="1"/>
          </g>

          <path d="M235 52 L350 52 L350 125 L235 125 Z" fill="rgba(217, 119, 6, 0.15)" stroke="#D97706" stroke-width="1.8" stroke-dasharray="3 3"/>
          <path d="M245 148 L370 148 L370 215 L245 215 Z" fill="rgba(217, 119, 6, 0.15)" stroke="#D97706" stroke-width="1.8" stroke-dasharray="3 3"/>

          ${gisOverlay}
          <text x="12" y="282" fill="#D97706" font-size="8.5" font-family="monospace" font-weight="bold">T2 UPDATED SCENE · 2026-03-24</text>
        </svg>
      `;
    }

    if (type === 'changemask') {
      return `
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
          <rect width="400" height="300" fill="#040712"/>

          <path d="M0 180 Q120 160 220 220 T400 240" stroke="#1E293B" stroke-width="1.2" stroke-dasharray="2 2" fill="none"/>

          <g fill="#DC2626" stroke="#D97706" stroke-width="1.5" opacity="0.85">
            <path d="M240 58 L345 58 L340 122 L240 118 Z"/>
            <path d="M250 152 L365 152 L360 212 L250 205 Z"/>
          </g>

          <path d="M235 52 L350 52 L350 125 L235 125 Z" fill="none" stroke="#0284C7" stroke-width="1.5"/>
          <path d="M245 148 L370 148 L370 215 L245 215 Z" fill="none" stroke="#0284C7" stroke-width="1.5"/>

          <text x="245" y="44" fill="#38BDF8" font-size="8" font-family="monospace" font-weight="bold">EASTERN CHANGE ZONE</text>
          <text x="255" y="140" fill="#38BDF8" font-size="8" font-family="monospace" font-weight="bold">SOUTHERN CHANGE ZONE</text>

          ${gisOverlay}
          <text x="12" y="282" fill="#DC2626" font-size="8.5" font-family="monospace" font-weight="bold">DETECTION MASK · SOUTH & EAST ZONES</text>
        </svg>
      `;
    }

    if (type === 'single') {
      return `
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
          <rect width="400" height="300" fill="#0D261B"/>
          
          <path d="M0 0 L120 0 Q100 150 140 300 L0 300 Z" fill="#0284C7"/>
          
          <path d="M140 0 L320 0 L280 140 L130 100 Z" fill="#16A34A" stroke="#15803D" stroke-width="1.5"/>
          <path d="M130 100 L280 140 L240 280 L140 300 Z" fill="#22C55E" stroke="#15803D" stroke-width="1.5"/>

          <path d="M120 0 L140 300" stroke="#94A3B8" stroke-width="3"/>
          <path d="M130 100 L400 120" stroke="#94A3B8" stroke-width="2.5"/>

          <g stroke="#DC2626" stroke-width="1.5" fill="rgba(220, 38, 38, 0.2)">
            <rect x="300" y="40" width="30" height="25"/>
            <rect x="340" y="35" width="25" height="30"/>
            <rect x="310" y="150" width="40" height="35"/>
          </g>

          <text x="300" y="32" fill="#EF4444" font-size="7.5" font-family="sans-serif">BUILDING DETECTED</text>
          <text x="35" y="150" fill="#E0F2FE" font-size="8.5" font-family="sans-serif">WATER BODY</text>
          <text x="180" y="60" fill="#DCFCE7" font-size="8.5" font-family="sans-serif">AGRICULTURE</text>

          ${gisOverlay}
          <text x="12" y="282" fill="#38BDF8" font-size="8.5" font-family="monospace">SINGLE SCENE VQA SEGMENTATION</text>
        </svg>
      `;
    }

    if (type === 'optical') {
      return `
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
          <rect width="400" height="300" fill="#112519"/>
          <path d="M0 50 Q100 80 120 300 L0 300 Z" fill="#072B47"/>
          <path d="M220 50 L380 50 L360 250 L200 250 Z" fill="#184225"/>

          <ellipse cx="280" cy="120" rx="60" ry="35" fill="rgba(255,255,255,0.4)" opacity="0.8"/>
          <ellipse cx="300" cy="110" rx="45" ry="25" fill="rgba(255,255,255,0.6)"/>

          ${gisOverlay}
          <text x="12" y="282" fill="#38BDF8" font-size="8.5" font-family="monospace">OPTICAL (CLOUD INTERFERENCE)</text>
        </svg>
      `;
    }

    if (type === 'sar') {
      return `
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
          <rect width="400" height="300" fill="#0F172A"/>
          
          <path d="M0 50 Q100 80 120 300 L0 300 Z" fill="#020617"/>

          <g fill="#F8FAFC" opacity="0.9">
            <rect x="230" y="80" width="12" height="12"/>
            <rect x="250" y="75" width="15" height="15"/>
            <rect x="280" y="90" width="18" height="12"/>
            <rect x="240" y="130" width="22" height="18"/>
            <rect x="270" y="140" width="20" height="20"/>
            <rect x="300" y="125" width="25" height="15"/>
          </g>

          <text x="230" y="65" fill="#E2E8F0" font-size="7.5" font-family="monospace">SAR DOUBLE-BOUNCE</text>

          ${gisOverlay}
          <text x="12" y="282" fill="#E2E8F0" font-size="8.5" font-family="monospace">SAR (C-BAND RADAR BACKSCATTER)</text>
        </svg>
      `;
    }

    if (type === 'fused') {
      return `
        <svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
          <rect width="400" height="300" fill="#050B18"/>

          <path d="M0 50 Q100 80 120 300 L0 300 Z" fill="#0284C7" fill-opacity="0.8" stroke="#00F0FF" stroke-width="1.5"/>
          
          <path d="M210 60 L340 60 L330 200 L200 180 Z" fill="rgba(217, 119, 6, 0.2)" stroke="#D97706" stroke-width="1.5" stroke-dasharray="3 3"/>
          
          <g fill="#DC2626" stroke="#D97706" stroke-width="1.2">
            <rect x="230" y="80" width="16" height="14"/>
            <rect x="255" y="75" width="18" height="18"/>
            <rect x="280" y="90" width="20" height="14"/>
            <rect x="240" y="130" width="24" height="20"/>
            <rect x="270" y="140" width="22" height="22"/>
          </g>

          <text x="20" y="180" fill="#38BDF8" font-size="8" font-family="monospace">WESTERN WATER ZONE</text>
          <text x="215" y="48" fill="#D97706" font-size="8" font-family="monospace">DENSE BUILT-UP STRUCTURES</text>

          ${gisOverlay}
          <text x="12" y="282" fill="#10B981" font-size="8.5" font-family="monospace" font-weight="bold">OPTICAL+SAR FUSED CLASSIFICATION</text>
        </svg>
      `;
    }

    return '';
  }

  function renderAuditTrail() {
    const mode = state.currentMode;
    const config = flowConfig[mode];
    const steps = config.audit;

    dom.auditFlowTree.innerHTML = steps.map((s, idx) => `
      <div class="audit-step">
        <span class="audit-node">${idx + 1}</span>
        <div class="audit-info">
          <span class="audit-name">${s.title}</span>
          <span class="audit-desc">${s.desc}</span>
        </div>
      </div>
      ${idx < steps.length - 1 ? '<div class="audit-arrow">↓</div>' : ''}
    `).join('');
  }

  function renderReportPreview() {
    const config = flowConfig[state.currentMode];
    const today = new Date().toISOString().split('T')[0];

    const reportText = `======================================================================
SATQUERY AI — SATELLITE INTELLIGENCE REPORT (ISRO/SAC CONCEPT DEMO)
======================================================================
Report Generated: ${today}
Scene Telemetry ID: ${config.sceneId}
Analysis Mode: ${config.modeName}
Coordinate Reference System: EPSG:4326 (WGS84)
Status: Processing completed
Confidence: High

----------------------------------------------------------------------
1. USER QUERY
----------------------------------------------------------------------
"${state.queryText}"

----------------------------------------------------------------------
2. AI ANALYSIS ANSWER
----------------------------------------------------------------------
"${config.aiAnswer}"

----------------------------------------------------------------------
3. INPUT DETAILS & MODALITY
----------------------------------------------------------------------
Inputs: ${config.files.map(f => f.name).join(', ')}
Modality Details: ${config.valModality}
Validation Result: Passed (Format, count, spatial co-registration)

----------------------------------------------------------------------
4. ORCHESTRATED WORKFLOW & TOOLS
----------------------------------------------------------------------
Workflow: ${config.workflowName}
Tools Selected: ${config.metaTools}

----------------------------------------------------------------------
5. AUDIT TRAIL / EXECUTION STEPS
----------------------------------------------------------------------
${config.audit.map((a, i) => `Step ${i + 1}: ${a.title} -> ${a.desc}`).join('\n')}

======================================================================
Notice: SatQuery AI Demonstration Prototype for ISRO / SAC Presentation.
No backend computations, AI model inference, or server APIs connected.
======================================================================`;

    dom.reportPreviewContent.textContent = reportText;
  }

  function downloadReportFile() {
    const textContent = dom.reportPreviewContent.textContent;
    const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SatQuery_AI_Analysis_Report_${state.currentMode}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // -------------------------------------------------------------------------
  // MOBILE MENU CONTROLLER
  // -------------------------------------------------------------------------
  function toggleMobileMenu() {
    dom.sidebar.classList.toggle('mobile-open');
    dom.sidebarOverlay.classList.toggle('mobile-open');
  }

  function closeMobileMenu() {
    dom.sidebar.classList.remove('mobile-open');
    dom.sidebarOverlay.classList.remove('mobile-open');
  }

  // -------------------------------------------------------------------------
  // EVENT LISTENERS BINDING
  // -------------------------------------------------------------------------
  function setupEventListeners() {
    // Mobile Drawer Toggle
    dom.menuToggleBtn.addEventListener('click', toggleMobileMenu);
    dom.sidebarOverlay.addEventListener('click', closeMobileMenu);

    // Navigation Sidebar Buttons
    dom.navItems.forEach(item => {
      item.addEventListener('click', () => {
        const targetScreen = item.getAttribute('data-screen');
        navigateTo(targetScreen);
      });
    });

    // Screen 1: Home
    dom.btnHomeStart.addEventListener('click', () => {
      navigateTo('screen-upload');
    });

    dom.featureCards.forEach(card => {
      card.addEventListener('click', () => {
        const mode = card.getAttribute('data-mode');
        setMode(mode);
        navigateTo('screen-upload');
      });
    });

    // Screen 2: Upload Imagery Mode Switchers
    dom.modeCards.forEach(card => {
      card.addEventListener('click', () => {
        const mode = card.getAttribute('data-mode');
        setMode(mode);
      });
    });

    dom.btnUploadBack.addEventListener('click', () => {
      navigateTo('screen-home');
    });

    dom.btnUploadContinue.addEventListener('click', () => {
      navigateTo('screen-validation');
    });

    // Screen 3: Input Validation Buttons
    dom.btnValidationBack.addEventListener('click', () => {
      navigateTo('screen-upload');
    });

    dom.btnValidationProceed.addEventListener('click', () => {
      navigateTo('screen-query');
    });

    // Screen 4: Ask A Question
    dom.queryTextarea.addEventListener('input', (e) => {
      state.queryText = e.target.value;
      dom.resultUserQuery.textContent = `“${state.queryText}”`;
    });

    dom.btnClearQuery.addEventListener('click', () => {
      dom.queryTextarea.value = '';
      state.queryText = '';
    });

    dom.promptChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const text = chip.getAttribute('data-prompt');
        dom.queryTextarea.value = text;
        state.queryText = text;
        dom.resultUserQuery.textContent = `“${text}”`;

        dom.promptChips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
      });
    });

    dom.btnQueryBack.addEventListener('click', () => {
      navigateTo('screen-validation');
    });

    dom.btnQueryRun.addEventListener('click', () => {
      if (!dom.queryTextarea.value.trim()) {
        state.queryText = flowConfig[state.currentMode].defaultQuery;
        dom.queryTextarea.value = state.queryText;
      }
      navigateTo('screen-analysis');
    });

    // Screen 5: Agentic Analysis
    dom.btnAnalysisBack.addEventListener('click', () => {
      navigateTo('screen-query');
    });

    dom.btnAnalysisViewResults.addEventListener('click', () => {
      navigateTo('screen-results');
    });

    // Screen 6: Results & Audit Trail
    dom.accordionToggle.addEventListener('click', () => {
      state.accordionOpen = !state.accordionOpen;
      if (state.accordionOpen) {
        dom.accordionContent.classList.add('open');
        dom.accordionToggle.classList.add('active');
      } else {
        dom.accordionContent.classList.remove('open');
        dom.accordionToggle.classList.remove('active');
      }
    });

    dom.btnResultsNewAnalysis.addEventListener('click', () => {
      navigateTo('screen-upload');
    });

    dom.btnResultsAskAnother.addEventListener('click', () => {
      navigateTo('screen-query');
    });

    dom.btnResultsDownload.addEventListener('click', () => {
      navigateTo('screen-report');
    });

    // Screen 7: Report
    dom.btnReportNewAnalysis.addEventListener('click', () => {
      navigateTo('screen-upload');
    });

    dom.btnReportDownload.addEventListener('click', downloadReportFile);
  }

  // DOMContentLoaded initialization trigger
  document.addEventListener('DOMContentLoaded', init);

})();
